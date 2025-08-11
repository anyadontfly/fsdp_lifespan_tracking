import logging
import os
from typing import Any, Callable, Optional, no_type_check

import torch
import torch.nn as nn
from torch.distributed.fsdp._runtime_utils import (
    _post_backward_hook,
    _post_forward,
    _post_backward_final_callback,
    _unshard,
)
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._flat_param import (
    FlatParamHandle,
    HandleTrainingState,
)
import torch.cuda.nvtx as nvtx
from torch.utils.weak import WeakIdKeyDictionary


def setup_logger(rank, log_dir="./fsdp_module_lifespan_logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_rank{rank}.txt")
    
    logger = logging.getLogger(f"logger_rank{rank}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger(int(os.environ["LOCAL_RANK"]))


class _FSDPTimingEvents:
    def __init__(self):
        self.fwd_all_gather_event: Optional[torch.cuda.Event] = None
        self.fwd_reshard_event: Optional[torch.cuda.Event] = None
        self.bwd_all_gather_event: Optional[torch.cuda.Event] = None
        self.bwd_reshard_event: Optional[torch.cuda.Event] = None
        self.bwd_reduce_scatter_event: Optional[torch.cuda.Event] = None
        self.synced: bool = False

    def sync_on_events(self) -> bool:
        if not self.finished or self.bwd_reduce_scatter_event is None:
            return False
        self.bwd_reduce_scatter_event.synchronize()
        self.synced = True
        return True

    def get_fwd_params_lifespan(self) -> Optional[float]:
        if not self.synced or self.is_root:
            return None
        if self.fwd_reshard_event is None or self.fwd_all_gather_event is None:
            return None
        return self.fwd_all_gather_event.elapsed_time(self.fwd_reshard_event)

    def get_bwd_params_lifespan(self) -> Optional[float]:
        if not self.synced or self.is_root:
            return None
        if self.bwd_reshard_event is None or self.bwd_all_gather_event is None:
            return None
        return self.bwd_all_gather_event.elapsed_time(self.bwd_reshard_event)

    def get_bwd_grads_lifespan(self) -> Optional[float]:
        if not self.synced or self.is_root:
            return None
        if self.bwd_all_gather_event is None or self.bwd_reduce_scatter_event is None:
            return None
        return self.bwd_all_gather_event.elapsed_time(self.bwd_reduce_scatter_event)

    def get_root_params_lifespan(self) -> Optional[float]:
        if not self.synced or not self.is_root:
            return None
        if self.fwd_reshard_event is None or self.fwd_all_gather_event is None:
            return None
        return self.fwd_all_gather_event.elapsed_time(self.bwd_reshard_event)

    def get_root_grads_lifespan(self) -> Optional[float]:
        if not self.synced or not self.is_root:
            return None
        if self.fwd_all_gather_event is None or self.bwd_reduce_scatter_event is None:
            return None
        return self.fwd_all_gather_event.elapsed_time(self.bwd_reduce_scatter_event)
    
    def clear(self) -> None:
        self.fwd_all_gather_event = None
        self.fwd_reshard_event = None
        self.bwd_all_gather_event = None
        self.bwd_reshard_event = None
        self.bwd_reduce_scatter_event = None
        self.synced = False

    def __repr__(self):
        return (f"_FSDPTimingEvents(fwd_all_gather_event={self.fwd_all_gather_event is not None}, "
                f"fwd_reshard_event={self.fwd_reshard_event is not None}, "
                f"bwd_all_gather_event={self.bwd_all_gather_event is not None}, "
                f"bwd_reshard_event={self.bwd_reshard_event is not None}, "
                f"bwd_reduce_scatter_event={self.bwd_reduce_scatter_event is not None}, "
                f"synced={self.synced})")

    @property
    def finished(self) -> bool:
        return self.bwd_reduce_scatter_event is not None

    @property
    def is_root(self) -> bool:
        return self.finished and self.bwd_all_gather_event is None


timing_events = WeakIdKeyDictionary()


@no_type_check
def _wrapped_post_forward(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
    reshard_fn: Callable,
    module: nn.Module,
    input: Any,
    output: Any,
) -> Any:
    output = _post_forward(
        state,
        handle,
        reshard_fn,
        module,
        input,
        output
    )

    if not state._is_root:
        default_stream = state._default_stream
        handle_module = handle._fully_sharded_module
        fwd_reshard_event = torch.cuda.Event(enable_timing=True)
        nvtx.mark("FSDP post-forward reshard")
        fwd_reshard_event.record(default_stream)
        if not timing_events.get(handle_module):
            timing_events[handle_module] = _FSDPTimingEvents()
        timing_events[handle_module].fwd_reshard_event = fwd_reshard_event

    return output


@no_type_check
@torch.no_grad()
def _wrapped_post_backward_hook(
    state: _FSDPState,
    handle: FlatParamHandle,
    flat_param,
    *unused: Any,
):
    _post_backward_hook(
        state,
        handle,
        flat_param,
        *unused
    )

    default_stream = state._default_stream
    reduce_scatter_stream = state._post_backward_stream

    handle_module = handle._fully_sharded_module

    bwd_reshard_event = torch.cuda.Event(enable_timing=True)
    nvtx.mark("FSDP post-backward reshard")
    bwd_reshard_event.record(default_stream)

    bwd_reduce_scatter_event = torch.cuda.Event(enable_timing=True)
    nvtx.mark("FSDP post-backward reduce-scatter")
    bwd_reduce_scatter_event.record(reduce_scatter_stream)

    if not timing_events.get(handle_module):
        timing_events[handle_module] = _FSDPTimingEvents()
    timing_events[handle_module].bwd_reshard_event = bwd_reshard_event
    timing_events[handle_module].bwd_reduce_scatter_event = bwd_reduce_scatter_event


@no_type_check
def _wrapped_unshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
) -> None:
    _unshard(
        state,
        handle,
        unshard_stream,
        pre_unshard_stream,
    )

    all_gather_stream = state._unshard_stream
    handle_module = handle._fully_sharded_module

    if handle._training_state == HandleTrainingState.FORWARD:
        fwd_all_gather_event = torch.cuda.Event(enable_timing=True)
        nvtx.mark("FSDP pre-forward all-gather")
        fwd_all_gather_event.record(all_gather_stream)
        if not timing_events.get(handle_module):
            timing_events[handle_module] = _FSDPTimingEvents()
        timing_events[handle_module].fwd_all_gather_event = fwd_all_gather_event
    elif handle._training_state == HandleTrainingState.BACKWARD_PRE:
        bwd_all_gather_event = torch.cuda.Event(enable_timing=True)
        nvtx.mark("FSDP pre-backward all-gather")
        bwd_all_gather_event.record(all_gather_stream)
        if not timing_events.get(handle_module):
            timing_events[handle_module] = _FSDPTimingEvents()
        timing_events[handle_module].bwd_all_gather_event = bwd_all_gather_event


@no_type_check
@torch.no_grad()
def _wrapped_post_backward_final_callback(
    state: _FSDPState,
    module: nn.Module,
):

    _post_backward_final_callback(
        state,
        module,
    )

    if state._is_root:
        rank = int(os.environ["LOCAL_RANK"])
        for fsdp_state in state._all_fsdp_states:
            if state == fsdp_state:
                timing_events[fsdp_state._fsdp_wrapped_module].bwd_all_gather_event = None
            if timing_events[fsdp_state._fsdp_wrapped_module].finished:
                timing_events[fsdp_state._fsdp_wrapped_module].sync_on_events()
            logger.info(f"Module: {fsdp_state._fsdp_wrapped_module.__class__.__name__}")
            if timing_events[fsdp_state._fsdp_wrapped_module].is_root:
                logger.info(f"Rank {rank} root params lifespan: {timing_events[fsdp_state._fsdp_wrapped_module].get_root_params_lifespan()} ms")
                logger.info(f"Rank {rank} root grads lifespan: {timing_events[fsdp_state._fsdp_wrapped_module].get_root_grads_lifespan()} ms")
            else:
                logger.info(f"Rank {rank} fwd params lifespan: {timing_events[fsdp_state._fsdp_wrapped_module].get_fwd_params_lifespan()} ms")
                logger.info(f"Rank {rank} bwd params lifespan: {timing_events[fsdp_state._fsdp_wrapped_module].get_bwd_params_lifespan()} ms")
                logger.info(f"Rank {rank} bwd grads lifespan: {timing_events[fsdp_state._fsdp_wrapped_module].get_bwd_grads_lifespan()} ms")

        logger.info(f"================ Rank {rank} completed step ================\n")
        timing_events.clear()
