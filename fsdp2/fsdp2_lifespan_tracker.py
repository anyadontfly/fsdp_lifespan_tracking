import os
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, NamedTuple, Optional
import logging

import torch
from torch.distributed.fsdp import FSDPModule
from torch.utils.weak import WeakIdKeyDictionary, weakref
import torch.cuda.nvtx as nvtx


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


class _NamedFSDPModuleWeakRef(NamedTuple):
    mod_ref: weakref.ref
    name: str


class _SavedFSDPMethods(NamedTuple):
    pre_backward: Callable
    post_backward: Callable


class _FSDPHookState(Enum):
    PRE_FW = "pre-forward"
    POST_FW = "post-forward"
    PRE_BW = "pre-backward"
    POST_BW = "post-backward"


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

    @property
    def finished(self) -> bool:
        return self.bwd_reduce_scatter_event is not None

    @property
    def is_root(self) -> bool:
        return self.finished and self.bwd_all_gather_event is None


class FSDPLifespanTracker:
    def __init__(
        self,
        root_mod: torch.nn.Module,
        rank: int,
        num_warmup_steps: int = 0,
    ):
        self.root_mod = root_mod
        self.rank = rank
        self.num_warmup_steps = num_warmup_steps
        self.mod_to_timing_events: Dict[_NamedFSDPModuleWeakRef, _FSDPTimingEvents] = {}
        self.mod_to_saved_hooks: WeakIdKeyDictionary = WeakIdKeyDictionary()
        self.steps = None

    def _wrapped_hook(
        self,
        named_fsdp_weak_ref: _NamedFSDPModuleWeakRef,
        orig_fsdp_state_hook: Callable,
        hook_state: _FSDPHookState,
    ):
        @wraps(orig_fsdp_state_hook)
        def inner(*args, **kwargs):
            fsdp_mod, _ = named_fsdp_weak_ref.mod_ref, named_fsdp_weak_ref.name
            if hook_state == _FSDPHookState.PRE_FW:
                args, kwargs = orig_fsdp_state_hook(*args, **kwargs)
                all_gather_stream = fsdp_mod()._get_fsdp_state()._comm_ctx.all_gather_stream
                
                fwd_all_gather_event = torch.cuda.Event(enable_timing=True)
                nvtx.mark("FSDP pre-forward all-gather")
                fwd_all_gather_event.record(all_gather_stream)
                self.mod_to_timing_events[named_fsdp_weak_ref].fwd_all_gather_event = fwd_all_gather_event
                return args, kwargs
            elif hook_state == _FSDPHookState.POST_FW:
                output = orig_fsdp_state_hook(*args, **kwargs)
                fsdp_mod_reshard_after_fwd = fsdp_mod()._get_fsdp_state()._fsdp_param_group._reshard_after_forward

                if fsdp_mod_reshard_after_fwd:
                    fwd_reshard_event = torch.cuda.Event(enable_timing=True)
                    nvtx.mark("FSDP post-forward reshard")
                    fwd_reshard_event.record()
                    self.mod_to_timing_events[named_fsdp_weak_ref].fwd_reshard_event = fwd_reshard_event
                return output
            elif hook_state == _FSDPHookState.PRE_BW:
                orig_fsdp_state_hook(*args, **kwargs)
                fsdp_mod_reshard_after_fwd = fsdp_mod()._get_fsdp_state()._fsdp_param_group._reshard_after_forward

                if fsdp_mod_reshard_after_fwd:
                    all_gather_stream = fsdp_mod()._get_fsdp_state()._comm_ctx.all_gather_stream
                    bwd_all_gather_event = torch.cuda.Event(enable_timing=True)
                    nvtx.mark("FSDP pre-backward all-gather")
                    bwd_all_gather_event.record(all_gather_stream)
                    self.mod_to_timing_events[named_fsdp_weak_ref].bwd_all_gather_event = bwd_all_gather_event
            elif hook_state == _FSDPHookState.POST_BW:
                orig_fsdp_state_hook(*args, **kwargs)
                bwd_reshard_event = torch.cuda.Event(enable_timing=True)
                nvtx.mark("FSDP post-backward reshard")
                bwd_reshard_event.record()

                reduce_scatter_stream = fsdp_mod()._get_fsdp_state()._comm_ctx.reduce_scatter_stream
                bwd_reduce_scatter_event = torch.cuda.Event(enable_timing=True)
                nvtx.mark("FSDP post-backward reduce-scatter")
                bwd_reduce_scatter_event.record(reduce_scatter_stream)

                self.mod_to_timing_events[named_fsdp_weak_ref].bwd_reshard_event = bwd_reshard_event
                self.mod_to_timing_events[named_fsdp_weak_ref].bwd_reduce_scatter_event = bwd_reduce_scatter_event
            
        return inner
    
    def _register_module_hooks(self) -> None:
        for fsdp_mod_name, fsdp_mod in self.root_mod.named_modules():
            if not isinstance(fsdp_mod, FSDPModule):
                continue
            if (fsdp_state := fsdp_mod._get_fsdp_state()) is None:
                continue
            named_fsdp_weak_ref = _NamedFSDPModuleWeakRef(
                weakref.ref(fsdp_mod),
                "root module" if fsdp_mod_name == "" else fsdp_mod_name
            )
            self.mod_to_timing_events[named_fsdp_weak_ref] = _FSDPTimingEvents()
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                fsdp_state._pre_forward_hook_handle.remove()
                fsdp_state._post_forward_hook_handle.remove()
                fsdp_state._pre_forward_hook_handle = (
                    fsdp_mod.register_forward_pre_hook(
                        self._wrapped_hook(
                            named_fsdp_weak_ref,
                            fsdp_state._pre_forward,
                            _FSDPHookState.PRE_FW
                        ),
                        prepend=True,
                        with_kwargs=True,
                    )
                )
                fsdp_state._post_forward_hook_handle = (
                    fsdp_mod.register_forward_hook(
                        self._wrapped_hook(
                            named_fsdp_weak_ref,
                            fsdp_state._post_forward,
                            _FSDPHookState.POST_FW
                        ),
                        prepend=False,
                        always_call=True,
                    )
                )
                self.mod_to_saved_hooks[fsdp_mod] = _SavedFSDPMethods(
                    fsdp_param_group.pre_backward,
                    fsdp_param_group.post_backward,
                )
                fsdp_param_group.pre_backward = self._wrapped_hook(
                    named_fsdp_weak_ref,
                    fsdp_param_group.pre_backward,
                    _FSDPHookState.PRE_BW
                )
                fsdp_param_group.post_backward = self._wrapped_hook(
                    named_fsdp_weak_ref,
                    fsdp_param_group.post_backward,
                    _FSDPHookState.POST_BW
                )
            
    def _deregister_module_hooks(self) -> None:
        for (
            fsdp_mod,
            saved_methods,
        ) in self.mod_to_saved_hooks.items():
            fsdp_state = fsdp_mod._get_fsdp_state()
            fsdp_state._pre_forward_hook_handle.remove()
            fsdp_state._post_forward_hook_handle.remove()
            fsdp_state._pre_forward_hook_handle = fsdp_mod.register_forward_pre_hook(
                fsdp_state._pre_forward, prepend=True, with_kwargs=True
            )
            fsdp_state._post_forward_hook_handle = fsdp_mod.register_forward_hook(
                fsdp_state._post_forward, prepend=False
            )
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                fsdp_param_group.pre_backward = saved_methods.pre_backward
                fsdp_param_group.post_backward = saved_methods.post_backward
        self.mod_to_saved_hooks.clear()

    def step(self) -> None:
        if self.steps is None:
            raise RuntimeError("Tracker must be used within a context manager.")
        if self.steps >= self.num_warmup_steps:
            self.summarize_lifespans()
            self.logger.info(f"============= Rank {self.rank} completed step {self.steps} =============\n")
        self.clear_timing_events()
        self.steps += 1
    
    def summarize_lifespans(self) -> None:
        for named_fsdp_weak_ref, timing_events in self.mod_to_timing_events.items():
            if not timing_events.finished:
                continue
            timing_events.sync_on_events()
            self.logger.info(f"Module: {named_fsdp_weak_ref.name}")
            if timing_events.is_root:
                self.logger.info(f"Rank {self.rank} root params lifespan: {timing_events.get_root_params_lifespan()} ms")
                self.logger.info(f"Rank {self.rank} root grads lifespan: {timing_events.get_root_grads_lifespan()} ms")
            else:
                self.logger.info(f"Rank {self.rank} fwd params lifespan: {timing_events.get_fwd_params_lifespan()} ms")
                self.logger.info(f"Rank {self.rank} bwd params lifespan: {timing_events.get_bwd_params_lifespan()} ms")
                self.logger.info(f"Rank {self.rank} bwd grads lifespan: {timing_events.get_bwd_grads_lifespan()} ms")

    def clear_timing_events(self) -> None:
        for timing_events in self.mod_to_timing_events.values():
            timing_events.clear()
    
    def __enter__(self) -> "FSDPLifespanTracker":
        self.steps = 0
        self.logger = setup_logger(self.rank)
        self._register_module_hooks()
        return self

    def __exit__(self, *args: Any) -> None:
        self.steps = None
        self._deregister_module_hooks()
        self.mod_to_saved_hooks.clear()
        self.mod_to_timing_events.clear()
        self.logger = None
