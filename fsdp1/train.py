import os
import argparse
import time

import torch
import torch.distributed.fsdp._runtime_utils as _runtime_utils
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import wrapping
import fsdp_config as config
from llama3 import Transformer, LLAMA_8B
from _fsdp_tracking_utils import (
    _wrapped_unshard,
    _wrapped_post_backward_hook,
    _wrapped_post_backward_final_callback,
    _wrapped_post_forward,
)


def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = LLAMA_8B

    _runtime_utils._unshard = _wrapped_unshard
    _runtime_utils._post_backward_hook = _wrapped_post_backward_hook
    _runtime_utils._post_backward_final_callback = _wrapped_post_backward_final_callback
    fsdp_module._post_forward = _wrapped_post_forward

    transformer_auto_wrap_policy = wrapping.get_transformer_wrapper()
    model = Transformer(model_args).to_empty(device="meta")
    model = FSDP(
        model,
        auto_wrap_policy=transformer_auto_wrap_policy,
        sharding_strategy=config.fsdp_config.sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=config.fsdp_config.limit_all_gathers
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train_step():
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x, 0).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    )
    
    if args.profile:
        with prof:
            for _ in range(5):
                prof.step()
                train_step()
                time.sleep(0.5)
    else:
        for _ in range(5):
            train_step()
            time.sleep(0.5)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch llama3 FSDP Example')
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()
    main(args)