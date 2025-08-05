import os
import argparse
import torch
from llama3 import Transformer, LLAMA_8B

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

import wrapping
import fsdp_config as config
import time


def fsdp_main(args):

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

    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    #     for _ in range(5):
    #         prof.step()
    #         x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    #         loss = model(x, 0).sum()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optim.step()
    #         optim.zero_grad()
    #         time.sleep(0.5)

    for _ in range(5):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x, 0).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()
        time.sleep(0.5)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch llama3 FSDP Example')
    args = parser.parse_args()

    fsdp_main(args)