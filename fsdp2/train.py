import os
import time
import argparse

from llama3 import Transformer, LLAMA_8B
from fsdp2_lifespan_tracker import FSDPLifespanTracker

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


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
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
        
    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    model.to_empty(device="cuda")
    model.reset_parameters()

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train_step():
        if args.explicit_prefetching:
            model.unshard()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x, 0).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

    tracker = FSDPLifespanTracker(model, rank, num_warmup_steps=1)

    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    )
    
    if args.profile:
        with prof:
            with tracker:
                for _ in range(5):
                    prof.step()
                    train_step()
                    tracker.step()
                    time.sleep(0.5)
    else:
        with tracker:
            for _ in range(5):
                train_step()
                tracker.step()
                time.sleep(0.5)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()
    main(args)