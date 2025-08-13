import os

from llama3 import Transformer, LLAMA_MINI

import torch
from torch.distributed.fsdp import fully_shard


def write_pid(rank):
    pid = os.getpid()
    with open(f"pid_rank{rank}.txt", "w") as f:
        f.write(str(pid))

def main():
    rank = int(os.environ["LOCAL_RANK"])
    write_pid(rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    num_iters = 1
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = LLAMA_MINI
    with torch.device("meta"):
        model = Transformer(model_args)

    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
        
    model.to_empty(device="cuda")
    model.reset_parameters()

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(num_iters):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x, 0).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()