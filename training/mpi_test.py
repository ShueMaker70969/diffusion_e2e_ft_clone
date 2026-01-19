import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] initialized (world_size={world_size})")

    x = torch.tensor([rank], device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    print(f"[Rank {rank}] all_reduce result = {x.item()}")

    dist.barrier()
    print(f"[Rank {rank}] finished")

if __name__ == "__main__":
    main()
