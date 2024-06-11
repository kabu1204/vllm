import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import nccl_cpu

# server = dist.TCPStore("127.0.0.1", 1234, 2, True)

def gather(rank, size):
    """ Simple gather communication. """
    device = "cuda:0" if rank == 0 else "cpu"
    dtype = torch.float16 if rank == 0 else torch.bfloat16
    pin_memory = True if device == "cpu" else False
    tensor = torch.rand((1, 2), dtype=dtype, device=device, pin_memory=pin_memory)
    print(f"[rank{rank}] {tensor}")
    if rank == 1:
        tensor_list = [torch.ones(1, 2, dtype=dtype, device=device, pin_memory=pin_memory) for _ in range(size)]
    else:
        tensor_list = []
    dist.gather(tensor, tensor_list, dst=1)
    if rank == 1:
        print('Rank ', rank, ' has data ', tensor_list)

def broadcast(rank, size):
    """ Simple broadcast communication. """
    device = "cuda:0" if rank == 0 else "cpu"
    dtype = torch.float16 if rank == 0 else torch.bfloat16
    pin_memory = True if device == "cpu" else False
    tensor = torch.rand((1, 2), dtype=dtype, device=device, pin_memory=pin_memory)
    print(f"[rank{rank}] {tensor}")
    dist.broadcast(tensor, src=1)
    print('Rank ', rank, ' has data ', tensor)

def allreduce(rank, size):
    """ Simple allreduce communication. """
    # group = dist.new_group([0, 1])
    device = "cuda:0" if rank == 0 else "cpu"
    dtype = torch.float16 if rank == 0 else torch.bfloat16
    pin_memory = True if device == "cpu" else False
    tensor = torch.rand((1, 2), dtype=dtype, device=device, pin_memory=pin_memory)
    print(f"[rank{rank}] {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor)

def allgather(rank, size):
    """ Simple allgather communication. """
    # group = dist.new_group([0, 1])
    device = "cuda:0" if rank == 0 else "cpu"
    dtype = torch.float16 if rank == 0 else torch.bfloat16
    pin_memory = True if device == "cpu" else False
    tensor_list = [rank*torch.zeros(2, dtype=dtype, device=device, pin_memory=pin_memory) for _ in range(2)]
    tensor = torch.rand((1,2), dtype=dtype, device=device, pin_memory=pin_memory)
    print(f"[rank{rank}] {tensor}")
    dist.all_gather(tensor_list, tensor)
    print('Rank ', rank, ' has data ', tensor_list)

def prefixStore(rank, size):
    tensor = rank*torch.ones(1)
    store = dist.TCPStore("127.0.0.1", 1234, 2, rank == 0)
    if rank == 0:
        store.set("tensor", "ok")
        while(1):
            pass
    else:
        x = store.get("tensor")
        print(x)

def run(rank, size):
    gather(rank, size)
    # broadcast(rank, size)
    # allreduce(rank, size)
    # allgather(rank, size)

def init_process(rank, size, fn, backend='nccl-cpu'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    os.environ["VLLM_CPU_WORKER_NUM"] = "1"
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()