import torch
import torch.distributed as dist
import os

def init_distributed_mode():
    """
    Initialize distributed mode.
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    num_gpus_per_node = torch.cuda.device_count()
    node_rank = rank // num_gpus_per_node  # Calculate node rank

    print("got to here \n")

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    return rank, local_rank, node_rank, world_size

def check_cuda(global_rank, local_rank, node_rank):
    """
    Check CUDA functionality and print node rank and GPU rank.
    """
    device = torch.device(f'cuda:{local_rank}')
    try:
        torch.rand(1).to(device)
        print(f"Node rank: {node_rank}, Global GPU rank: {global_rank}, Local GPU rank: {local_rank} - Success")
    except Exception as e:
        print(f"Node rank: {node_rank}, Global GPU rank: {global_rank}, Local GPU rank: {local_rank} - Failed with error: {e}")

def main():
    """
    Main function to initialize distributed mode and check CUDA.
    """
    global_rank, local_rank, node_rank, _ = init_distributed_mode()
    check_cuda(global_rank, local_rank, node_rank)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
