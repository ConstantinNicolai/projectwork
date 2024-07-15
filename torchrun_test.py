import torch
import torch.distributed as dist
import os

def init_distributed_mode():
    """
    Initialize distributed mode.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        print('Not using distributed mode')
        rank = 0
        world_size = 1

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    return rank, world_size

def check_cuda(rank):
    """
    Check CUDA functionality and print node rank and GPU rank.
    """
    num_gpus = torch.cuda.device_count()
    for gpu in range(num_gpus):
        device = torch.device(f'cuda:{gpu}')
        try:
            torch.rand(1).to(device)
            print(f"Node rank: {rank}, GPU rank: {gpu} - Success")
        except Exception as e:
            print(f"Node rank: {rank}, GPU rank: {gpu} - Failed with error: {e}")

def main():
    """
    Main function to initialize distributed mode and check CUDA.
    """
    rank, _ = init_distributed_mode()
    check_cuda(rank)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
