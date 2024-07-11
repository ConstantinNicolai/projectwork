#!/bin/bash
#SBATCH --partition=brook
#SBATCH --job-name=DDP-test
#SBATCH --output=ddp_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

# Check if Nvidia SMI is installed

torchrun --standalone --nproc_per_node=4 multgpu_singlenode_torchrun.py 50 10