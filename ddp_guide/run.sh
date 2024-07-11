#!/bin/bash
#SBATCH --partition=brook
#SBATCH --job-name=DDP-test
#SBATCH --output=ddp_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2

# Check if Nvidia SMI is installed

srun torchrun \
--standalone \
--nproc_per_node=2 \
multgpu_singlenode_torchrun.py 50 10