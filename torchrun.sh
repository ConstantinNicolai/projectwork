#!/bin/bash
#SBATCH --job-name=torchrun     # create a short name for your job
#SBATCH --partition=brook
#SBATCH --output=torchrun.out
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of allocated gpus per node


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


# torchrun --nnodes=2 --nproc_per_node=1 torchrun_test.py

torchrun --nnodes=2 --nproc-per-node=1 --max-restarts=2 --rdzv-id=5634 --rdzv-backend=c10d --rdzv-endpoint=HOST_NODE_ADDR  torchrun_test.py
