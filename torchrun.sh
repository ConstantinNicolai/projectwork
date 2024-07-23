#!/bin/bash
#SBATCH --job-name=torchrun     # create a short name for your job
#SBATCH --partition=brook
#SBATCH --output=torchrun.out
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --gres=gpu:4             # number of allocated gpus per node


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


torchrun --nnodes=2 --nproc-per-node=4  torchrun_test.py --max-restarts=2 --rdzv-id=5634 --rdzv-backend=c10d --rdzv-endpoint=147.142.43.167
