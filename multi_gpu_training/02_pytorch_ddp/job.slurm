#!/bin/bash
#SBATCH --job-name=princtionddp     # create a short name for your job
#SBATCH --partition=brook
#SBATCH --output=prince_%j.out
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of allocated gpus per node

export MASTER_PORT=1026
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python mnist_classify_ddp.py --epochs=2
