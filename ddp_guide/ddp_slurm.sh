#!/bin/bash
#SBATCH --partition=brook
#SBATCH --job-name=multinode-example
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass

srun torchrun --nnodes 2 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip multinode_torchrun.py 50 10