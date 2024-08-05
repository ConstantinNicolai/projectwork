#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=smi_meas
#SBATCH --output=rolling_output_nojobnumber.out
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:2



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


# Check if Nvidia SMI is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: Nvidia SMI is not installed on this node."
    exit 1
fi

# Function to read GPU model
read_gpu_model() {
    gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "GPU Model: $gpu_model"
}


kill_background_jobs() {
    for pid in $@; do
        kill $pid
    done
}


# Main script

read_gpu_model

gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })
for gpu_id in "${gpu_ids[@]}"; do
nvidia-smi -i ${gpu_id} -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> logs/gpu_usage_node${SLURM_NODEID}_gpu${gpu_id}.log &
done


# srun log_gpu_usage &  # Run the logging function in the background

#sleep 20
# Run the benchmark
srun torchrun resnet_multi.py >> logs/training_output_${SLURM_JOB_ID}.log

#kill of background logging
bg_pids=$(jobs -p)
kill_background_jobs $bg_pids
