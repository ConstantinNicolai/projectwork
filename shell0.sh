#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 MODEL_NAME BATCH_SIZE GPU_MODEL NUM_GPUS"
  exit 1
fi

# Variables for the model name, batch size, GPU model, and number of GPUs
MODEL_NAME=$1
BATCH_SIZE=$2
GPU_MODEL=$3
NUM_GPUS=$4

# SBATCH Directives
#SBATCH --job-name=create-folder-job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=all
#SBATCH --gres=gpu:${GPU_MODEL}:${NUM_GPUS}


# Construct the folder name using the model name, batch size, and number of GPUs
FOLDER_NAME="logs/${MODEL_NAME}_${BATCH_SIZE}_${GPU_MODEL}${NUM_GPUS}"

# Check if the folder already exists, if not, create it
if [ ! -d "$FOLDER_NAME" ]; then
  mkdir -p "$FOLDER_NAME"
  echo "Directory $FOLDER_NAME created."
else
  echo "Directory $FOLDER_NAME already exists."
fi


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


# Function to log GPU usage for each GPU on the node
log_gpu_usage() {
  local gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })
  for gpu_id in "${gpu_ids[@]}"; do
    nvidia-smi -i ${gpu_id} -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> $FOLDER_NAME/gpu_usage_node${SLURM_NODEID}_gpu${gpu_id}.log &
  done
}


# Main script

read_gpu_model

gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })
for gpu_id in "${gpu_ids[@]}"; do
nvidia-smi -i ${gpu_id} -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> $FOLDER_NAME/gpu_usage_node${SLURM_NODEID}_gpu${gpu_id}.log &
done


# srun log_gpu_usage &  # Run the logging function in the background

# Run the benchmark
srun torchrun resnet_multi.py >> $FOLDER_NAME/training_output_${SLURM_JOB_ID}.log

#kill of background logging
bg_pids=$(jobs -p)
kill_background_jobs $bg_pids