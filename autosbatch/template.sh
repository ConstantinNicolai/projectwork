#!/bin/bash
#SBATCH --partition=all
#SBATCH --job-name=autosbatch
#SBATCH --output=output_PLACEHOLDER_GPU_MODEL_PLACEHOLDER_GPU_COUNT.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=PLACEHOLDER_GPU_COUNT
#SBATCH --gres=gpu:PLACEHOLDER_GPU_MODEL:PLACEHOLDER_GPU_COUNT



MODEL_NAME=resnet18
BATCH_SIZE=32
GPU_MODEL=PLACEHOLDER_GPU_MODEL
NUM_GPUS=PLACEHOLDER_GPU_COUNT

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


# Construct the folder name using the model name, batch size, and number of GPUs
FOLDER_NAME="logs/${MODEL_NAME}_${BATCH_SIZE}_${GPU_MODEL}${NUM_GPUS}"

# Check if the folder already exists, if not, create it
if [ ! -d "$FOLDER_NAME" ]; then
  mkdir -p "$FOLDER_NAME"
  echo "Directory $FOLDER_NAME created."
else
  echo "Directory $FOLDER_NAME already exists."
fi


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
nvidia-smi -i ${gpu_id} -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> $FOLDER_NAME/gpu_usage_node${SLURM_NODEID}_gpu${gpu_id}.log &
done


# srun log_gpu_usage &  # Run the logging function in the background

# Run the benchmark
srun torchrun ./../resnet_multi.py >> $FOLDER_NAME/training_output_${SLURM_JOB_ID}.log

#kill of background logging
bg_pids=$(jobs -p)
kill_background_jobs $bg_pids