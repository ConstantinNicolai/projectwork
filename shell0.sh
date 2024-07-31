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
FOLDER_NAME="${MODEL_NAME}_${BATCH_SIZE}_${GPU_MODEL}${NUM_GPUS}"

# Check if the folder already exists, if not, create it
if [ ! -d "$FOLDER_NAME" ]; then
  mkdir -p "$FOLDER_NAME"
  echo "Directory $FOLDER_NAME created."
else
  echo "Directory $FOLDER_NAME already exists."
fi
