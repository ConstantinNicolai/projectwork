#!/bin/bash

#SBATCH --job-name=create-folder-job
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Variables for the model name and batch size
MODEL_NAME="your_model_name"
BATCH_SIZE=32

# Construct the folder name
FOLDER_NAME="${MODEL_NAME}_batch${BATCH_SIZE}"

# Check if the folder already exists, if not, create it
if [ ! -d "$FOLDER_NAME" ]; then
  mkdir -p "$FOLDER_NAME"
  echo "Directory $FOLDER_NAME created."
else
  echo "Directory $FOLDER_NAME already exists."
fi
