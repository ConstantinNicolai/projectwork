#!/bin/bash
#SBATCH --job-name=autosbatch
#SBATCH --output=output_PLACEHOLDER_GPU_MODEL_PLACEHOLDER_GPU_COUNT.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=PLACEHOLDER_GPU_COUNT
#SBATCH --gres=gpu:PLACEHOLDER_GPU_MODEL:PLACEHOLDER_GPU_COUNT


# Your actual job commands go here
echo "Running job with GPU model: PLACEHOLDER_GPU_MODEL and GPU count: PLACEHOLDER_GPU_COUNT"
