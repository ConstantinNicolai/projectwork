#!/bin/bash

# Define the lists of GPU counts and models
gpu_counts=(1 3)
gpu_models=("TitanX" "RTX2080TI")

# Create a log file
log_file="submission_log.txt"
echo "Submission log - $(date)" > $log_file

# Loop through all combinations of GPU counts and models
for gpu_count in "${gpu_counts[@]}"; do
  for gpu_model in "${gpu_models[@]}"; do
    # Define the script name for the current combination
    script_name="job_gpu_${gpu_count}_model_${gpu_model}.sh"
    
    # Copy the template script to a new file
    cp template.sh $script_name
    
    # Replace the placeholders with the current combination values
    sed -i "s/PLACEHOLDER_GPU_COUNT/${gpu_count}/g" $script_name
    sed -i "s/PLACEHOLDER_GPU_MODEL/${gpu_model}/g" $script_name
    
    # Submit the job and log the result
    sbatch_output=$(sbatch $script_name 2>&1)
    if [[ $? -eq 0 ]]; then
      echo "Submitted job with GPU model: $gpu_model and GPU count: $gpu_count - $sbatch_output" >> $log_file
    else
      echo "Failed to submit job with GPU model: $gpu_model and GPU count: $gpu_count - $sbatch_output" >> $log_file
    fi
    
    # Optionally, remove the temporary script after submission
    rm $script_name
  done
done
