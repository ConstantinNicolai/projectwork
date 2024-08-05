import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the logs
log_dir = 'logs'

# Define the intervals for plotting
intervals = [1, 5, 10]  # in ms

# Function to create plots
def create_plots(data, model, batch_size, gpu_model, gpu_id, output_dir):
    timestamps = pd.to_datetime(data.iloc[:, 0])
    utilization = data.iloc[:, 1]
    power_draw = data.iloc[:, 2]
    memory_used = data.iloc[:, 3]
    memory_total = data.iloc[:, 4]
    normalized_memory_used = memory_used / memory_total

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Utilization over time
    plt.figure()
    plt.plot(timestamps, utilization, label='Utilization')
    plt.xlabel('Time')
    plt.ylabel('GPU Utilization (%)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_{gpu_id}_utilization_over_time')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_{gpu_id}_utilization_over_time.png'))
    plt.close()

    # Power draw over time
    plt.figure()
    plt.plot(timestamps, power_draw, label='Power Draw')
    plt.xlabel('Time')
    plt.ylabel('Power Draw (W)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_{gpu_id}_power_draw_over_time')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_{gpu_id}_power_draw_over_time.png'))
    plt.close()

    # Power draw over utilization
    plt.figure()
    plt.scatter(utilization, power_draw)
    plt.xlabel('GPU Utilization (%)')
    plt.ylabel('Power Draw (W)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_{gpu_id}_power_draw_over_utilization')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_{gpu_id}_power_draw_over_utilization.png'))
    plt.close()

    # Memory usage over time
    plt.figure()
    plt.plot(timestamps, normalized_memory_used, label='Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (Normalized)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_{gpu_id}_memory_usage_over_time')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_{gpu_id}_memory_usage_over_time.png'))
    plt.close()

    # Memory usage over utilization
    plt.figure()
    plt.scatter(utilization, normalized_memory_used)
    plt.xlabel('GPU Utilization (%)')
    plt.ylabel('Memory Usage (Normalized)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_{gpu_id}_memory_usage_over_utilization')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_{gpu_id}_memory_usage_over_utilization.png'))
    plt.close()

    # Power draw over memory usage
    plt.figure()
    plt.scatter(normalized_memory_used, power_draw)
    plt.xlabel('Memory Usage (Normalized)')
    plt.ylabel('Power Draw (W)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_{gpu_id}_power_draw_over_memory_usage')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_{gpu_id}_power_draw_over_memory_usage.png'))
    plt.close()

# Function to process a single log file
def process_log_file(filepath):
    return pd.read_csv(filepath, header=None)

# Function to traverse directories and process logs
def traverse_and_plot(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.log') and 'gpu_usage' in file:
                # Extract model, batch_size, gpu_model, and gpu_id from the directory structure
                parts = root.split(os.sep)
                model_batch_gpu = parts[-1]
                model = model_batch_gpu.split('_')[0]
                batch_size = model_batch_gpu.split('_')[1]
                gpu_model = model_batch_gpu.split('_')[2][:-1]
                gpu_id = model_batch_gpu.split('_')[2][-1]

                # Process the log file
                log_filepath = os.path.join(root, file)
                data = process_log_file(log_filepath)
                
                # Create output directory for plots
                output_dir = os.path.join(root, 'plots')
                
                # Generate plots
                create_plots(data, model, batch_size, gpu_model, gpu_id, output_dir)


# Run the function
traverse_and_plot(log_dir)
