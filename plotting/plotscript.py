import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define the directory containing the logs
log_dir = 'logs'

# Function to create plots
def create_plots(data, model, batch_size, gpu_model, gpu_id, output_dir):
    timestamps = pd.to_datetime(data.iloc[:, 0])
    utilization = data.iloc[:, 1].replace('[N/A]', 0).astype(float)
    power_draw = data.iloc[:, 2].replace('[N/A]', 0).astype(float)
    memory_used = data.iloc[:, 3].replace('[N/A]', 0).astype(float)
    memory_total = data.iloc[:, 4].iloc[0]  # Assuming total memory is constant
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

# Function to create aggregate plots
def create_aggregate_plots(data_list, model, batch_size, gpu_model, output_dir):
    combined_data = pd.concat(data_list)
    timestamps = pd.to_datetime(combined_data.iloc[:, 0])
    utilization = combined_data.iloc[:, 1].replace('[N/A]', 0).astype(float)
    power_draw = combined_data.iloc[:, 2].replace('[N/A]', 0).astype(float)
    memory_used = combined_data.iloc[:, 3].replace('[N/A]', 0).astype(float)
    memory_total = combined_data.iloc[:, 4].iloc[0]  # Assuming total memory is constant
    normalized_memory_used = memory_used / memory_total

    # Averaging utilization and memory usage
    avg_utilization = utilization.groupby(timestamps).mean()
    avg_memory_used = normalized_memory_used.groupby(timestamps).mean()
    total_power_draw = power_draw.groupby(timestamps).sum()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Average Utilization over time
    plt.figure()
    plt.plot(avg_utilization.index, avg_utilization, label='Average Utilization')
    plt.xlabel('Time')
    plt.ylabel('Average GPU Utilization (%)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_aggregate_utilization_over_time')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_aggregate_utilization_over_time.png'))
    plt.close()

    # Total Power draw over time
    plt.figure()
    plt.plot(total_power_draw.index, total_power_draw, label='Total Power Draw')
    plt.xlabel('Time')
    plt.ylabel('Total Power Draw (W)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_aggregate_power_draw_over_time')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_aggregate_power_draw_over_time.png'))
    plt.close()

    # Average Memory usage over time
    plt.figure()
    plt.plot(avg_memory_used.index, avg_memory_used, label='Average Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Average Memory Usage (Normalized)')
    plt.title(f'{model}_{batch_size}_{gpu_model}_aggregate_memory_usage_over_time')
    plt.savefig(os.path.join(output_dir, f'{model}_{batch_size}_{gpu_model}_aggregate_memory_usage_over_time.png'))
    plt.close()

# Function to process a single log file
def process_log_file(filepath):
    data = pd.read_csv(filepath, header=None)
    data.replace('[N/A]', 0, inplace=True)  # Replace '[N/A]' with 0
    return data

# Function to create proxy logs
def create_proxy_log(data, start_time, end_time, interval):
    timestamps = pd.to_datetime(data.iloc[:, 0])
    utilization = data.iloc[:, 1].replace('[N/A]', 0).astype(float)
    power_draw = data.iloc[:, 2].replace('[N/A]', 0).astype(float)
    memory_used = data.iloc[:, 3].replace('[N/A]', 0).astype(float)
    memory_total = data.iloc[:, 4].iloc[0]  # Assuming total memory is constant

    proxy_data = []
    current_time = start_time
    while current_time <= end_time:
        mask = (timestamps >= current_time) & (timestamps < current_time + interval)
        if mask.any():
            avg_utilization = utilization[mask].mean()
            avg_power_draw = power_draw[mask].mean()
            avg_memory_used = memory_used[mask].mean()
        else:
            avg_utilization = 0
            avg_power_draw = 0
            avg_memory_used = 0

        proxy_data.append([current_time, avg_utilization, avg_power_draw, avg_memory_used, memory_total])
        current_time += interval

    return pd.DataFrame(proxy_data, columns=['timestamp', 'utilization', 'power_draw', 'memory_used', 'memory_total'])

# Function to traverse directories and process logs
def traverse_and_plot(base_dir):
    for root, dirs, files in os.walk(base_dir):
        gpu_usage_files = [file for file in files if file.endswith('.log') and 'gpu_usage' in file]
        
        data_list = []
        for file in gpu_usage_files:
            # Extract model, batch_size, gpu_model, and gpu_id from the directory structure
            parts = root.split(os.sep)
            model_batch_gpu = parts[-1]
            model = model_batch_gpu.split('_')[0]
            batch_size = model_batch_gpu.split('_')[1]
            gpu_model = model_batch_gpu.split('_')[2][:-1]
            gpu_id = file.split('_')[-1][0]

            # Process the log file
            log_filepath = os.path.join(root, file)
            data = process_log_file(log_filepath)
            data_list.append(data)
            
            # Create output directory for plots
            output_dir = os.path.join(root, 'plots')
            
            # Generate plots for individual GPU
            create_plots(data, model, batch_size, gpu_model, gpu_id, output_dir)

        # Generate aggregate plots if more than one GPU log file is found
        if len(gpu_usage_files) > 1:
            # Determine the start and end time for the proxy logs
            start_time = min(pd.to_datetime(data.iloc[0, 0]) for data in data_list)
            end_time = max(pd.to_datetime(data.iloc[-1, 0]) for data in data_list)
            interval = timedelta(milliseconds=25)

            # Create proxy logs
            proxy_data_list = [create_proxy_log(data, start_time, end_time, interval) for data in data_list]

            # Generate aggregate plots using proxy logs
            aggregate_output_dir = os.path.join(root, 'plots')
            create_aggregate_plots(proxy_data_list, model, batch_size, gpu_model, aggregate_output_dir)

# Run the function
traverse_and_plot(log_dir)
