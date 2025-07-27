#!/bin/bash

# Configuration
cd /home/xinyu/dev/MathVista
temperature=0.6
top_p=0.99
num_answers=4  # Number of answers to generate per problem
dataset_size=1000  # Default dataset size, will be updated based on actual data
tensor_parallel_size=2  # GPUs per process

# Detect number of available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # If CUDA_VISIBLE_DEVICES is not set, detect all GPUs
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
    available_gpus=$(seq 0 $((num_gpus-1)) | tr '\n' ',' | sed 's/,$//')
else
    # Use specified GPUs
    IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    num_gpus=${#gpu_array[@]}
    available_gpus="$CUDA_VISIBLE_DEVICES"
fi

echo "Detected $num_gpus GPUs: $available_gpus"

# Calculate number of processes based on tensor parallel size
num_processes=$((num_gpus / tensor_parallel_size))
if [ $num_processes -eq 0 ]; then
    num_processes=1
    tensor_parallel_size=$num_gpus
fi

# Create output directory name with temperature, top_p, GPU count, and number of answers
output_dir="logs/qwen2vl_cot_temp_${temperature}_topp_${top_p}_gpu_${num_gpus}_n${num_answers}"
output_file="results.json"

echo "Running $num_processes processes with $tensor_parallel_size GPUs each"

# Get dataset size by loading and counting
echo "Getting dataset size..."
dataset_size=1000

echo "Dataset size: $dataset_size"

# Calculate data splits
items_per_process=$((dataset_size / num_processes))
remainder=$((dataset_size % num_processes))

echo "Items per process: $items_per_process"

# Function to run inference on a GPU subset
run_inference() {
    local gpu_ids=$1
    local start_idx=$2
    local end_idx=$3
    local process_id=$4
    
    echo "Process $process_id: GPUs $gpu_ids, processing items $start_idx to $end_idx"
    
    CUDA_VISIBLE_DEVICES=$gpu_ids /home/xinyu/miniconda3/envs/mathvista/bin/python vllm_inference.py \
        --dataset_name /run/determined/NAS1/data/AI4Math/MathVista \
        --model qwen2vl \
        --qwen2vl_model_path /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2-VL-7B-Instruct \
        --tensor_parallel_size $tensor_parallel_size \
        --gpu_memory_utilization 0.95 \
        --temperature $temperature \
        --top_p $top_p \
        --max_new_tokens 1024 \
        --output_dir $output_dir \
        --shot_num 0 \
        --shot_type solution \
        --start_idx $start_idx \
        --end_idx $end_idx \
        --process_id $process_id \
        --num_answers $num_answers &
}

# Launch inference processes
pids=()
IFS=',' read -ra gpu_array <<< "$available_gpus"

for ((i=0; i<num_processes; i++)); do
    # Calculate start and end indices for this process
    start_idx=$((i * items_per_process))
    if [ $i -eq $((num_processes - 1)) ]; then
        # Last process handles remainder
        end_idx=$dataset_size
    else
        end_idx=$(((i + 1) * items_per_process))
    fi
    
    # Assign GPUs to this process
    gpu_start=$((i * tensor_parallel_size))
    gpu_end=$((gpu_start + tensor_parallel_size - 1))
    
    # Build GPU ID string
    gpu_ids=""
    for ((j=gpu_start; j<=gpu_end && j<num_gpus; j++)); do
        if [ -z "$gpu_ids" ]; then
            gpu_ids="${gpu_array[j]}"
        else
            gpu_ids="${gpu_ids},${gpu_array[j]}"
        fi
    done
    
    # Run inference
    run_inference "$gpu_ids" "$start_idx" "$end_idx" "$i"
    pids+=($!)
done

echo "Launched ${#pids[@]} inference processes"

# Wait for all processes to complete
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo "All inference processes completed"

# Merge results from all processes
echo "Merging results..."
merged_output_file="merged_results.json"

python3 -c "
import json
import sys
import os

results = {}
output_dir = '$output_dir'
merged_file = f'{output_dir}/${merged_output_file}'
num_processes = ${num_processes}

for i in range(num_processes):
    process_file = f'{output_dir}/{i:03d}.json'
    if os.path.exists(process_file):
        print(f'Loading {process_file}')
        with open(process_file, 'r') as f:
            process_results = json.load(f)
            results.update(process_results)
    else:
        print(f'Warning: {process_file} not found')

print(f'Total merged results: {len(results)}')
with open(merged_file, 'w') as f:
    json.dump(results, f, indent=2)

print('Results merged successfully')
"

# Extract answers for merged results
echo "Extracting answers..."
/home/xinyu/miniconda3/envs/mathvista/bin/python evaluation/extract_answer_no_api.py \
    --results_file $output_dir/$merged_output_file \
    --response_label response \
    --verbose

# Calculate score for merged results
echo "Calculating scores..."
/home/xinyu/miniconda3/envs/mathvista/bin/python -m evaluation.calculate_score \
    --dataset_name /run/determined/NAS1/data/AI4Math/MathVista \
    --output_dir $output_dir \
    --output_file $merged_output_file \
    --score_file scores_qwen2vl_multi_gpu.json

# Evaluate multiple answers with pass@k and majority voting
echo "Evaluating multiple answers..."
/home/xinyu/miniconda3/envs/mathvista/bin/python evaluation/evaluate_multiple_answers.py \
    --results_file $output_dir/$merged_output_file \
    --output_file $output_dir/multiple_answers_evaluation.json

echo "Evaluation completed. Results saved to $output_dir/$merged_output_file"
echo "Scores saved to $output_dir/scores_qwen2vl_multi_gpu.json"
echo "Multiple answers evaluation saved to $output_dir/multiple_answers_evaluation.json"