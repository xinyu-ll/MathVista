#!/bin/bash

# Configuration
cd /home/xinyu/dev/MathVista
temperature=0
top_p=0.99
num_answers=1  # Number of answers to generate per problem
num_gpus=2

# Create output directory name with temperature, top_p, GPU count, and number of answers
output_dir="logs/qwen2vl_temp_${temperature}_topp_${top_p}_gpu_${num_gpus}_n${num_answers}"
merged_output_file="merged_results.json"

echo "Using existing results from: $output_dir/$merged_output_file"

# Check if merged results file exists
if [ ! -f "$output_dir/$merged_output_file" ]; then
    echo "Error: $output_dir/$merged_output_file not found"
    echo "Please run inference first or check the output directory"
    exit 1
fi

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

echo "Evaluation completed. Results loaded from $output_dir/$merged_output_file"
echo "Scores saved to $output_dir/scores_qwen2vl_multi_gpu.json"
echo "Multiple answers evaluation saved to $output_dir/multiple_answers_evaluation.json"