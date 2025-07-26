#!/bin/bash
temperature=0
top_p=0.9
output_file="qwen2vl_temp_${temperature}_topp_${top_p}.json"
##### qwen2vl_single_gpu #####
# generate response
CUDA_VISIBLE_DEVICES=4,5 /home/xinyu/miniconda3/envs/mathvista/bin/python -m evaluation.generate_response \
--dataset_name /run/determined/NAS1/data/AI4Math/MathVista \
--model qwen2vl \
--qwen2vl_model_path /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2-VL-7B-Instruct \
--tensor_parallel_size 2 \
--gpu_memory_utilization 0.85 \
--temperature $temperature \
--top_p $top_p \
--max_new_tokens 512 \
--output_dir logs/qwen2vl \
--output_file $output_file \
--shot_num 0 \
--shot_type solution \
# --use_caption \
# --use_ocr \
# --caption_file data/texts/captions_bard.json \
# --ocr_file data/texts/ocrs_easyocr.json

# extract answer (using regex-based extraction without API)
/home/xinyu/miniconda3/envs/mathvista/bin/python evaluation/extract_answer_no_api.py \
--results_file logs/qwen2vl/$output_file \
--response_label response \
--verbose

# calculate score
/home/xinyu/miniconda3/envs/mathvista/bin/python -m evaluation.calculate_score \
--dataset_name /run/determined/NAS1/data/AI4Math/MathVista \
--output_dir logs/qwen2vl \
--output_file $output_file \
--score_file scores_qwen2vl_single_gpu.json 