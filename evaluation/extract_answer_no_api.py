#!/usr/bin/env python3
"""
无API答案提取脚本 - 基于正则表达式的答案提取
增强版本，支持 "Final Answer:" 格式和多种备用提取模式
"""

import argparse
import json
import re
import logging
import sys
import os
from tqdm import tqdm

# Add parent directory to path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import read_json, save_json


def extract_answer_regex(response, problem):
    """
    使用正则表达式提取答案，不依赖API调用
    优先提取 "Final Answer: X" 格式的答案
    """
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem.get('choices', [])
    
    if not response or response.strip() == "":
        return ""
    
    response = response.strip()
    
    # 首先尝试提取 "Final Answer:" 格式的答案
    final_answer_pattern = r'Final Answer:\s*(.+?)(?:\n|$)'
    final_answer_match = re.search(final_answer_pattern, response, re.IGNORECASE)
    
    if final_answer_match:
        extracted = final_answer_match.group(1).strip()
        
        if question_type == 'multi_choice':
            # 多选题处理
            if extracted in choices:
                return extracted
            
            # 如果是选项字母，转换为对应的选项内容
            if len(extracted) == 1 and extracted.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                letter_index = ord(extracted.upper()) - ord('A')
                if 0 <= letter_index < len(choices):
                    return choices[letter_index]
        
        elif answer_type in ["integer", "float"]:
            # 数值型答案处理
            try:
                if answer_type == "integer":
                    return str(int(float(extracted)))
                else:
                    return str(float(extracted))
            except ValueError:
                pass
        
        elif answer_type == "list":
            # 列表型答案处理
            if extracted.startswith('[') and extracted.endswith(']'):
                return extracted
    
    # 如果没有找到 "Final Answer:" 格式，使用备用提取方法
    return extract_answer_fallback(response, problem)


def extract_answer_fallback(response, problem):
    """
    备用答案提取方法，用于处理没有标准格式的回答
    """
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem.get('choices', [])
    
    if question_type == 'multi_choice':
        # 多选题备用提取
        # 1. 直接匹配选项内容
        for choice in choices:
            if choice.lower() in response.lower():
                return choice
        
        # 2. 提取选项字母 - 多种模式
        letter_patterns = [
            r'答案是\s*([A-Z])',
            r'answer is\s*([A-Z])',
            r'选择\s*([A-Z])',
            r'选项\s*([A-Z])',
            r'正确答案是\s*([A-Z])',
            r'correct answer is\s*([A-Z])',
            r'因此答案是\s*([A-Z])',
            r'所以答案是\s*([A-Z])',
            r'therefore.*?([A-Z])',
            r'thus.*?([A-Z])',
            r'\(([A-Z])\)',  # (A), (B), (C), (D)
            r'([A-Z])(?:[.,!?]|\s|$)'  # 单独的字母后跟标点或空格
        ]
        
        for pattern in letter_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                letter = matches[-1].upper()
                letter_index = ord(letter) - ord('A')
                if 0 <= letter_index < len(choices):
                    return choices[letter_index]
    
    elif answer_type in ["integer", "float"]:
        # 数值型答案备用提取
        number_patterns = [
            r'答案是\s*([-+]?\d*\.?\d+)',
            r'answer is\s*([-+]?\d*\.?\d+)',
            r'结果是\s*([-+]?\d*\.?\d+)',
            r'result is\s*([-+]?\d*\.?\d+)',
            r'等于\s*([-+]?\d*\.?\d+)',
            r'equals?\s*([-+]?\d*\.?\d+)',
            r'为\s*([-+]?\d*\.?\d+)',
            r'是\s*([-+]?\d*\.?\d+)',
            r'总共\s*([-+]?\d*\.?\d+)',
            r'total\s*:?\s*([-+]?\d*\.?\d+)',
            r'共\s*([-+]?\d*\.?\d+)',
            r'需要\s*\$?([-+]?\d*\.?\d+)',
            r'need\s*\$?([-+]?\d*\.?\d+)',
            r'costs?\s*\$?([-+]?\d*\.?\d+)',
            r'价格是\s*\$?([-+]?\d*\.?\d+)',
            r'price is\s*\$?([-+]?\d*\.?\d+)',
            r'因此.*?([-+]?\d*\.?\d+)',
            r'所以.*?([-+]?\d*\.?\d+)',
            r'therefore.*?([-+]?\d*\.?\d+)',
            r'thus.*?([-+]?\d*\.?\d+)',
            r'([-+]?\d*\.?\d+)(?:[.,!?]|\s*$)'  # 句末的数字
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    num = float(matches[-1])
                    if answer_type == "integer":
                        return str(int(num))
                    else:
                        return str(num)
                except ValueError:
                    continue
        
        # 寻找美元符号后的数字
        dollar_pattern = r'\$\s*([-+]?\d*\.?\d+)'
        matches = re.findall(dollar_pattern, response)
        if matches:
            try:
                num = float(matches[-1])
                if answer_type == "integer":
                    return str(int(num))
                else:
                    return str(num)
            except ValueError:
                pass
    
    elif answer_type == "list":
        # 列表型答案备用提取
        list_pattern = r'\[([^\]]+)\]'
        matches = re.findall(list_pattern, response)
        if matches:
            return f"[{matches[-1]}]"
        
        # 寻找年份范围
        year_pattern = r'(\d{4})\s*(?:和|and|to|-|至)\s*(\d{4})'
        matches = re.findall(year_pattern, response)
        if matches:
            return f"[{matches[-1][0]}, {matches[-1][1]}]"
    
    return ""


def verify_extraction(extraction):
    """验证提取的答案是否有效"""
    if not extraction or extraction.strip() == "":
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="无API答案提取脚本")
    parser.add_argument('--results_file', type=str, required=True, help='包含模型响应的结果文件路径')
    parser.add_argument('--response_label', type=str, default='response', help='响应字段的标签名')
    parser.add_argument('--output_file', type=str, help='输出文件路径（如果不指定，将覆盖输入文件）')
    parser.add_argument('--rerun', action='store_true', help='重新运行所有问题的答案提取')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='限制处理的问题数量')
    parser.add_argument('--save_every', type=int, default=100, help='每处理多少个问题保存一次')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("MathVista: 无API答案提取 - 开始")
    
    # 读取结果文件
    print(f"读取结果文件: {args.results_file}")
    results = read_json(args.results_file)
    
    full_pids = list(results.keys())
    
    # 确定需要处理的问题
    skip_pids = []
    if not args.rerun:
        for pid, problem in results.items():
            extraction = problem.get('extraction')
            if extraction is not None and verify_extraction(extraction):
                skip_pids.append(pid)
    
    if len(skip_pids) > 0:
        print(f"找到 {len(skip_pids)} 个已有有效答案的问题，将跳过这些问题...")
    
    test_pids = [pid for pid in full_pids if pid not in skip_pids]
    
    if args.max_num_problems > 0:
        test_pids = test_pids[:min(args.max_num_problems, len(test_pids))]
        print(f'限制处理问题数量为 {args.max_num_problems}')
    
    print(f"需要处理的问题数量: {len(test_pids)}")
    
    # 处理每个问题
    for i, pid in enumerate(tqdm(test_pids, desc="提取答案")):
        problem = results[pid]
        
        if args.response_label not in problem:
            logging.warning(f"问题 {pid} 中没有找到响应字段 '{args.response_label}'")
            continue
        
        response = problem[args.response_label]
        extraction = extract_answer_regex(response, problem)
        results[pid]['extraction'] = extraction
        
        # if args.verbose and extraction:
        #     print(f"问题 {pid}: 提取到答案 '{extraction}'")
        # elif args.verbose:
        #     print(f"问题 {pid}: 未能提取到答案")
        
        # 定期保存结果
        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            output_file = args.output_file if args.output_file else args.results_file
            save_json(results, output_file)
            print(f"已保存结果到 {output_file}")
    
    print("MathVista: 无API答案提取 - 完成")
    
    # 统计提取成功率
    total_processed = len(test_pids)
    successful_extractions = sum(1 for pid in test_pids if verify_extraction(results[pid].get('extraction', '')))
    success_rate = successful_extractions / total_processed if total_processed > 0 else 0
    
    print(f"提取统计:")
    print(f"  总处理问题数: {total_processed}")
    print(f"  成功提取答案数: {successful_extractions}")
    print(f"  提取成功率: {success_rate:.2%}")


if __name__ == '__main__':
    main() 