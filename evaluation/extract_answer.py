import argparse
import logging
import os
import re

from openai import AzureOpenAI
from rich.logging import RichHandler
from tqdm import tqdm

from evaluation.prompts.ext_ans import demo_prompt
from models import gpt
from utilities import read_json, save_json


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer_regex(response, problem):
    """
    使用正则表达式提取答案，不依赖API调用
    """
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    
    if response == "":
        return ""
    
    response = response.strip()
    
    # 首先尝试提取 "Final Answer: X" 格式的答案
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
    choices = problem['choices']
    
    if question_type == 'multi_choice':
        # 多选题备用提取
        # 1. 直接匹配选项内容
        for choice in choices:
            if choice.lower() in response.lower():
                return choice
        
        # 2. 提取选项字母
        letter_patterns = [
            r'答案是\s*([A-Z])',
            r'answer is\s*([A-Z])',
            r'选择\s*([A-Z])',
            r'选项\s*([A-Z])',
            r'正确答案是\s*([A-Z])',
            r'\(([A-Z])\)',
            r'([A-Z])(?:[.,!?]|\s|$)'
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
            r'等于\s*([-+]?\d*\.?\d+)',
            r'为\s*([-+]?\d*\.?\d+)',
            r'是\s*([-+]?\d*\.?\d+)',
            r'([-+]?\d*\.?\d+)(?:[.,!?]|\s*$)'
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
    
    elif answer_type == "list":
        # 列表型答案备用提取
        list_pattern = r'\[([^\]]+)\]'
        matches = re.findall(list_pattern, response)
        if matches:
            return f"[{matches[-1]}]"
    
    return ""


def extract_answer(model, response, problem, quick_extract=False):
    """
    主要的答案提取函数，优先使用正则表达式提取
    """
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    # 使用新的正则表达式提取方法
    extraction = extract_answer_regex(response, problem)
    if extraction:
        return extraction

    # 如果正则表达式提取失败，并且quick_extract为True，尝试旧的快速提取
    if quick_extract:
        logging.info("Quickly extracting answer...")
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    # 最后尝试使用API提取（如果model不为None）
    if model is not None:
        try:
            full_prompt = create_test_prompt(demo_prompt, query, response)
            extraction = model.get_response(user_prompt=full_prompt)
            return extraction
        except Exception as e:
            logging.info(f"Error in extracting answer for problem: {pid} with response: {response}")
            logging.info(e)

    return ""


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results_file_path', type=str, default='answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The max number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')

    parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))

    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Extract Answers - Start")
    args = parse_args()

    # args
    label = args.response_label

    assert (
        args.azure_openai_api_endpoint is not None
    ), "Env var AZURE_OPENAI_API_ENDPOINT is not set but is required for OpenAI client."
    assert (
        args.azure_openai_api_key is not None
    ), "Env var AZURE_OPENAI_API_KEY is not set but is required for OpenAI client."
    assert (
        args.azure_openai_api_version is not None
    ), "Env var AZURE_OPENAI_API_VERSION is not set but is required for OpenAI client."
    assert (
        args.azure_openai_model is not None
    ), "Env var AZURE_OPENAI_MODEL is not set but is required for OpenAI client."

    client = AzureOpenAI(
        azure_endpoint=args.azure_openai_api_endpoint,
        api_key=args.azure_openai_api_key,
        api_version=args.azure_openai_api_version,
    )
    model = gpt.GPT_Model(client=client, model=args.azure_openai_model)

    logging.info(f"Reading {args.results_file_path}...")
    results = read_json(args.results_file_path)

    full_pids = list(results.keys())

    skip_pids = []
    for pid, problem in results.items():
        extraction = problem.get('extraction')
        if extraction is not None and verify_extraction(extraction):
            skip_pids.append(problem['pid'])

    if args.rerun:
        test_pids = full_pids
    else:
        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
            )
        test_pids = [pid for pid in full_pids if pid not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.info(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]
        extraction = extract_answer(model, response, problem, args.quick_extract)
        results[pid]['extraction'] = extraction

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, args.results_file_path)
            logging.info(f"Saved results to {args.results_file_path}")

    logging.info("MathVista: Extract Answers - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
