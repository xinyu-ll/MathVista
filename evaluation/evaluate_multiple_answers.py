import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Any
import logging
from Levenshtein import distance


def get_most_similar(prediction, choices):
    """Use the Levenshtein distance to determine which choice is most similar to the prediction"""
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]


def safe_equal(prediction, answer):
    """Check if the prediction is equal to the answer, even if they are of different types"""
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        logging.info(e)
        return False


def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False):
    """Normalize the extracted answer to match the answer type (consistent with calculate_score.py)"""
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # if the extraction is empty, return None
        if ignore_empty_extractions and not extraction:
            return None

        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = get_most_similar(extraction, choices)
        
        return normalized_extraction

    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None
    else:  # text
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_answer_from_response(response, problem):
    """
    Extract answer from response using the same logic as extract_answer_no_api.py
    """
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem.get('choices', [])
    
    if not response or response.strip() == "":
        return ""
    
    response = response.strip()
    
    # First try to extract "Final Answer:" format
    final_answer_pattern = r'Final Answer:\s*(.+?)(?:\n|$)'
    final_answer_match = re.search(final_answer_pattern, response, re.IGNORECASE)
    
    if final_answer_match:
        extracted = final_answer_match.group(1).strip()
        
        if question_type == 'multi_choice':
            # Multi-choice handling
            if extracted in choices:
                return extracted
            
            # If it's a letter, convert to corresponding choice
            if len(extracted) == 1 and extracted.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                letter_index = ord(extracted.upper()) - ord('A')
                if 0 <= letter_index < len(choices):
                    return choices[letter_index]
        
        elif answer_type in ["integer", "float"]:
            # Numeric answer handling
            try:
                float_val = float(extracted)
                if answer_type == "integer":
                    if float_val == float('inf') or float_val == float('-inf') or float_val != float_val:  # Check for inf, -inf, nan
                        return None
                    return str(int(float_val))
                else:
                    return str(float_val)
            except (ValueError, OverflowError):
                pass
        
        elif answer_type == "list":
            # List answer handling
            if extracted.startswith('[') and extracted.endswith(']'):
                return extracted
    
    # Fallback extraction methods (simplified version of extract_answer_fallback)
    if question_type == 'multi_choice':
        # Direct match with choices
        for choice in choices:
            if choice.lower() in response.lower():
                return choice
        
        # Extract option letters
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
            r'([A-Z])(?:[.,!?]|\s|$)'  # Single letter followed by punctuation or space
        ]
        
        for pattern in letter_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                letter = matches[-1].upper()
                letter_index = ord(letter) - ord('A')
                if 0 <= letter_index < len(choices):
                    return choices[letter_index]
    
    elif answer_type in ["integer", "float"]:
        # Numeric answer fallback extraction
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
            r'([-+]?\d*\.?\d+)(?:[.,!?]|\s*$)'  # Number at end of sentence
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
        
        # Look for numbers after dollar signs
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
        # List answer fallback extraction
        list_pattern = r'\[([^\]]+)\]'
        matches = re.findall(list_pattern, response)
        if matches:
            return f"[{matches[-1]}]"
        
        # Look for year ranges
        year_pattern = r'(\d{4})\s*(?:和|and|to|-|至)\s*(\d{4})'
        matches = re.findall(year_pattern, response)
        if matches:
            return f"[{matches[-1][0]}, {matches[-1][1]}]"
    
    return ""


def majority_vote_prediction(predictions: List[Any]) -> Any:
    """Get majority vote from list of predictions, handling None values and unhashable types"""
    # Filter out None values
    valid_predictions = [pred for pred in predictions if pred is not None]
    if not valid_predictions:
        return None
    
    # Convert unhashable types (like lists) to strings for counting
    hashable_predictions = []
    for pred in valid_predictions:
        if isinstance(pred, list):
            hashable_predictions.append(str(pred))
        else:
            hashable_predictions.append(pred)
    
    # Get most common prediction
    counter = Counter(hashable_predictions)
    most_common_str = counter.most_common(1)[0][0]
    
    # Find the original prediction that corresponds to the most common string
    for i, pred in enumerate(valid_predictions):
        if isinstance(pred, list):
            if str(pred) == most_common_str:
                return pred
        else:
            if pred == most_common_str:
                return pred
    
    return valid_predictions[0]  # fallback


def calculate_pass_at_k(correct_list: List[bool], k: int) -> float:
    """Calculate pass@k metric"""
    if not correct_list or k <= 0:
        return 0.0
    
    # pass@k = 1 if any of the first k answers are correct
    # For pass@N, k should be min(k, len(correct_list)) to use all available answers
    k = min(k, len(correct_list))
    return 1.0 if any(correct_list[:k]) else 0.0


def evaluate_multiple_answers(results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate results with multiple answers per problem using consistent extraction logic"""
    
    evaluation_results = {
        "total_problems": 0,
        "problems_with_multiple_answers": 0,
        "pass_at_1_scores": [],
        "pass_at_n_scores": [],
        "majority_vote_scores": [],
        "detailed_results": {}
    }
    
    for problem_id, result in results.items():
        # Skip if no responses
        if 'responses' not in result or not result['responses']:
            continue
            
        evaluation_results["total_problems"] += 1
        responses = result['responses']
        num_answers = len(responses)
        
        if num_answers > 1:
            evaluation_results["problems_with_multiple_answers"] += 1
        
        # Get problem metadata for evaluation
        choices = result.get('choices', [])
        question_type = result.get('question_type', 'free_form')
        answer_type = result.get('answer_type', 'text')
        precision = result.get('precision', 1)
        ground_truth = result.get('answer', 'UNKNOWN')
        
        # Normalize ground truth to match the same format as predictions
        if answer_type == 'integer':
            try:
                ground_truth = str(int(ground_truth)) if isinstance(ground_truth, (int, float)) else str(int(float(ground_truth)))
            except:
                pass
        elif answer_type == 'float':
            try:
                if isinstance(ground_truth, (int, float)):
                    ground_truth = str(round(float(ground_truth), int(precision)))
                else:
                    ground_truth = str(round(float(ground_truth), int(precision)))
            except:
                pass
        elif answer_type == 'text':
            ground_truth = str(ground_truth)
        
        # Get predictions for all responses
        predictions = []
        correct_answers = []
        extractions = []  # For debugging
        
        # Use existing prediction for the first response if available and consistent
        if 'prediction' in result and result['prediction'] is not None:
            first_prediction = result['prediction']
            predictions.append(first_prediction)
            extractions.append(result.get('extraction', ''))
            
            # Use existing true_false for the first response
            if 'true_false' in result:
                correct_answers.append(result['true_false'])
            else:
                is_correct = safe_equal(first_prediction, ground_truth)
                correct_answers.append(is_correct)
        else:
            # Fallback: extract and normalize the first response
            first_extraction = extract_answer_from_response(responses[0], result) if responses else ""
            extractions.append(first_extraction)
            first_prediction = normalize_extracted_answer(
                first_extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False
            )
            predictions.append(first_prediction)
            is_correct = safe_equal(first_prediction, ground_truth)
            correct_answers.append(is_correct)
        
        # Process remaining responses
        if len(responses) > 1:
            for response in responses[1:]:
                extraction = extract_answer_from_response(response, result)
                extractions.append(extraction)
                prediction = normalize_extracted_answer(
                    extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False
                )
                predictions.append(prediction)
                is_correct = safe_equal(prediction, ground_truth)
                correct_answers.append(is_correct)
        
        # Calculate metrics
        pass_at_1 = calculate_pass_at_k(correct_answers, 1)
        pass_at_n = calculate_pass_at_k(correct_answers, num_answers)
        
        # Majority voting
        majority_prediction = majority_vote_prediction(predictions)
        majority_correct = safe_equal(majority_prediction, ground_truth)
        
        evaluation_results["pass_at_1_scores"].append(pass_at_1)
        evaluation_results["pass_at_n_scores"].append(pass_at_n)
        evaluation_results["majority_vote_scores"].append(1.0 if majority_correct else 0.0)
        
        # Store detailed results
        evaluation_results["detailed_results"][problem_id] = {
            "responses": responses,
            "extractions": extractions,
            "predictions": predictions,
            "correct_answers": correct_answers,
            "ground_truth": ground_truth,
            "pass_at_1": pass_at_1,
            "pass_at_n": pass_at_n,
            "majority_prediction": majority_prediction,
            "majority_correct": majority_correct
        }
    
    # Calculate overall metrics
    if evaluation_results["pass_at_1_scores"]:
        evaluation_results["overall_pass_at_1"] = sum(evaluation_results["pass_at_1_scores"]) / len(evaluation_results["pass_at_1_scores"])
        evaluation_results["overall_pass_at_n"] = sum(evaluation_results["pass_at_n_scores"]) / len(evaluation_results["pass_at_n_scores"])
        evaluation_results["overall_majority_vote"] = sum(evaluation_results["majority_vote_scores"]) / len(evaluation_results["majority_vote_scores"])
    else:
        evaluation_results["overall_pass_at_1"] = 0.0
        evaluation_results["overall_pass_at_n"] = 0.0
        evaluation_results["overall_majority_vote"] = 0.0
    
    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple answers with pass@k and majority voting')
    parser.add_argument('--results_file', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output evaluation file')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}")
    results = load_results(args.results_file)
    
    # Evaluate
    print("Evaluating multiple answers...")
    evaluation = evaluate_multiple_answers(results)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Total problems: {evaluation['total_problems']}")
    print(f"Problems with multiple answers: {evaluation['problems_with_multiple_answers']}")
    print(f"Overall Pass@1: {evaluation['overall_pass_at_1']:.3f}")
    print(f"Overall Pass@N: {evaluation['overall_pass_at_n']:.3f}")
    print(f"Overall Majority Vote: {evaluation['overall_majority_vote']:.3f}")
    
    # Save detailed results
    with open(args.output_file, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    print(f"\nDetailed evaluation saved to {args.output_file}")


if __name__ == "__main__":
    main()