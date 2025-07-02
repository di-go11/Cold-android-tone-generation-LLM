#!/usr/bin/env python3
"""
Evaluation Script for ELYZA LLM Fine-tuning

This script evaluates the fine-tuned model's performance on generating
cold android tone text.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(test_file: str) -> List[Dict]:
    """Load test data from JSON file."""
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_model(model, tokenizer, test_data: List[Dict], output_file: str = None):
    """Evaluate the model on test data."""
    results = []
    
    for i, example in enumerate(test_data):
        try:
            # Format prompt
            if 'input' in example and example['input']:
                prompt = f"### 指示:\n{example['instruction']}\n\n### 入力:\n{example['input']}\n\n### 応答:\n"
            else:
                prompt = f"### 指示:\n{example['instruction']}\n\n### 応答:\n"
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            result = {
                "id": i,
                "instruction": example['instruction'],
                "input": example.get('input', ''),
                "expected": example.get('output', ''),
                "generated": generated_text
            }
            results.append(result)
            
            logger.info(f"Processed example {i+1}/{len(test_data)}")
            
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            continue
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return results


def print_sample_results(results: List[Dict], num_samples: int = 3):
    """Print sample results for manual inspection."""
    print("\n" + "="*50)
    print("SAMPLE EVALUATION RESULTS")
    print("="*50)
    
    for i, result in enumerate(results[:num_samples]):
        print(f"\n--- Example {i+1} ---")
        print(f"指示: {result['instruction']}")
        if result['input']:
            print(f"入力: {result['input']}")
        print(f"期待値: {result['expected']}")
        print(f"生成結果: {result['generated']}")
        print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned ELYZA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/sample_data.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model name (for PEFT adapters)"
    )
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data)
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.base_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    model.eval()
    logger.info("Model loaded successfully")
    
    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluate_model(model, tokenizer, test_data, args.output_file)
    
    # Print sample results
    print_sample_results(results)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()