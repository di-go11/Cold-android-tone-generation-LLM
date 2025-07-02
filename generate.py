#!/usr/bin/env python3
"""
Text Generation Script for ELYZA LLM

This script loads the fine-tuned model and generates text in cold android tone.
"""

import argparse
import logging
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str, base_model: str = None, use_4bit: bool = False):
    """Load the fine-tuned model and tokenizer."""
    
    # Setup quantization if needed
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if base_model:
        # Load base model first, then PEFT adapter
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load the complete fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    model.eval()
    return model, tokenizer


def format_prompt(instruction: str, input_text: str = None) -> str:
    """Format the prompt for generation."""
    if input_text:
        prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n"
    else:
        prompt = f"### 指示:\n{instruction}\n\n### 応答:\n"
    return prompt


def generate_text(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """Generate text using the model."""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return only the generated part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def interactive_mode(model, tokenizer, config):
    """Run interactive text generation."""
    print("🤖 ELYZA Android Tone Generator")
    print("Enter your instructions (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            instruction = input("\n指示: ").strip()
            if instruction.lower() in ['quit', 'exit', 'q']:
                break
            
            input_text = input("入力 (optional): ").strip()
            if not input_text:
                input_text = None
            
            prompt = format_prompt(instruction, input_text)
            
            print("\n生成中...")
            generated = generate_text(
                model, 
                tokenizer, 
                prompt,
                max_new_tokens=config.get('max_new_tokens', 512),
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 0.9),
                do_sample=config.get('do_sample', True)
            )
            
            print(f"\n🤖 応答: {generated}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"エラー: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with fine-tuned ELYZA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model name (for PEFT adapters)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for generation"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text for the prompt"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using defaults")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        args.base_model,
        config.get('use_4bit', False)
    )
    logger.info("Model loaded successfully")
    
    if args.interactive:
        interactive_mode(model, tokenizer, config)
    elif args.prompt:
        prompt = format_prompt(args.prompt, args.input)
        generated = generate_text(
            model, 
            tokenizer, 
            prompt,
            max_new_tokens=config.get('max_new_tokens', 512),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.9),
            do_sample=config.get('do_sample', True)
        )
        print(f"生成結果: {generated}")
    else:
        print("Please specify --prompt or use --interactive mode")


if __name__ == "__main__":
    main()