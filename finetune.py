#!/usr/bin/env python3
"""
ELYZA LLM Fine-tuning Script for Cold Android Tone Generation

This script fine-tunes ELYZA's Japanese LLM using LoRA/QLoRA techniques
to generate text in a cold, android-like tone.
"""

import os
import yaml
import argparse
import logging
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """Setup quantization configuration for QLoRA."""
    if not config.get('use_4bit', False):
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=config.get('use_nested_quant', False),
        bnb_4bit_quant_type=config.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_compute_dtype=getattr(torch, config.get('bnb_4bit_compute_dtype', 'float16'))
    )


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Setup LoRA configuration."""
    return LoraConfig(
        r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 32),
        target_modules=config.get('lora_target_modules', ["q_proj", "v_proj"]),
        lora_dropout=config.get('lora_dropout', 0.1),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def format_instruction(example):
    """Format training examples into instruction format."""
    if 'instruction' in example and 'output' in example:
        if 'input' in example and example['input']:
            prompt = f"### 指示:\n{example['instruction']}\n\n### 入力:\n{example['input']}\n\n### 応答:\n"
        else:
            prompt = f"### 指示:\n{example['instruction']}\n\n### 応答:\n"
        return prompt + example['output']
    else:
        return example.get('text', '')


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples."""
    texts = [format_instruction(example) for example in examples]
    return tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ELYZA LLM for cold android tone")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup model and tokenizer
    model_name = config['model_name']
    logger.info(f"Loading model: {model_name}")
    
    # Setup quantization
    quantization_config = setup_quantization_config(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.get('fp16', True) else torch.float32
    )
    
    # Prepare model for k-bit training if using quantization
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    if config.get('use_lora', True):
        lora_config = setup_lora_config(config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load and prepare dataset
    logger.info(f"Loading dataset from {config['data_path']}")
    if config['data_path'].endswith('.json'):
        dataset = load_dataset('json', data_files=config['data_path'])['train']
    else:
        dataset = load_dataset(config['data_path'])['train']
    
    # Split dataset
    train_dataset = dataset.train_test_split(test_size=0.1)['train']
    eval_dataset = dataset.train_test_split(test_size=0.1)['test']
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        learning_rate=config.get('learning_rate', 2e-4),
        num_train_epochs=config.get('num_train_epochs', 3),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        warmup_ratio=config.get('warmup_ratio', 0.1),
        weight_decay=config.get('weight_decay', 0.01),
        logging_steps=config.get('logging_steps', 10),
        save_steps=config.get('save_steps', 500),
        eval_steps=config.get('eval_steps', 500),
        save_total_limit=config.get('save_total_limit', 3),
        evaluation_strategy="steps",
        logging_dir="./logs",
        fp16=config.get('fp16', True),
        gradient_checkpointing=config.get('gradient_checkpointing', True),
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        remove_unused_columns=config.get('remove_unused_columns', False),
        report_to=config.get('report_to', "tensorboard"),
        run_name=config.get('run_name', "elyza-android-tone-finetune"),
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config['output_dir'])
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()