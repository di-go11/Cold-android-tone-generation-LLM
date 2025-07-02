#!/usr/bin/env python3
"""
Example Script for ELYZA LLM Fine-tuning

This script demonstrates the complete workflow from data preparation
to model evaluation.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_environment():
    """Check if the environment is properly set up."""
    logger.info("Checking environment setup...")
    
    required_files = [
        'requirements.txt',
        'config/training_config.yaml',
        'data/sample_data.json'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file missing: {file_path}")
            return False
    
    # Check if we can import required packages
    try:
        import torch
        import transformers
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available. Training will be slow on CPU.")
        
    except ImportError as e:
        logger.error(f"Required package not installed: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    return True


def create_example_data():
    """Create additional example data for demonstration."""
    logger.info("Creating example data...")
    
    examples = [
        {
            "instruction": "以下の感情的な文章を冷静なアンドロイド調に変換してください。",
            "input": "わあ！この景色は本当に美しいですね！感動しています！",
            "output": "視覚センサーが良好な景観データを検出しました。美的判定パラメータは範囲外です。"
        },
        {
            "instruction": "人間的な表現を機械的な表現に変更してください。",
            "input": "頑張って最後まで諦めずにやり遂げましょう！",
            "output": "設定されたタスクを完了まで継続実行します。中断機能は無効化されています。"
        },
        {
            "instruction": "アンドロイドのような客観的な視点で説明してください。",
            "input": "このケーキは本当に美味しいです！",
            "output": "味覚センサーによる分析結果：糖度、脂質含有量が基準値内です。主観的評価は処理対象外です。"
        }
    ]
    
    # Add to existing sample data
    sample_file = Path("data/sample_data.json")
    if sample_file.exists():
        with open(sample_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        existing_data.extend(examples)
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Added {len(examples)} examples to sample data")
    else:
        logger.error("Sample data file not found")


def run_demo_training():
    """Run a demonstration training with minimal settings."""
    logger.info("Starting demo training (this may take a while)...")
    
    # Create a demo config with minimal settings
    demo_config = {
        'model_name': 'elyza/ELYZA-japanese-Llama-2-7b-instruct',
        'output_dir': './demo_output',
        'data_path': './data/sample_data.json',
        'learning_rate': 2e-4,
        'num_train_epochs': 1,  # Just 1 epoch for demo
        'per_device_train_batch_size': 1,  # Small batch size
        'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 4,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'logging_steps': 5,
        'save_steps': 50,
        'eval_steps': 50,
        'save_total_limit': 2,
        'use_lora': True,
        'lora_r': 4,  # Smaller rank for demo
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'lora_target_modules': ['q_proj', 'v_proj'],
        'use_4bit': True,  # Enable quantization to save memory
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_quant_type': 'nf4',
        'use_nested_quant': False,
        'fp16': True,
        'gradient_checkpointing': True,
        'dataloader_num_workers': 2,
        'remove_unused_columns': False,
        'report_to': 'none',  # Disable wandb for demo
        'run_name': 'elyza-demo'
    }
    
    # Save demo config
    import yaml
    demo_config_path = 'config/demo_config.yaml'
    with open(demo_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(demo_config, f, default_flow_style=False)
    
    logger.info(f"Created demo config: {demo_config_path}")
    
    # Run training
    cmd = f"python finetune.py --config {demo_config_path}"
    logger.info(f"Running: {cmd}")
    
    # Note: In a real scenario, you would run this command
    logger.info("To run the demo training, execute:")
    logger.info(f"  {cmd}")
    logger.info("\nThis will train for 1 epoch with minimal resources.")


def demo_generation():
    """Demonstrate text generation."""
    logger.info("Demo text generation...")
    
    # Check if a trained model exists
    if Path("demo_output").exists():
        logger.info("Using demo model for generation")
        model_path = "demo_output"
    else:
        logger.info("No trained model found. Showing generation command:")
        logger.info("python generate.py --model_path ./demo_output --interactive")
        return
    
    # Show example generation commands
    examples = [
        "以下の文章を冷たいアンドロイド調に変換してください。",
        "感情を排除して事実のみを述べてください。",
        "機械的な口調で応答してください。"
    ]
    
    logger.info("Example generation commands:")
    for example in examples:
        cmd = f'python generate.py --model_path {model_path} --prompt "{example}" --input "こんにちは、今日はいい天気ですね。"'
        logger.info(f"  {cmd}")


def main():
    """Main demonstration function."""
    print("🤖 ELYZA LLM Fine-tuning Demo")
    print("=" * 40)
    
    # Step 1: Check environment
    if not check_environment():
        logger.error("Environment check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Step 2: Create example data
    create_example_data()
    
    # Step 3: Run demo training
    print("\n" + "=" * 40)
    print("Demo Training Setup")
    print("=" * 40)
    run_demo_training()
    
    # Step 4: Show generation examples
    print("\n" + "=" * 40)
    print("Text Generation Demo")
    print("=" * 40)
    demo_generation()
    
    print("\n" + "=" * 40)
    print("Demo completed!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Run the training command shown above")
    print("2. Use the generation script to test your model")
    print("3. Evaluate the model with: python evaluate.py")
    print("4. For interactive generation: python generate.py --interactive")


if __name__ == "__main__":
    main()