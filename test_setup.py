#!/usr/bin/env python3
"""
Test Script for ELYZA LLM Fine-tuning Setup

This script tests if all components are working correctly.
"""

import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required packages can be imported."""
    logger.info("Testing package imports...")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'yaml',
        'pandas',
        'numpy'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError as e:
            logger.error(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0


def test_file_structure():
    """Test if all required files and directories exist."""
    logger.info("Testing file structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'finetune.py',
        'generate.py',
        'prepare_data.py',
        'evaluate.py',
        'config/training_config.yaml',
        'data/sample_data.json',
        'data/prompts.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
        else:
            logger.info(f"✅ {file_path}")
    
    return len(missing_files) == 0


def test_sample_data():
    """Test if sample data is properly formatted."""
    logger.info("Testing sample data format...")
    
    try:
        with open('data/sample_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("❌ Sample data should be a list")
            return False
        
        if len(data) == 0:
            logger.error("❌ Sample data is empty")
            return False
        
        # Check first example format
        example = data[0]
        required_keys = ['instruction', 'output']
        for key in required_keys:
            if key not in example:
                logger.error(f"❌ Missing key '{key}' in sample data")
                return False
        
        logger.info(f"✅ Sample data format correct ({len(data)} examples)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading sample data: {e}")
        return False


def test_config_file():
    """Test if configuration file is properly formatted."""
    logger.info("Testing configuration file...")
    
    try:
        import yaml
        with open('config/training_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['model_name', 'output_dir', 'learning_rate']
        for key in required_keys:
            if key not in config:
                logger.error(f"❌ Missing key '{key}' in config")
                return False
        
        logger.info("✅ Configuration file format correct")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading config file: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 ELYZA LLM Fine-tuning Setup Test")
    print("===================================")
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Sample Data", test_sample_data),
        ("Configuration", test_config_file)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if not test_func():
            all_passed = False
    
    print("\n" + "="*40)
    if all_passed:
        print("🎉 All tests passed! Setup is ready.")
        print("\nYou can now run:")
        print("  python finetune.py --config config/training_config.yaml")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1)


if __name__ == "__main__":
    main()