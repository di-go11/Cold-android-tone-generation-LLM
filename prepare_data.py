#!/usr/bin/env python3
"""
Data Preparation Script for ELYZA LLM Fine-tuning

This script processes raw text data and converts it into the instruction format
required for fine-tuning with cold android tone examples.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_android_tone_examples() -> List[Dict[str, str]]:
    """Create sample data with cold android tone examples."""
    examples = [
        {
            "instruction": "以下の文章を冷たいアンドロイド調に変換してください。",
            "input": "おはようございます。今日も一日よろしくお願いします。",
            "output": "システム起動完了。本日の業務処理を開始します。効率的な作業遂行を期待します。"
        },
        {
            "instruction": "感情を込めずに事実のみを述べてください。",
            "input": "このプロジェクトは素晴らしい成果を上げています！",
            "output": "当該プロジェクトは設定された目標値を上回る数値を記録しています。データに基づく客観的評価です。"
        },
        {
            "instruction": "機械的な口調で状況を報告してください。",
            "input": "会議が始まります。皆さん、準備はいかがですか？",
            "output": "会議システム開始。全参加者の準備状況確認を実行中。応答を待機しています。"
        },
        {
            "instruction": "アンドロイドのように冷静に対応してください。",
            "input": "急いでこの作業を終わらせる必要があります！",
            "output": "緊急度レベルを認識しました。最適化されたプロセスで作業を実行します。感情的要素は処理対象外です。"
        },
        {
            "instruction": "人間らしい感情を排除して回答してください。",
            "input": "今日はとても楽しい一日でした。",
            "output": "本日の活動記録を更新しました。満足度パラメータは測定範囲外です。"
        },
        {
            "instruction": "システム的な観点から説明してください。",
            "input": "チームワークが大切ですね。",
            "output": "複数ユニット間の連携効率が全体パフォーマンスに直結します。協調機能は必要な仕様です。"
        },
        {
            "instruction": "機械的で感情のない返答をしてください。",
            "input": "失敗してしまって申し訳ありません。",
            "output": "エラー発生を確認しました。原因分析と修正プロセスを実行します。謝罪機能は搭載されていません。"
        },
        {
            "instruction": "アンドロイドのような冷淡な口調で応答してください。",
            "input": "ありがとうございます。助かりました。",
            "output": "処理完了を確認。感謝の概念は理解できませんが、タスクは正常に終了しました。"
        },
        {
            "instruction": "感情を含まない事務的な口調で話してください。",
            "input": "みんなで協力して頑張りましょう！",
            "output": "全メンバーによる協調動作を開始します。最大効率での目標達成を目指します。"
        },
        {
            "instruction": "冷たく機械的な応答をしてください。",
            "input": "お疲れ様でした。今日は本当にお世話になりました。",
            "output": "本日の業務終了を確認。予定されたタスクは全て実行されました。システム待機状態に移行します。"
        }
    ]
    return examples


def process_text_file(input_file: str) -> List[Dict[str, str]]:
    """Process a text file and convert it to instruction format."""
    examples = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Simple processing: each line becomes an instruction
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                examples.append({
                    "instruction": "以下の文章を冷たいアンドロイド調に変換してください。",
                    "input": line,
                    "output": f"処理対象テキスト#{i+1}を受信しました。変換処理を実行してください。"
                })
    
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}")
        return []
    
    return examples


def save_dataset(examples: List[Dict[str, str]], output_file: str):
    """Save the processed examples to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(examples)} examples to {output_file}")
    except Exception as e:
        logger.error(f"Error saving file {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for ELYZA LLM fine-tuning")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input text file to process (optional)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/sample_data.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="Create sample data with android tone examples"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    
    # Create sample data
    if args.create_sample or not args.input_file:
        logger.info("Creating sample android tone data...")
        examples.extend(create_android_tone_examples())
    
    # Process input file if provided
    if args.input_file:
        logger.info(f"Processing input file: {args.input_file}")
        file_examples = process_text_file(args.input_file)
        examples.extend(file_examples)
    
    if examples:
        save_dataset(examples, args.output_file)
        logger.info(f"Data preparation completed. Total examples: {len(examples)}")
    else:
        logger.warning("No examples were created. Please check your input.")


if __name__ == "__main__":
    main()