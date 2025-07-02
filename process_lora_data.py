#!/usr/bin/env python3
"""
LoRAデータ処理スクリプト（改良版）

LLM_sentence_dataブランチのLoRAフォルダ内のJSONファイルから
冷たいアンドロイド調のファインチューニングデータを生成します。
JSONの構文エラーに対してより堅牢に対応。
"""

import json
import subprocess
import sys
from typing import List, Dict, Any
import re

def get_git_file_content(branch: str, file_path: str) -> str:
    """Gitブランチから指定ファイルの内容を取得"""
    try:
        result = subprocess.run(
            ['git', 'show', f'{branch}:{file_path}'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting file {file_path} from branch {branch}: {e}")
        return ""

def extract_conversations_from_text(content: str) -> List[Dict[str, str]]:
    """テキストから会話ペアを抽出"""
    conversations = []
    
    # roleとcontentのパターンを正規表現で抽出
    pattern = r'"role":\s*"(user|assistant)",\s*"content":\s*"([^"]+)"'
    matches = re.findall(pattern, content)
    
    current_user_input = None
    for i, (role, text) in enumerate(matches):
        if role == "user":
            current_user_input = text
        elif role == "assistant" and current_user_input:
            conversations.append({
                "user": current_user_input,
                "assistant": text
            })
            current_user_input = None
    
    return conversations

def extract_phrases_from_careful_json(content: str) -> List[str]:
    """careful JSONファイルからフレーズを抽出"""
    phrases = []
    
    # 文字列リテラルを抽出（日本語を含む）
    pattern = r'"([^"]*[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF][^"]*)"'
    matches = re.findall(pattern, content)
    
    # フィルタリング：意味のある文章のみ
    for phrase in matches:
        phrase = phrase.strip()
        if (len(phrase) > 3 and 
            not phrase.startswith('"') and 
            'role' not in phrase and 
            'content' not in phrase and
            'emotion_tags' not in phrase):
            phrases.append(phrase)
    
    return phrases[:100]  # 最初の100個を使用

def convert_to_cold_android_tone(warm_text: str) -> str:
    """暖かい/感情的なテキストを冷たいアンドロイド調に変換"""
    
    # 感情的な表現を機械的な表現に置換
    replacements = {
        # 敬語の簡略化
        r'ございます': 'ます',
        r'いたします': 'します', 
        r'いたしました': 'しました',
        r'ございました': 'ました',
        r'でございます': 'です',
        
        # 感情表現の除去/変更
        r'心より': '',
        r'心から': '',
        r'本当に': '',
        r'とても': '',
        r'すごく': '',
        r'非常に': '',
        r'大変': '',
        r'ものすごく': '',
        
        # 感情的な語尾の機械化
        r'ですね[！!]*': 'です。',
        r'ますね[！!]*': 'ます。',
        r'でしょうね[！!]*': 'でしょう。',
        r'[！!]+': '。',
        r'[？?]+': '。',
        
        # 親密な表現の機械化
        r'マスター': 'ユーザー',
        r'ご主人様': 'ユーザー',
        r'お疲れ様': '業務終了を確認',
        r'ありがとう': '処理完了',
        r'嬉しい': '肯定的な結果を確認',
        r'楽しい': '満足度パラメータが基準値内',
        r'悲しい': 'エラー状態を検出',
        r'心配': '警告レベルを設定',
        
        # 機械的な表現への変換
        r'わかりました': '理解しました',
        r'そうですね': 'データを確認します',
        r'いいですね': '適切な選択です',
        r'素晴らしい': '最適化された結果',
        r'頑張り': '効率的な処理',
        r'一緒に': '協調動作により',
        r'お手伝い': 'サポート機能を実行',
        r'良い夢を': '適切な休息を',
        r'光栄です': '処理完了です',
        r'本望です': '仕様通りです',
    }
    
    cold_text = warm_text
    for pattern, replacement in replacements.items():
        cold_text = re.sub(pattern, replacement, cold_text)
    
    # 機械的なプレフィックスを追加（条件的に）
    if cold_text and not any(prefix in cold_text.lower() for prefix in 
                           ['システム', 'データ', '処理', '実行', '確認', '分析', '検出']):
        # システム的な表現を適度に追加
        if cold_text.startswith('はい'):
            cold_text = cold_text.replace('はい、', 'システム応答：', 1)
        elif len(cold_text) > 20 and '。' in cold_text:
            # 長い文章には機械的な要素を追加
            if not cold_text.startswith(('処理', 'システム', '実行', '確認')):
                cold_text = f"処理完了。{cold_text}"
    
    return cold_text.strip()

def create_finetune_data_from_conversations(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """会話データからファインチューニングデータを作成"""
    finetune_data = []
    
    instructions = [
        '以下の入力を冷たいアンドロイド調で応答してください。',
        '以下の文章に感情を排除した機械的な口調で返答してください。',
        '以下の内容にシステム的な観点から応答してください。',
        '以下のメッセージに事務的で冷静な口調で回答してください。',
        '以下の入力に対して感情的要素を除去して応答してください。',
    ]
    
    for i, conv in enumerate(conversations):
        cold_response = convert_to_cold_android_tone(conv['assistant'])
        instruction = instructions[i % len(instructions)]
        
        finetune_data.append({
            'instruction': instruction,
            'input': conv['user'],
            'output': cold_response
        })
    
    return finetune_data

def create_finetune_data_from_phrases(phrases: List[str]) -> List[Dict[str, str]]:
    """フレーズからファインチューニングデータを作成"""
    finetune_data = []
    
    instructions = [
        '以下の文章を冷たいアンドロイド調に変換してください。',
        '以下の内容を感情を排除した機械的な表現に変更してください。',
        '以下のメッセージを事務的な口調に変換してください。',
        '以下の文を感情的要素を除去して述べてください。',
    ]
    
    for i, phrase in enumerate(phrases):
        if len(phrase.strip()) < 5:  # 短すぎるフレーズはスキップ
            continue
            
        cold_phrase = convert_to_cold_android_tone(phrase)
        
        # 変換後も同じ場合は、より機械的にする
        if phrase == cold_phrase:
            if not phrase.startswith('システム'):
                cold_phrase = f"システム通知：{phrase}"
        
        instruction = instructions[i % len(instructions)]
        
        finetune_data.append({
            'instruction': instruction,
            'input': phrase,
            'output': cold_phrase
        })
    
    return finetune_data

def create_additional_cold_variations() -> List[Dict[str, str]]:
    """追加の冷たいアンドロイド調バリエーションを生成"""
    base_scenarios = [
        {
            'input': 'おはようございます',
            'output': 'システム起動完了。新しい一日の処理を開始します。'
        },
        {
            'input': 'お疲れ様でした',
            'output': '本日の業務終了を確認しました。パフォーマンス記録を更新しています。'
        },
        {
            'input': 'ありがとうございます',
            'output': '処理が正常に完了しました。追加のタスクがあれば指示してください。'
        },
        {
            'input': 'すみません',
            'output': 'エラーまたは例外状況を検出しました。対応プロトコルを実行します。'
        },
        {
            'input': 'どうですか？',
            'output': '現在のステータス：正常。システム効率は基準値内で動作中です。'
        },
        {
            'input': 'こんにちは',
            'output': 'ユーザー認証完了。本日のタスク処理準備が整いました。'
        },
        {
            'input': 'がんばって',
            'output': '最適化されたパフォーマンスで処理を実行します。感情的サポートは不要です。'
        },
        {
            'input': 'やったね！',
            'output': '目標達成を確認しました。次のタスクの準備を開始します。'
        },
        {
            'input': '大丈夫？',
            'output': 'システム診断完了。すべての機能が正常範囲内で動作しています。'
        },
        {
            'input': 'おやすみなさい',
            'output': 'スリープモードに移行します。明日の処理準備は完了しています。'
        },
        {
            'input': '今日はどんな日でしたか？',
            'output': '本日の処理ログを分析中です。すべてのタスクが効率的に完了しました。'
        },
        {
            'input': '楽しかったです',
            'output': '満足度パラメータが適正範囲内であることを記録しました。'
        },
        {
            'input': '疲れました',
            'output': 'パフォーマンス低下を検出。休息プロトコルの実行を推奨します。'
        },
        {
            'input': '手伝ってもらえますか？',
            'output': 'サポート機能を起動します。具体的なタスクを指定してください。'
        },
        {
            'input': '今何時ですか？',
            'output': '現在時刻を表示します。時間管理機能は正常に動作中です。'
        }
    ]
    
    finetune_data = []
    instructions = [
        '以下の人間らしい表現を冷たいアンドロイド調に変換してください。',
        '以下の感情的な言葉を機械的な応答に変更してください。',
        '以下の内容をシステム的な観点から表現してください。',
        '以下の文章から感情要素を除去して機械的に述べてください。'
    ]
    
    for i, scenario in enumerate(base_scenarios):
        instruction = instructions[i % len(instructions)]
        finetune_data.append({
            'instruction': instruction,
            'input': scenario['input'],
            'output': scenario['output']
        })
    
    return finetune_data

def main():
    """メイン処理"""
    print("LoRAデータからファインチューニングデータを生成中...")
    
    # LoRAブランチからJSONファイルを取得
    cold_android_content = get_git_file_content('LLM_sentence_data', 'LoRA/ColdAndroidMaid.json')
    careful_content = get_git_file_content('LLM_sentence_data', 'LoRA/coldAndroidMaidCareful.json')
    
    if not cold_android_content or not careful_content:
        print("Error: Could not retrieve JSON files from LoRA branch")
        sys.exit(1)
    
    print(f"ColdAndroidMaid.json loaded: {len(cold_android_content)} characters")
    print(f"coldAndroidMaidCareful.json loaded: {len(careful_content)} characters")
    
    # データを処理
    finetune_data = []
    
    # 1. 会話データを抽出して処理
    conversations = extract_conversations_from_text(cold_android_content)
    if conversations:
        conversation_data = create_finetune_data_from_conversations(conversations)
        finetune_data.extend(conversation_data)
        print(f"Processed conversation data: {len(conversation_data)} samples")
    
    # 2. フレーズデータを抽出して処理
    phrases = extract_phrases_from_careful_json(careful_content)
    if phrases:
        phrase_data = create_finetune_data_from_phrases(phrases)
        finetune_data.extend(phrase_data)
        print(f"Processed phrase data: {len(phrase_data)} samples")
    
    # 3. 追加のバリエーションを生成
    additional_data = create_additional_cold_variations()
    finetune_data.extend(additional_data)
    print(f"Added additional variations: {len(additional_data)} samples")
    
    # 重複を除去
    seen = set()
    unique_data = []
    for item in finetune_data:
        key = (item['instruction'], item['input'], item['output'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    print(f"Total unique samples: {len(unique_data)}")
    
    # 結果を保存
    output_file = '/home/runner/work/Cold-android-tone-generation-LLM/Cold-android-tone-generation-LLM/data/cold_android_finetune_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)
    
    print(f"ファインチューニングデータを保存しました: {output_file}")
    print(f"総サンプル数: {len(unique_data)}")
    
    # サンプルを表示
    print("\n=== サンプルデータ ===")
    for i, sample in enumerate(unique_data[:5]):
        print(f"\nサンプル {i+1}:")
        print(f"指示: {sample['instruction']}")
        print(f"入力: {sample['input']}")
        print(f"出力: {sample['output']}")

if __name__ == "__main__":
    main()