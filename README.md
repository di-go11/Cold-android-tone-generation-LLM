# ELYZA LLM Fine-tuning for Cold Android Tone Generation

ELYZAのLLMをファインチューニングして、クールなアンドロイド調のテキストを生成するためのプロジェクトです。

## 概要

このプロジェクトは、ELYZA社の日本語LLMを使用して、冷たく機械的なアンドロイド風の文体でテキストを生成できるようにファインチューニングするためのツールセットです。

## 特徴

- **ELYZA LLMベース**: ELYZA社の高性能日本語LLMを使用
- **効率的なファインチューニング**: LoRAやQLoRAを使用したメモリ効率的な学習
- **カスタマイズ可能**: トーンや文体を調整可能
- **簡単なセットアップ**: 数ステップで学習環境を構築

## 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨）
- 16GB以上のRAM（推奨）

## インストール

```bash
git clone https://github.com/di-go11/Cold-android-tone-generation-LLM.git
cd Cold-android-tone-generation-LLM
pip install -r requirements.txt
```

## 使用方法

### 1. データ準備

```bash
python prepare_data.py --input_file your_data.txt --output_file processed_data.json
```

### 2. ファインチューニング

```bash
python finetune.py --config config/training_config.yaml
```

### 3. テキスト生成

```bash
python generate.py --model_path ./output/model --prompt "あなたの質問をここに入力"
```

## ファインチューニング手法

### 1. LoRA (Low-Rank Adaptation)
- メモリ効率的なファインチューニング手法
- 元のモデルパラメータを凍結し、小さなアダプターレイヤーのみを学習
- 学習時間とメモリ使用量を大幅に削減

### 2. QLoRA (Quantized LoRA)
- LoRAをさらに効率化した手法
- 4bit量子化と組み合わせて使用
- 限られたGPUメモリでも大規模モデルの学習が可能

### 3. データ形式
```json
{
  "instruction": "ユーザーの指示",
  "input": "入力テキスト",
  "output": "冷たいアンドロイド調の出力"
}
```

## 設定ファイル

`config/training_config.yaml`で学習パラメータを調整できます：

```yaml
model_name: "elyza/ELYZA-japanese-Llama-2-7b-instruct"
output_dir: "./output"
learning_rate: 2e-4
batch_size: 4
epochs: 3
lora_r: 8
lora_alpha: 32
```

## サンプルデータ

プロジェクトには冷たいアンドロイド調のサンプルデータが含まれています：

- `data/sample_data.json`: 学習用サンプルデータ
- `data/prompts.txt`: テスト用プロンプト集

## トラブルシューティング

### よくある問題

1. **GPUメモリ不足**
   - `batch_size`を小さくする
   - QLoRAを使用する
   - `gradient_checkpointing`を有効にする

2. **学習が進まない**
   - 学習率を調整する
   - データの品質を確認する
   - エポック数を増やす

### ログとモニタリング

学習の進行状況は以下で確認できます：
- TensorBoard: `tensorboard --logdir ./logs`
- Wandb（オプション）: 設定ファイルで有効化

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 参考文献

- [ELYZA](https://elyza.ai/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)