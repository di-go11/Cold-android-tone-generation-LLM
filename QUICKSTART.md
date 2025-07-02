# クイックスタートガイド

ELYZAのLLMをファインチューニングして冷たいアンドロイド調のテキストを生成するためのクイックスタートガイドです。

## 1. 環境セットアップ (約5分)

```bash
# リポジトリをクローン
git clone https://github.com/di-go11/Cold-android-tone-generation-LLM.git
cd Cold-android-tone-generation-LLM

# 自動セットアップを実行
bash setup.sh

# または手動で設定
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. データ準備 (約1分)

サンプルデータは既に用意されていますが、独自のデータを使用する場合：

```bash
# 独自データを準備
python prepare_data.py --input_file your_data.txt --output_file data/custom_data.json

# サンプルデータを確認
python prepare_data.py --create_sample
```

## 3. ファインチューニング実行

### 基本的な実行 (GPU必須、約30分〜2時間)

```bash
python finetune.py --config config/training_config.yaml
```

### メモリ不足の場合の対処

`config/training_config.yaml`を編集：

```yaml
# バッチサイズを小さく
per_device_train_batch_size: 2

# QLoRAを有効化
use_4bit: true

# グラディエントチェックポイントを有効化
gradient_checkpointing: true
```

## 4. テキスト生成テスト

### インタラクティブモード

```bash
python generate.py --interactive
```

### 単発生成

```bash
python generate.py --prompt "以下の文章を冷たいアンドロイド調に変換してください。" --input "おはようございます"
```

## 5. モデル評価

```bash
python evaluate.py --model_path ./output --test_data data/sample_data.json
```

## よくある問題と解決法

### 🚨 GPU メモリ不足

```yaml
# config/training_config.yaml
per_device_train_batch_size: 1  # さらに小さく
use_4bit: true                   # 量子化を有効化
gradient_accumulation_steps: 8   # 勾配蓄積を増やす
```

### 🚨 CUDA エラー

```bash
# CPU のみで実行
export CUDA_VISIBLE_DEVICES=""
python finetune.py --config config/training_config.yaml
```

### 🚨 依存関係エラー

```bash
# 仮想環境を再作成
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 高度な使用法

### カスタム設定でのファインチューニング

```bash
# 独自の設定ファイルを作成
cp config/training_config.yaml config/my_config.yaml

# 設定を編集してから実行
python finetune.py --config config/my_config.yaml
```

### 事前学習済みモデルからの継続学習

```bash
python finetune.py --config config/training_config.yaml --resume_from_checkpoint ./output/checkpoint-500
```

## パフォーマンス最適化

### 学習速度向上

```yaml
# より大きなバッチサイズ（GPU メモリが十分な場合）
per_device_train_batch_size: 8
gradient_accumulation_steps: 1

# 混合精度学習
fp16: true

# データローダーの並列化
dataloader_num_workers: 8
```

### 品質向上

```yaml
# より多くのエポック
num_train_epochs: 5

# より小さな学習率
learning_rate: 1e-4

# より大きな LoRA rank
lora_r: 16
lora_alpha: 64
```

## 次のステップ

1. **データの品質向上**: より多くの高品質なアンドロイド調のサンプルを収集
2. **モデルの評価**: 生成されたテキストの品質を定量的に評価
3. **本番デプロイ**: 学習済みモデルをAPIサーバーとしてデプロイ
4. **継続的な改善**: ユーザーフィードバックを基にした継続学習

## サポート

問題が発生した場合は、以下を確認してください：

1. `python test_setup.py` でセットアップを検証
2. ログファイル (`logs/` ディレクトリ) を確認
3. GitHub Issues で質問を投稿