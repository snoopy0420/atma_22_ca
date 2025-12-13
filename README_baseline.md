# atmaCup#22 Baseline - Basketball Player Identification

## 概要
このベースラインは、事前学習済みResNet50を使った特徴量抽出とKNN分類による、シンプルながら効果的な実装です。Shopee Product Matching 1位解法の知見を参考にしています。

## 実装内容

### ファイル構成
```
notebooks/
└── baseline_experiment.ipynb  # 実験用ノートブック（推奨実行方法）

src/
├── dataset.py           # データ読み込み・bbox切り出し
├── feature_extractor.py # ResNet50特徴抽出
├── predictor.py         # KNN予測・提出ファイル生成
└── run_baseline.py      # メイン実行スクリプト（CLI・バッチ処理用）
```

### 主な機能
1. **データ処理**: bbox領域を切り出して224x224にリサイズ
2. **特徴抽出**: ResNet50の最終層手前の特徴量を使用（L2正規化済み）
3. **予測方法**:
   - **Prototype法**: 各クラスの平均特徴量との類似度で判定（高速）
   - **KNN法**: 全訓練データとの類似度でk近傍投票
4. **Unknown判定**: 閾値ベースで類似度が低い場合は-1を出力

## 実行方法

### 推奨: Notebookで実行
```bash
jupyter notebook notebooks/baseline_experiment.ipynb
```

Notebook内でパラメータを調整しながら、インタラクティブに実験できます:
- セルごとに実行可能
- 結果の可視化が容易
- パラメータチューニングが簡単
- ラベル分布や類似度の分析が可能

### CLI実行（バッチ処理用）
```bash
# Prototype法（推奨: 高速）
python -m src.run_baseline

# KNN法（より正確だが低速）
python -m src.run_baseline --method knn --k 5
```

### パラメータ調整
```bash
# 閾値を調整（Unknown判定の厳しさ）
python -m src.run_baseline --threshold 0.6 --min2_threshold 0.4

# モデルを変更
python -m src.run_baseline --model efficientnet_b0

# バッチサイズ調整（GPU メモリに応じて）
python -m src.run_baseline --batch_size 64

# キャッシュを使わず最初から実行
python -m src.run_baseline --no_cache
```

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|----------|----------|------|
| `--model` | `resnet50` | 特徴抽出モデル (`resnet50`, `efficientnet_b0`) |
| `--method` | `prototype` | 予測方法 (`prototype`, `knn`) |
| `--k` | `5` | KNN法のk値 |
| `--threshold` | `0.5` | 類似度閾値（これ以上なら予測を採用） |
| `--min2_threshold` | `0.3` | Min2閾値（2番目に近い選手の最低閾値） |
| `--batch_size` | `32` | バッチサイズ |
| `--use_cache` | `True` | 特徴量のキャッシュ使用 |

## 期待される性能

- **実行時間**: 初回 30-60分（特徴抽出）、2回目以降 1-3分（キャッシュ使用）
- **Public LB**: 0.3-0.4程度（十分なベースライン）

## 改善のヒント

### 短期的改善（すぐ試せる）
1. **閾値調整**: `--threshold` と `--min2_threshold` を変更
2. **KNN法を試す**: より多くの訓練データを参照
3. **Top視点の活用**: test_top_meta.csv の情報を統合

### 中期的改善（数日）
1. **ArcFace導入**: Metric Learningで特徴量の質を向上
2. **Data Augmentation**: Cutmix, HorizontalFlip等
3. **アンサンブル**: 複数モデルの予測を統合

### 長期的改善（1週間以上）
1. **INB (Iterative Neighborhood Blending)**: Shopee 1位の秘密兵器
2. **マルチビュー統合**: side + top の Late Fusion
3. **時系列情報**: 同一quarter/session内の連続性を活用

## トラブルシューティング

### GPU メモリエラー
```bash
# バッチサイズを小さく
python -m src.run_baseline --batch_size 16
```

### 特徴抽出が遅い
```bash
# キャッシュが有効か確認
# 2回目以降は自動的にキャッシュを使用
python -m src.run_baseline --use_cache
```

### 画像が見つからないエラー
```bash
# data/raw/input/images/ に画像があるか確認
# パスの設定を configs/config.py で確認
```

## 参考資料
- [Shopee Product Matching 1st Place Solution](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136)
- [Google Landmark Recognition](https://www.kaggle.com/competitions/landmark-recognition-2021)
- [ArcFace論文](https://arxiv.org/abs/1801.07698)
