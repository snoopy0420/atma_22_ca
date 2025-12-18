# 概要

atmaCup #22 - バスケットボール選手判別チャレンジ

## 重要: リスタート後のデータ構造

**2024年12月にコンペがリスタートし、データ構造が変更されました:**

### 変更点
1. **テストデータ**: 完全な画像ではなく、**BBox領域を切り抜いた画像(crops/)** のみ提供
2. **メタデータ**: `test_meta.csv` に `rel_path` カラムが追加され、切り抜き済み画像のパスを指定
3. **メタデータディレクトリ**: `atmaCup22_metadata/` → `atmaCup22_2nd_meta/`

### データ構造
```
data/raw/input/
├── images/                          # 学習データ: 完全な画像
├── crops/                           # テストデータ: 切り抜き済み画像
│   └── Q4-XXX/sess_XXXX/...
└── atmaCup22_2nd_meta/
    ├── train_meta.csv              # 学習データのメタ情報
    └── test_meta.csv               # テストデータのメタ情報 (rel_pathカラム追加)
```

### 実装対応
- `BasketballDataset`: `use_crops` パラメータを追加し、切り抜き済み画像に対応
- `config.py`: 新しいパス定義 (`DIR_CROPS`, `DIR_META`, `FILE_TRAIN_META`, `FILE_TEST_META`)
- Notebooks: `FILE_TRAIN_META`, `FILE_TEST_META` を使用してデータ読み込み

## 環境
python: 3.9

## 実行
