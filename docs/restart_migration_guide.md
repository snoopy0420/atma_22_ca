# リスタート対応修正ガイド

## 修正概要

atmaCup #22がリスタートし、テストデータの形式が変更されました。本ドキュメントでは実施した修正内容をまとめます。

## 主な変更点

### 1. データ構造の変更

**リスタート前:**
- テストデータ: 完全な画像 + bbox情報
- 予測時に画像を読み込んでbbox切り出し

**リスタート後:**
- テストデータ: **bbox領域を切り抜いた画像のみ**
- `crops/` ディレクトリに保存
- `test_meta.csv` に `rel_path` カラムが追加

### 2. 実施した修正

#### 2.1 configs/config.py
新しいパス定義を追加:

```python
DIR_CROPS = os.path.join(DIR_INPUT, 'crops')
DIR_META = os.path.join(DIR_INPUT, 'atmaCup22_2nd_meta')

FILE_TRAIN_META = os.path.join(DIR_META, 'train_meta.csv')
FILE_TEST_META = os.path.join(DIR_META, 'test_meta.csv')
```

#### 2.2 src/dataset_image.py
**変更内容:**
- `BasketballDataset.__init__()` に `use_crops` パラメータを追加
- `_load_and_crop()` メソッドを修正:
  - `use_crops=True` かつ `rel_path` が存在する場合、切り抜き済み画像を直接読み込み
  - それ以外（学習データ）は従来通り完全な画像からbbox切り出し

```python
def __init__(self, df: pd.DataFrame, transform=None, is_train=True, use_crops=False):
    """
    Args:
        use_crops: テストデータで切り抜き済み画像(crops)を使用するか
    """
    self.use_crops = use_crops
```

#### 2.3 src/model_resnet_base.py
**変更内容:**
- `_extract_features_with_dataloader()` で split に応じて `use_crops` を設定
- テストデータ (`split in ['test', 'valid']`) の場合は `use_crops=True`

```python
use_crops = (split in ['test', 'valid'])
dataset = BasketballDataset(df, transform=transform, is_train=is_train, use_crops=use_crops)
```

#### 2.4 Notebooks修正
すべてのnotebookで以下の修正を実施:

**変更前:**
```python
df_train = pd.read_csv(os.path.join(DIR_INPUT, 'atmaCup22_metadata', 'train_meta.csv'))
df_test = pd.read_csv(os.path.join(DIR_INPUT, 'atmaCup22_metadata', 'test_meta.csv'))
```

**変更後:**
```python
df_train = pd.read_csv(FILE_TRAIN_META)
df_test = pd.read_csv(FILE_TEST_META)
```

修正対象ノートブック:
- `notebooks/eda.ipynb`
- `notebooks/exp_knn.ipynb`
- `notebooks/exp_knn_local.ipynb`
- `notebooks/exp_resnet.ipynb`
- `notebooks/validate_cv_leak.ipynb`

#### 2.5 その他
- `analyze_cv_strategy.py`: メタデータパスを `atmaCup22_2nd_meta/` に修正
- `README.md`: リスタート後のデータ構造を説明するセクションを追加

## 動作確認

修正後、以下の動作を確認してください:

1. **学習データ**: 従来通り完全な画像からbbox切り出し
2. **テストデータ**: 切り抜き済み画像を直接読み込み
3. **特徴抽出**: 両方で正常に動作

## テスト実行例

```python
from configs.config import *
import pandas as pd
from src.dataset_image import BasketballDataset, get_transforms

# テストデータ読み込み
df_test = pd.read_csv(FILE_TEST_META)
print(f"Test data columns: {df_test.columns.tolist()}")
print(f"Has rel_path: {'rel_path' in df_test.columns}")

# Dataset作成（切り抜き済み画像を使用）
transform = get_transforms(is_train=False)
dataset = BasketballDataset(df_test, transform=transform, is_train=False, use_crops=True)

# 1サンプル読み込み
img = dataset[0]
print(f"Image shape: {img.shape}")  # torch.Size([3, 224, 224])
```

## 互換性

- 既存のモデル（学習済み特徴量）はそのまま使用可能
- CV戦略・評価ロジックは変更不要
- リスタート前のコードとの互換性を維持
