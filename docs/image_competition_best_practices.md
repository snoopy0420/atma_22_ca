# 画像コンペ特有のベストプラクティス

このドキュメントでは、テーブルコンペとは異なる**画像コンペ特有の改良点**をまとめます。

## 1. Dataset/DataLoaderの活用

### ❌ 悪い例（テーブルコンペの流儀）
```python
# 全画像を一度にメモリに読み込む
images = []
for idx in range(len(df)):
    img = load_image(df.iloc[idx])
    images.append(img)

features = model.extract(images)  # メモリ不足の危険
```

### ✅ 良い例（画像コンペの流儀）
```python
from torch.utils.data import DataLoader
from src.dataset_image import BasketballDataset

# Datasetでバッチごとに読み込み
dataset = BasketballDataset(df, transform=get_transforms())
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

for batch_images, batch_labels in dataloader:
    features = model.extract(batch_images)  # バッチ処理
```

**メリット**:
- メモリ効率: 一度に全データを読み込まない
- 並列化: `num_workers`でマルチプロセス読み込み
- GPU転送の最適化: `pin_memory=True`で高速化

---

## 2. 特徴量キャッシュ戦略

### 画像コンペの課題
- 特徴抽出が**非常に重い**（ResNet50で1画像=数十ms）
- 24,920枚 × 数十ms = 10-30分
- パラメータチューニングで何度も実行 → **時間の無駄**

### 解決策: FeatureCacheクラス
```python
from src.feature_cache import FeatureCache

cache = FeatureCache()
cache_key = cache.get_cache_key(df_train, 'resnet50', params)

if cache.exists(cache_key, 'train'):
    # キャッシュから高速読み込み（1秒未満）
    train_features, train_labels = cache.load(cache_key, 'train')
else:
    # 初回のみ特徴抽出（10-30分）
    train_features = extract_features(df_train)
    cache.save(cache_key, train_features, train_labels, 'train')
```

**メリット**:
- 2回目以降は**1秒未満**で特徴量取得
- 異なるパラメータは自動的に別キャッシュ
- ディスク容量: 24,920 × 2048次元 × 4byte = 約200MB（許容範囲）

---

## 3. Data Augmentation（必須）

### 画像コンペでは必須技術
テーブルコンペにはない概念。画像を人工的に増やして汎化性能向上。

### 実装例
```python
from src.dataset_image import get_transforms

# 訓練時: Data Augmentation ON
train_transform = get_transforms(is_train=True, augmentation_level='medium')
train_dataset = BasketballDataset(df_train, transform=train_transform)

# テスト時: Augmentation OFF（固定前処理のみ）
test_transform = get_transforms(is_train=False)
test_dataset = BasketballDataset(df_test, transform=test_transform)
```

### Augmentationレベル
| レベル | 変換内容 | 用途 |
|--------|---------|------|
| `none` | リサイズ、正規化のみ | 特徴抽出のみ（今回のベースライン） |
| `light` | + 水平反転 | 手軽に汎化性能向上 |
| `medium` | + ColorJitter, Affine | バランス型（推奨） |
| `heavy` | + RandomCrop, Grayscale | 過学習が強い場合 |

**注意**: 特徴抽出ベースライン（今回）では`none`が正解。ArcFace学習時に`medium`以上を使う。

---

## 4. モデル構成の分離

### 画像コンペの2段階アプローチ

```
Phase 1: 特徴抽出（現在のベースライン）
  ResNet50（事前学習済み） → 2048次元特徴量 → KNN/Prototype分類

Phase 2: End-to-End学習（ArcFace等）
  ResNet50（Fine-tuning） → FC層 → ArcFace Loss → 学習
```

### ファイル構成の改善案
```
src/
├── model.py                    # 抽象クラス
├── model_feature_extractor.py  # Phase 1: 特徴抽出専用
├── model_arcface.py            # Phase 2: ArcFace学習専用
├── dataset_image.py            # Dataset/DataLoader
└── feature_cache.py            # キャッシュ管理
```

---

## 5. メトリクス計算の効率化

### テーブルコンペとの違い
- テーブル: 予測は一瞬、メトリクス計算も一瞬
- 画像: 予測に時間がかかる、**キャッシュした予測結果で何度もメトリクス計算**したい

### 実装例
```python
# 予測結果を保存
Util.dump(va_predictions, 'models/run_name/fold0_va_pred.pkl')

# 後から異なる閾値でメトリクス再計算（予測不要）
va_pred = Util.load('models/run_name/fold0_va_pred.pkl')
for threshold in [0.3, 0.4, 0.5, 0.6]:
    # 閾値だけ変えて再評価（高速）
    predictions = apply_threshold(va_pred, threshold)
    score = metric(va_true, predictions)
```

---

## 6. TTA (Test Time Augmentation)

### 画像コンペ特有のテクニック
テスト時にも複数のAugmentationを適用して予測をアンサンブル。

```python
# 元画像
pred1 = model.predict(test_image)

# 水平反転
pred2 = model.predict(horizontal_flip(test_image))

# 最終予測（平均）
final_pred = (pred1 + pred2) / 2
```

**効果**: +1-2%の精度向上（コストは2倍の推論時間）

---

## 7. 推奨する改良の優先順位

### 現在のベースライン（Phase 1: 特徴抽出）での改良
1. **FeatureCacheの導入** ← 最優先（時間短縮）
2. **DataLoaderの活用** ← メモリ効率改善
3. **閾値チューニング** ← キャッシュ予測結果で高速実験

### Phase 2（ArcFace学習）での改良
4. **Data Augmentation** ← 汎化性能向上
5. **TTA** ← 最終スコア向上
6. **Multi-view統合** ← side/topの情報統合

---

## 8. 実装済みファイル

| ファイル | 説明 | 優先度 |
|---------|------|--------|
| `src/dataset_image.py` | Dataset/DataLoader、Transforms | ⭐⭐⭐ |
| `src/feature_cache.py` | 特徴量キャッシュ管理 | ⭐⭐⭐ |
| `src/model_resnet.py` | 現行モデル（改良余地あり） | ⭐⭐ |

---

## まとめ

### テーブルコンペ vs 画像コンペ

| 観点 | テーブルコンペ | 画像コンペ |
|------|---------------|-----------|
| データ読み込み | DataFrame一括 | Dataset/DataLoaderでバッチ |
| 特徴量 | 手動作成 | CNNで自動抽出 |
| 重い処理 | 特徴量作成 | 特徴抽出・学習 |
| キャッシュ | 特徴量CSV | 特徴量npy + モデルpkl |
| Augmentation | なし | **必須**（学習時） |
| TTA | なし | あり（テスト時） |

画像コンペでは**計算時間とメモリ効率**が最重要課題。FeatureCacheとDataLoaderの活用が成功の鍵！
