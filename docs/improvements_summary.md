# 画像コンペ改良まとめ

## 実施した改良

### ✅ 1. FeatureCache統合（最優先）
**問題**: パラメータ調整のたびに30分の特徴抽出を繰り返す

**改良**:
- `src/feature_cache.py`でキャッシュ自動管理
- データ+パラメータからキャッシュキーを自動生成
- 2回目以降は1秒未満で特徴量読み込み

**効果**:
```python
# 初回: 30分
features = model.extract_features(df)

# 2回目以降: <1秒
features = cache.load(cache_key)  # 超高速！
```

### ✅ 2. DataLoader統合
**問題**: 全画像を一度にメモリに読み込む → メモリ不足リスク

**改良**:
- `src/dataset_image.py`でPyTorch標準のDataset/DataLoader
- バッチごとに読み込み（メモリ効率向上）
- `num_workers`でマルチプロセス並列化

**効果**:
```python
# 改良前: 全24,920枚を一度にメモリへ
images = [load(i) for i in range(24920)]  # メモリ爆発

# 改良後: 32枚ずつバッチ処理
loader = DataLoader(dataset, batch_size=32, num_workers=4)
for batch in loader:  # メモリ効率的
    process(batch)
```

### ✅ 3. 類似度計算と閾値適用の分離
**問題**: 閾値を変えるたびに予測全体を実行

**改良**:
- `_compute_similarities()`: 類似度計算（重い）
- `_apply_threshold()`: 閾値適用（軽い）
- `predict_with_custom_threshold()`: 類似度キャッシュ再利用

**効果**:
```python
# 初回: 類似度計算（5分）
similarities = model.compute_similarities(test)

# 複数閾値で高速実験（各<1秒）
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    predictions = apply_threshold(similarities, threshold)  # 一瞬！
```

### ✅ 4. Data Augmentation準備
**追加**:
- `get_transforms()`で段階的な拡張レベル
- `none`, `light`, `medium`, `heavy`
- Phase 1（特徴抽出）: `none`
- Phase 2（ArcFace学習）: `medium`

## 使い方の変化

### 改良前
```python
# 毎回30分待つ...
for threshold in [0.3, 0.4, 0.5]:
    model.threshold = threshold
    predictions = model.predict(test)  # 毎回30分
    score = evaluate(predictions)
```

### 改良後
```python
# 初回のみ30分
predictions = model.predict(test)  # 特徴抽出 + キャッシュ保存

# 2回目以降は閾値チューニングが超高速
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = model.predict_with_custom_threshold(threshold)  # <1秒
    score = evaluate(preds)
```

## パラメータの変更点

### model_resnet.py
```python
params = {
    'model_name': 'resnet50',
    'method': 'prototype',
    'k': 5,
    'threshold': 0.5,
    'min2_threshold': 0.3,
    'batch_size': 32,
    'num_workers': 4,      # ← NEW: マルチプロセス並列化
    'use_cache': True,     # ← NEW: キャッシュ使用
}
```

## 新しいNotebookセクション

### Section 8: 閾値チューニング
- 複数の閾値組み合わせを高速テスト
- ヒートマップで可視化
- 最適パラメータ自動選択

### Section 9: キャッシュ管理
- 利用可能なキャッシュ一覧表示
- キャッシュクリア機能

## ファイル構成

```
src/
├── model.py                 # 抽象クラス（変更なし）
├── model_resnet.py          # ✅ 改良: DataLoader + FeatureCache統合
├── runner.py                # 変更なし
├── util.py                  # 変更なし
├── dataset_image.py         # ✅ NEW: Dataset/DataLoader/Transforms
└── feature_cache.py         # ✅ NEW: キャッシュ自動管理

notebooks/
└── exp_resnet.ipynb         # ✅ 改良: Section 8-9追加

docs/
└── image_competition_best_practices.md  # ✅ NEW: ベストプラクティス
```

## 今後の改善ロードマップ

### Phase 1.5: 現在のベースライン最適化
- [x] FeatureCache統合
- [x] DataLoader統合
- [x] 閾値チューニング機能
- [ ] 最適閾値でCV全体を再実行
- [ ] EfficientNet-B0で実験

### Phase 2: ArcFace導入（中期）
- [ ] `model_arcface.py`作成
- [ ] Data Augmentation有効化（`medium`レベル）
- [ ] End-to-End学習
- [ ] Fine-tuning戦略

### Phase 3: 高度な技術（長期）
- [ ] Multi-view統合（side + top）
- [ ] TTA (Test Time Augmentation)
- [ ] アンサンブル戦略
- [ ] 時系列情報の活用

## まとめ

### 改良のメリット
1. **時間短縮**: 30分 → <1秒（2回目以降）
2. **メモリ効率**: バッチ処理で安定動作
3. **実験速度**: 閾値チューニングが超高速
4. **標準的**: 他の画像コンペと同じパターン

### テーブルコンペとの違い
| 観点 | テーブル | 画像（改良後） |
|------|---------|---------------|
| データ読み込み | DataFrame一括 | DataLoader（バッチ） |
| 重い処理 | 特徴量作成 | 特徴抽出（キャッシュ対応） |
| パラメータ調整 | 再学習 | キャッシュ再利用（超高速） |
| メモリ使用量 | 一定 | バッチサイズで制御 |
| 並列化 | joblib | DataLoader(num_workers) |

画像コンペでは**計算時間とメモリ効率**が最重要。今回の改良でその両方を大幅改善しました！
