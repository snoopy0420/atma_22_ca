# リークしない最善のCV戦略（atmaCup#22）

## 🎯 推奨戦略

### グループ化レベル: `quarter_session`

**理由**：
1. **session単位でのリーク防止**
   - 同一session内のframeは時間的に連続
   - 連続フレーム間で選手の見た目がほぼ同じ → 必ずsession単位でグループ化
   
2. **適切な粒度**
   - quarterだけでは粗すぎる（同じquarter内の異なるsessionが分かれる可能性）
   - frameレベルは細かすぎる（グループ数が多すぎてCV不可能）

3. **十分なグループ数確保**
   - 学習データ: 24,920サンプル
   - quarter_sessionの組み合わせで十分なグループ数が期待できる

## 🔧 実装方法

### 手法の変更: `StratifiedGroupKFold` → `GroupKFold`

**理由**：
- StratifiedGroupKFoldは選手分布（label_id）の層化を試みるが、グループ制約で困難
- sessionごとに出現する選手が異なるため、完全な層化は不可能
- **GroupKFoldの方がシンプルで確実にリークを防げる**

### コード実装

#### 1. データ前処理（ノートブック）

```python
# グループ列の作成
df_train['group'] = df_train['quarter'] + '_' + df_train['session'].astype(str)

# グループ数の確認
print(f"Total groups: {df_train['group'].nunique()}")
print(f"Samples per group (mean): {len(df_train) / df_train['group'].nunique():.1f}")
```

#### 2. Runner修正（src/runner.py）

```python
from sklearn.model_selection import GroupKFold  # StratifiedGroupKFold → GroupKFold

class Runner:
    def __init__(self, ...):
        # ...
        self.validator = GroupKFold(n_splits=self.n_splits)
        # shuffle, random_stateは不要（GroupKFoldはshuffleをサポートしない）
```

#### 3. CV設定（ノートブック）

```python
cv_setting = {
    "group_col": "group",  # quarter_sessionを格納した列
    "n_splits": 5,         # 5-fold CV
}

runner = Runner(
    run_name=run_name,
    model_cls=ModelResNet50,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting=cv_setting,
    logger=logger
)
```

## 📈 より高度な戦略（オプション）

### 1. 時系列考慮型CV（Leave-Future-Out）

**動機**: テストデータが新しいクオーターから来る可能性

```python
# クオーター順を保つ
quarters = sorted(df_train['quarter'].unique())
n_folds = 5

# 時系列splitの実装
for fold_idx in range(n_folds):
    # 古い4/5を訓練、新しい1/5を検証
    split_point = int(len(quarters) * (fold_idx + 1) / n_folds)
    train_quarters = quarters[:split_point-1]
    valid_quarters = [quarters[split_point-1]]
    
    train_idx = df_train[df_train['quarter'].isin(train_quarters)].index
    valid_idx = df_train[df_train['quarter'].isin(valid_quarters)].index
```

### 2. angle別CV（side/top両方ある場合）

```python
# angleごとに別々にCV
cv_side = GroupKFold(n_splits=5)
cv_top = GroupKFold(n_splits=5)

# または、angle + quarter_sessionでグループ化
df_train['group'] = df_train['quarter'] + '_' + df_train['angle'] + '_' + df_train['session'].astype(str)
```

## ✅ 検証項目

CVを実装後、以下を確認：

1. **リークチェック**
   ```python
   # 各foldで訓練と検証のグループが重複していないことを確認
   for train_idx, valid_idx in cv.split(X, y, groups):
       train_groups = set(groups[train_idx])
       valid_groups = set(groups[valid_idx])
       assert len(train_groups & valid_groups) == 0, "Leak detected!"
   ```

2. **分布の均等性**
   ```python
   # 各foldのサンプル数を確認
   for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
       print(f"Fold {fold_idx}: Train={len(train_idx)}, Valid={len(valid_idx)}")
   ```

3. **選手分布**
   ```python
   # 各foldで出現する選手を確認
   for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
       train_labels = set(y[train_idx])
       valid_labels = set(y[valid_idx])
       print(f"Fold {fold_idx}: Train players={len(train_labels)}, Valid players={len(valid_labels)}")
       print(f"  Overlap: {len(train_labels & valid_labels)} players")
   ```

## 🚨 注意点

### データの特性
- テストデータの`session`は0以外の値（1, 2, ...）
- → 訓練データのsession=0とテストデータのsession>0は時間的に分離されている
- → CVでも時間的分離を模倣すべき？

### 検討が必要な点
1. テストデータには訓練データにいない選手（unknown）がいる可能性
   - → CV時にも意図的にunknown相当のfoldを作る？（難しい）
   
2. angle（top/side）の分布
   - 訓練: topとside両方
   - テスト: 主にside
   - → CV時にangleの分布も考慮すべきか？

## 📝 実装の優先順位

1. **まずは基本**: `GroupKFold` with `quarter_session` グループ化
2. **確認**: リークチェック、分布の確認
3. **改善**: 必要に応じて時系列考慮型やangle考慮を追加

## 結論

**最善のCV戦略**: `GroupKFold` + `quarter_session`グループ化

- シンプルで確実
- リークを完全に防止
- 実装が容易
- 他の手法と比較可能
