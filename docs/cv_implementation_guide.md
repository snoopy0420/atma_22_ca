# CV戦略実装ガイド

## ✅ 実装済み内容

### 1. Runner.py の修正完了
- `GroupKFold`をデフォルトに設定（リーク防止最優先）
- `StratifiedGroupKFold`も選択可能（後方互換性）
- CV手法は`cv_setting['method']`で切り替え可能

### 2. exp_resnet.ipynb の更新完了
- データ読み込み時に`group`列を自動生成
- CV設定で`method='group'`を推奨設定として明記
- グループ数とサンプル分布を自動表示

## 🚀 使い方

### ノートブックで実行

```python
# データ読み込み後、自動的にgroup列が作成される
df_train['group'] = df_train['quarter'] + '_' + df_train['session'].astype(str)

# CV設定（GroupKFold推奨）
cv_setting = {
    'method': 'group',      # リーク防止最優先
    'group_col': 'group',   # quarter_session列
    'n_splits': 5,
}

# Runner作成・実行
runner = Runner(
    run_name=run_name,
    model_cls=ModelResNet50,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting=cv_setting,
    logger=logger
)

# CV学習
scores = runner.train_cv()
```

## 📊 リークチェック方法

ノートブックに以下を追加して確認：

```python
# リークチェック
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
groups = df_train['group'].values
y = df_train['label_id'].values

for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(df_train, y, groups)):
    train_groups = set(groups[train_idx])
    valid_groups = set(groups[valid_idx])
    
    # 重複チェック
    overlap = train_groups & valid_groups
    assert len(overlap) == 0, f"Fold {fold_idx}: Leak detected! {len(overlap)} groups overlap"
    
    print(f"Fold {fold_idx}: ✓ No leak")
    print(f"  Train: {len(train_idx)} samples, {len(train_groups)} groups")
    print(f"  Valid: {len(valid_idx)} samples, {len(valid_groups)} groups")
    
    # 選手分布
    train_labels = set(y[train_idx])
    valid_labels = set(y[valid_idx])
    print(f"  Players - Train: {len(train_labels)}, Valid: {len(valid_labels)}, Overlap: {len(train_labels & valid_labels)}")
    print()
```

## 🔄 StratifiedGroupKFoldへの切り替え

選手分布の層化も試したい場合：

```python
cv_setting = {
    'method': 'stratified_group',  # 層化を試みる
    'group_col': 'group',
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42,
}
```

**注意**: グループ制約により完全な層化は困難な場合あり

## 🎯 期待される効果

### Before（StratifiedGroupKFold with 'quarter'）
- ⚠️ 同一quarter内の異なるsessionが訓練と検証に分かれる可能性
- ⚠️ 連続フレームが異なるfoldに → リークのリスク
- CV: 高いが、LB: 低い（過学習）

### After（GroupKFold with 'quarter_session'）
- ✅ session単位で完全に分離
- ✅ 連続フレームは必ず同じfold
- ✅ CVとLBの相関が向上

## 📝 次のステップ

1. ノートブックでCV実行
2. リークチェックを実施
3. CV scoreとLB scoreの相関を確認
4. 必要に応じて時系列考慮型CVも試す
