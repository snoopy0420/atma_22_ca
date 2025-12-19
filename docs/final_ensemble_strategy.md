# 最終アンサンブル戦略

## 重要な設計判断

### ❌ Unknown CVのモデルは最終予測に含めない

**理由:**
- **Fold 0 (Q2訓練)**: 選手0を学習していない → 選手0を正しく予測できない
- **Fold 1 (Q1訓練)**: 選手5を学習していない → 選手5を正しく予測できない
- **テストデータ**: 選手0と5も普通に出現する（どのタイミングで入れ替えかは不明）

### ✅ Unknown CVの役割

Unknown CVは**評価専用**:
1. Unknown判定能力の測定
2. 最適な閾値（threshold）の決定
3. Unknown判定の性能指標を把握

## 最終予測戦略

### 使用するモデル
- **通常3-fold CVのモデルのみ** を使用
- 全選手（0-10）を学習済みなので安全

### Unknown CVの知見の活用方法

```python
# Step 1: Unknown CVで最適閾値を見つける
runner_unknown = Runner(..., cv_setting={'validator': Q1Q2Validator()})
runner_unknown.train_cv()

# Unknown評価
metrics = Metric.unknown_metrics(...)
optimal_threshold = find_best_threshold(metrics)  # 例: 0.6

# Step 2: 最適閾値で通常CVを学習
params['threshold'] = optimal_threshold
runner_normal = Runner(..., params=params)
runner_normal.train_cv()

# Step 3: 通常CVのモデルのみで予測
submission = runner_normal.predict_cv()
```

## 詳細な実装例

### 完全なワークフロー

```python
# Notebook での使用例

# ========== Step 1: 通常3-fold CVで学習 ==========
runner_normal = Runner(
    run_name='exp_normal',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={'n_splits': 3, 'method': 'stratified_group', 'group_col': 'quarter'},
    logger=logger
)
runner_normal.train_cv()
scores_normal, oof_normal = runner_normal.metric_cv()


# ========== Step 2: Unknown評価2-fold CVで学習 ==========
runner_unknown = Runner(
    run_name='exp_unknown',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={
        'validator': Q1Q2Validator(quarter_col='quarter'),
        'n_splits': 2
    },
    logger=logger
)
runner_unknown.train_cv()
## 詳細な実装例

### 完全なワークフロー

```python
from src.runner import Runner
from src.model_arcface import ModelArcFace
from src.util import Logger, Validation, Q1Q2Validator, Metric

logger = Logger('logs/')
df_train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')
df_test = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/test_meta.csv')
df_train['group'] = df_train['quarter']

params = {
    'model_name': 'efficientnet_b0',
    'embedding_dim': 512,
    'threshold': 0.5,
    # ... その他
}

# ========== Step 1: Unknown CVで閾値チューニング ==========
validator_unknown = Q1Q2Validator(quarter_col='quarter')
runner_unknown = Runner(
    run_name='exp_unknown_threshold',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={'validator': validator_unknown},
    logger=logger
)
runner_unknown.train_cv()

# Unknown評価
oof_df = pd.read_pickle(f'models/{runner_unknown.run_name}/va_pred.pkl')
oof_df = oof_df.merge(df_train[['quarter']], left_index=True, right_index=True)

fold0_mask = oof_df['quarter'].astype(str).str.startswith('Q1')
metrics_fold0 = Metric.unknown_metrics(
    oof_df[fold0_mask]['label_id'].values,
    oof_df[fold0_mask]['pred'].values,
    unknown_player_id=0
)

fold1_mask = oof_df['quarter'].astype(str).str.startswith('Q2')
metrics_fold1 = Metric.unknown_metrics(
    oof_df[fold1_mask]['label_id'].values,
    oof_df[fold1_mask]['pred'].values,
    unknown_player_id=5
)

avg_unknown_f1 = (metrics_fold0['unknown_f1'] + metrics_fold1['unknown_f1']) / 2
logger.info(f"Unknown F1 (threshold=0.5): {avg_unknown_f1:.4f}")

# 最適閾値を決定（例: 0.6）
optimal_threshold = 0.6

# ========== Step 2: 最適閾値で通常CVを学習 ==========
params['threshold'] = optimal_threshold

validator_normal = Validation.create_validator(
    method='stratified_group',
    n_splits=3,
    shuffle=True,
    random_state=42
)

runner_normal = Runner(
    run_name='exp_normal_optimized',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={'validator': validator_normal, 'group_col': 'quarter'},
    logger=logger
)

runner_normal.train_cv()
scores, oof_score = runner_normal.metric_cv()

# ========== Step 3: 通常CVのモデルのみで予測 ==========
submission = runner_normal.predict_cv()
submission_path = runner_normal.save_submission(submission)

logger.info(f"最終モデル数: 3 (通常CVのみ)")
```

## まとめ

**最終戦略:**
1. Unknown CVは**評価と閾値チューニング専用**
2. 最終予測は**通常3-fold CVのモデルのみ**
3. Unknown CVで得た知見を通常CVに適用

**メリット:**
- 全選手（0-10）を正しく予測可能
- Unknown判定能力も最適化
- 安全で確実な予測
