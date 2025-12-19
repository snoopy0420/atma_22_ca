"""
統合CV戦略の使用例
通常3-fold CV + Unknown評価2-fold CV = 合計5モデルでアンサンブル予測
"""

import sys
sys.path.append('/workspace/atma_22_ca/')

import pandas as pd
import numpy as np
from src.runner import Runner
from src.runner_ensemble import RunnerEnsemble
from src.model_arcface import ModelArcFace
from src.util import Logger, Q1Q2Validator, Metric

# ============================================================================
# セットアップ
# ============================================================================

logger = Logger('logs/')

# データ読み込み
df_train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')
df_test = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/test_meta.csv')

# 前処理
df_train['group'] = df_train['quarter']

logger.info(f"訓練データ: {len(df_train):,}サンプル, {df_train['label_id'].nunique()}選手")
logger.info(f"テストデータ: {len(df_test):,}サンプル")

# ============================================================================
# パラメータ設定
# ============================================================================

params = {
    'model_name': 'efficientnet_b0',
    'embedding_dim': 512,
    'img_size': 224,
    'batch_size': 64,
    'epochs': 20,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'arcface_s': 30.0,
    'arcface_m': 0.5,
    'threshold': 0.5,
    'use_ema': True,
    'ema_decay': 0.995,
    'num_workers': 8,
}

# ============================================================================
# Step 1: 通常の3-fold CVで学習（汎化性能重視）
# ============================================================================

logger.info("\n" + "="*80)
logger.info("Step 1: 通常3-fold CV")
logger.info("="*80)

# Validatorを作成
from src.util import Validation

validator_normal = Validation.create_validator(
    method='stratified_group',
    n_splits=3,
    shuffle=True,
    random_state=42
)

runner_normal = Runner(
    run_name='exp_normal_3fold',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={
        'validator': validator_normal,
        'group_col': 'quarter'
    },
    logger=logger
)

# 学習
runner_normal.train_cv()

# OOF評価
scores_normal, oof_normal = runner_normal.metric_cv()
logger.info(f"\n通常CV OOFスコア: {oof_normal:.5f}")
logger.info(f"Foldスコア: {[f'{s:.5f}' for s in scores_normal]}")

# ============================================================================
# Step 2: Unknown評価用の2-fold CV（Q1/Q2分離）
# ============================================================================

logger.info("\n" + "="*80)
logger.info("Step 2: Unknown評価2-fold CV (Q1/Q2分離)")
logger.info("="*80)

# Q1Q2Validatorを使用
q1q2_validator = Q1Q2Validator(quarter_col='quarter')

runner_unknown = Runner(
    run_name='exp_unknown_2fold',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={
        'validator': q1q2_validator
    },
    logger=logger
)

# 学習
runner_unknown.train_cv()

# Unknown評価指標を計算
logger.info("\n" + "="*80)
logger.info("Unknown判定性能の評価")
logger.info("="*80)

# OOF予測を取得
oof_pred_path = f'models/{runner_unknown.run_name}/va_pred.pkl'
oof_df = pd.read_pickle(oof_pred_path)

# 元のデータとマージしてquarter情報を取得
oof_df = oof_df.merge(
    df_train[['quarter']],
    left_index=True,
    right_index=True,
    how='left'
)

# Fold 0: Q1が検証（選手0がunknown）
fold0_mask = oof_df['quarter'].astype(str).str.startswith('Q1')
if fold0_mask.sum() > 0:
    metrics_fold0 = Metric.unknown_metrics(
        oof_df[fold0_mask]['label_id'].values,
        oof_df[fold0_mask]['pred'].values,
        unknown_player_id=0
    )
    logger.info(f"\nFold 0 (Q1検証, 選手0がunknown):")
    logger.info(f"  Unknown Precision: {metrics_fold0['unknown_precision']:.4f}")
    logger.info(f"  Unknown Recall:    {metrics_fold0['unknown_recall']:.4f}")
    logger.info(f"  Unknown F1:        {metrics_fold0['unknown_f1']:.4f}")
    logger.info(f"  既知選手 Macro F1: {metrics_fold0['known_macro_f1']:.4f}")

# Fold 1: Q2が検証（選手5がunknown）
fold1_mask = oof_df['quarter'].astype(str).str.startswith('Q2')
if fold1_mask.sum() > 0:
    metrics_fold1 = Metric.unknown_metrics(
        oof_df[fold1_mask]['label_id'].values,
        oof_df[fold1_mask]['pred'].values,
        unknown_player_id=5
    )
    logger.info(f"\nFold 1 (Q2検証, 選手5がunknown):")
    logger.info(f"  Unknown Precision: {metrics_fold1['unknown_precision']:.4f}")
    logger.info(f"  Unknown Recall:    {metrics_fold1['unknown_recall']:.4f}")
    logger.info(f"  Unknown F1:        {metrics_fold1['unknown_f1']:.4f}")
    logger.info(f"  既知選手 Macro F1: {metrics_fold1['known_macro_f1']:.4f}")

# 平均
if fold0_mask.sum() > 0 and fold1_mask.sum() > 0:
    avg_unknown_f1 = (metrics_fold0['unknown_f1'] + metrics_fold1['unknown_f1']) / 2
    avg_known_f1 = (metrics_fold0['known_macro_f1'] + metrics_fold1['known_macro_f1']) / 2
    logger.info(f"\n【平均】")
    logger.info(f"  Unknown F1:        {avg_unknown_f1:.4f}")
    logger.info(f"  既知選手 Macro F1: {avg_known_f1:.4f}")

# ============================================================================
# Step 3: 通常3-fold CVのモデルで予測
# ============================================================================

logger.info("\n" + "="*80)
logger.info("Step 3: 通常3-fold CVのモデルで予測")
logger.info("="*80)
logger.info("⚠️ 注意: Unknown CVのモデルは選手0/5を学習していないため、予測には使用しません")

# 通常CVのモデルのみで予測
submission = runner_normal.predict_cv()

# 提出ファイル保存
submission_path = runner_normal.save_submission(submission)

logger.info("\n" + "="*80)
logger.info("全工程完了！")
logger.info("="*80)
logger.info(f"提出ファイル: {submission_path}")
logger.info(f"通常CV OOF: {oof_normal:.5f}")
if fold0_mask.sum() > 0 and fold1_mask.sum() > 0:
    logger.info(f"Unknown判定能力 (参考): F1={avg_unknown_f1:.4f}")
logger.info(f"最終予測モデル数: 3 (通常CVのみ)")

# ============================================================================
# まとめ
# ============================================================================

"""
【実行内容】
1. 通常3-fold CV: 全データ（選手0-10）でCV学習 → OOF評価
2. Unknown 2-fold CV: Q1/Q2分離でunknown判定能力を評価
3. 最終予測: 通常3-fold CVのモデルのみで予測

【重要な設計判断】
❌ Unknown CVのモデルは最終予測に含めない理由:
   - Fold 0 (Q2訓練): 選手0を学習していない → 選手0を予測できない
   - Fold 1 (Q1訓練): 選手5を学習していない → 選手5を予測できない
   - テストデータには選手0/5も普通に出現する
   
✅ Unknown CVの役割:
   - Unknown判定能力の評価専用
   - 最適な閾値（threshold）の決定
   - Unknown判定の性能指標を把握

【最終予測戦略】
- 通常3-fold CVのモデルのみ使用（全選手0-10を学習済み）
- Unknown CVで得た最適閾値を通常CVに適用
- 安全で確実な予測を実現
"""
