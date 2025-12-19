"""
統合されたRunnerクラスの使用例

RunnerクラスにCV戦略が2つ組み込まれました：
1. train_cv() - 通常の5-fold CV（StratifiedGroupKFold）
2. train_unknown_cv() - Unknown判定評価用2-fold CV（Q1/Q2分離）
"""

# ============================================================================
# Notebookでの使用例
# ============================================================================

# ---- セル1: セットアップ ----
import sys
sys.path.append('/workspace/atma_22_ca/')

import pandas as pd
import numpy as np
from src.runner import Runner
from src.model_arcface import ModelArcFace
from src.util import Logger

logger = Logger()

# データ読み込み
df_train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')
df_test = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/test_meta.csv')

# 前処理: groupカラム作成
df_train['group'] = df_train['quarter']

logger.info(f"訓練データ: {len(df_train):,}サンプル, {df_train['label_id'].nunique()}選手")
logger.info(f"テストデータ: {len(df_test):,}サンプル")

# ---- セル2: パラメータ設定 ----
run_name = 'arcface_efficientnet_b0_' + pd.Timestamp.now().strftime('%Y%m%d%H%M')

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

# Validator作成（StratifiedGroupKFold）
from src.util import Validation

validator = Validation.create_validator(
    method='stratified_group',
    n_splits=3,
    shuffle=True,
    random_state=42
)

cv_setting = {
    'validator': validator,
    'group_col': 'quarter'
}

# ---- セル3: Runner初期化 ----
runner = Runner(
    run_name=run_name,
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting=cv_setting,
    logger=logger
)

logger.info("Runner初期化完了")

# ============================================================================
# パターンA: 通常の5-fold CVのみ実行（モデル選択・ハイパラ調整用）
# ============================================================================

# ---- セル4A: 通常CVで学習 ----
runner.train_cv()

# ---- セル5A: 通常CVで評価 ----
scores, oof_score = runner.metric_cv()

logger.info(f"\n最終OOFスコア: {oof_score:.5f}")
logger.info(f"各Foldスコア: {[f'{s:.5f}' for s in scores]}")

# ---- セル6A: テストデータで予測 ----
runner.predict_cv()

# ---- セル7A: 提出ファイル生成 ----
runner.save_submission()

# ============================================================================
# パターンB: 通常CV + Unknown評価CV（完全版）
# ============================================================================

# ---- セル4B: 通常CVで学習 ----
runner.train_cv()

# ---- セル5B: 通常CVで評価 ----
scores, oof_score = runner.metric_cv()
logger.info(f"通常CV OOFスコア: {oof_score:.5f}")

# ---- セル6B: Unknown判定評価CVを実行 ----
# これは別途実行（モデルを再学習）
unknown_results = runner.train_unknown_cv()

# 結果確認
logger.info("\n【Unknown CV結果サマリー】")
for result in unknown_results:
    logger.info(f"\n{result['fold_name']}:")
    logger.info(f"  Unknown F1: {result['unknown_f1']:.4f}")
    logger.info(f"  既知選手 Macro F1: {result['known_macro_f1']:.4f}")

# ---- セル7B: 閾値調整が必要な場合 ----
# Unknown CVの結果から、閾値を調整
if unknown_results:
    avg_unknown_f1 = np.mean([r['unknown_f1'] for r in unknown_results])
    avg_unknown_recall = np.mean([r['unknown_recall'] for r in unknown_results])
    
    logger.info(f"\nUnknown F1: {avg_unknown_f1:.4f}")
    logger.info(f"Unknown Recall: {avg_unknown_recall:.4f}")
    
    # Recallが低い → 閾値を上げる
    if avg_unknown_recall < 0.5:
        logger.info("⚠️ Unknown Recallが低い → 閾値を上げることを推奨")
        logger.info("例: params['threshold'] = 0.6 または 0.7")
    
    # Precisionが低い → 閾値を下げる
    avg_unknown_precision = np.mean([r['unknown_precision'] for r in unknown_results])
    if avg_unknown_precision < 0.5:
        logger.info("⚠️ Unknown Precisionが低い → 閾値を下げることを推奨")
        logger.info("例: params['threshold'] = 0.4 または 0.3")

# ---- セル8B: 最適閾値で再学習（オプション） ----
# 閾値を調整して再実行
# params['threshold'] = 0.6  # 例
# runner_v2 = Runner(run_name + '_v2', ModelArcFace, params, df_train, df_test, cv_setting, logger)
# runner_v2.train_cv()
# runner_v2.predict_cv()
# runner_v2.save_submission()

# ============================================================================
# パターンC: Unknown CVのみ実行（閾値チューニング専用）
# ============================================================================

# ---- セル4C: Unknown CVのみ ----
# 高速に閾値の影響を確認したい場合
unknown_results = runner.train_unknown_cv()

# 複数の閾値で試す
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    logger.info(f"\n{'='*80}")
    logger.info(f"閾値 = {threshold}")
    logger.info(f"{'='*80}")
    
    params_test = params.copy()
    params_test['threshold'] = threshold
    
    runner_test = Runner(
        f'{run_name}_th{threshold}',
        ModelArcFace,
        params_test,
        df_train,
        df_test,
        cv_setting,
        logger
    )
    
    results = runner_test.train_unknown_cv()
    
    # サマリー確認
    avg_f1 = np.mean([r['unknown_f1'] for r in results])
    logger.info(f"平均 Unknown F1: {avg_f1:.4f}")

# ============================================================================
# まとめ
# ============================================================================

"""
【推奨ワークフロー】

1. 初期開発: パターンA（通常CVのみ）
   - モデル構造の検証
   - ハイパーパラメータ調整
   - 特徴量エンジニアリング

2. 中間評価: パターンB（両方実行）
   - 通常CVでOOFスコア確認
   - Unknown CVでunknown判定能力確認
   - 閾値の初期値を決定

3. 最終調整: パターンC（Unknown CVのみ）
   - 複数の閾値で高速実験
   - 最適閾値を探索
   - 最終提出用モデルで再学習

【実行時間の目安】
- 通常CV (5-fold): 約1-2時間（GPU環境、epoch=20）
- Unknown CV (2-fold): 約30-40分（データ量が異なるため）
- 両方実行: 約1.5-2.5時間

【メリット】
- 1つのRunnerクラスで両方のCV戦略を管理
- Notebookからシンプルに切り替え可能
- 結果が自動的に保存・ログ出力される
"""

if __name__ == "__main__":
    print("統合Runner使用例を確認してください")
