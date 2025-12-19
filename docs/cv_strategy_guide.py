"""
CV戦略の完全ガイド - 使用例付き

このファイルは、コンペの特性を考慮した2段階CV戦略の使い方をまとめたものです。
"""

# ============================================================================
# 第1段階: 通常学習用CV（StratifiedGroupKFold 5-fold）
# ============================================================================

"""
【目的】
- モデルの基本性能評価
- ハイパーパラメータ調整
- 特徴量エンジニアリング
- モデル選択

【設定】
"""

# Notebookでの使用例
cv_setting = {
    "method": "stratified_group",  # GroupKFold → StratifiedGroupKFold
    "group_col": "quarter",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42
}

"""
【変更点】
- "method": "group" → "stratified_group"
  理由: 各foldで選手分布を均等にするため

【期待効果】
- 選手0（少数クラス）のばらつき軽減
- Macro F1評価の安定性向上
- CVスコアとLBスコアの相関向上
"""

# ============================================================================
# 第2段階: Unknown判定評価用CV（Q1/Q2分離 2-fold）
# ============================================================================

"""
【目的】
- 「見たことのない選手」に対するunknown判定能力の評価
- cos類似度閾値の最適化
- テストデータの状況（選手入れ替え）への対応

【コンペの重要な特性】
1. テストデータで選手の入れ替えが発生（タイミング不明）
2. 訓練にいない選手 → -1を予測する必要
3. 訓練データの選手0/5の入れ替えは、この状況を模している

【使用例: Notebookでの実装】
"""

# notebooks/exp_unknown_cv.ipynb

# ---- セル1: セットアップ ----
import sys
sys.path.append('/workspace/atma_22_ca/')

import pandas as pd
import numpy as np
from src.unknown_cv import UnknownEvaluationCV
from src.model_arcface import ModelArcFace
from src.util import Logger, Metric

logger = Logger()

# データ読み込み
df_train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')
logger.info(f"訓練データ: {len(df_train)}サンプル")

# ---- セル2: Unknown評価CVの初期化 ----
unknown_cv = UnknownEvaluationCV(df_train, logger)
folds = unknown_cv.get_folds()

logger.info(f"\n生成されたfold数: {len(folds)}")
for i, (tr, va, unknown_id) in enumerate(folds):
    logger.info(f"Fold {i}: 訓練{len(tr)}, 検証{len(va)}, Unknown選手{unknown_id}")

# ---- セル3: 各foldでモデル学習・評価 ----
results = []

for fold_idx, (train_df, val_df, unknown_player_id) in enumerate(folds):
    logger.info(f"\n{'='*80}")
    logger.info(f"Fold {fold_idx}: Unknown判定対象 = 選手{unknown_player_id}")
    logger.info(f"{'='*80}")
    
    # パラメータ設定
    params = {
        'model_name': 'efficientnet_b0',
        'embedding_dim': 512,
        'img_size': 224,
        'batch_size': 64,
        'epochs': 20,
        'lr': 1e-3,
        'arcface_s': 30.0,
        'arcface_m': 0.5,
        'threshold': 0.5,  # 初期閾値
        'use_ema': True,
        'ema_decay': 0.995,
        'num_workers': 8,
    }
    
    # モデル初期化
    run_name = f'unknown_cv_fold{fold_idx}'
    model = ModelArcFace(run_name, params, DIR_MODEL, logger)
    
    # 学習
    model.train(train_df, val_df)
    
    # 予測
    val_pred = model.predict(val_df, split='valid')
    
    # 評価
    eval_result = unknown_cv.evaluate_unknown_detection(
        y_true=val_df['label_id'].values,
        y_pred=val_pred['label_id'].values,
        unknown_player_id=unknown_player_id
    )
    
    results.append(eval_result)
    
    logger.info(f"\nFold {fold_idx} 結果:")
    logger.info(f"  Unknown Precision: {eval_result['unknown_precision']:.4f}")
    logger.info(f"  Unknown Recall:    {eval_result['unknown_recall']:.4f}")
    logger.info(f"  Unknown F1:        {eval_result['unknown_f1']:.4f}")
    logger.info(f"  既知選手精度:      {eval_result['known_accuracy']:.4f}")

# ---- セル4: サマリー表示 ----
unknown_cv.print_evaluation_summary(results)

# ---- セル5: 閾値チューニング（オプション） ----
"""
閾値チューニングを行う場合は、src/threshold_optimizer.py を使用
各foldで最適閾値を探索し、Unknown F1を最大化
"""

from src.threshold_optimizer import ThresholdOptimizer

threshold_optimizer = ThresholdOptimizer(logger)

# Fold 0で閾値最適化の例
fold_idx = 0
train_df, val_df, unknown_player_id = folds[fold_idx]

# モデルから埋め込みとプロトタイプを取得（省略: 実際にはモデルから抽出）
# embeddings = ...  # [N, 512]
# prototypes = ...  # [num_classes, 512]

# 最適閾値探索
# best_threshold, best_score = threshold_optimizer.optimize_threshold(
#     embeddings=embeddings,
#     prototypes=prototypes,
#     true_labels=val_df['label_id'].values,
#     threshold_range=(0.3, 0.8),
#     n_steps=100
# )

# logger.info(f"最適閾値: {best_threshold:.4f}, Unknown F1: {best_score:.4f}")

# ============================================================================
# まとめ: 推奨ワークフロー
# ============================================================================

"""
【ステップ1】通常の5-fold CVで基本性能確認
→ notebooks/exp_arcface_cloud.ipynb
→ cv_setting = {"method": "stratified_group", ...}

【ステップ2】ハイパーパラメータ調整
→ 同じく5-fold CVで複数設定を試す
→ OOFスコアで最良設定を選択

【ステップ3】Unknown判定評価（2-fold CV）
→ notebooks/exp_unknown_cv.ipynb（新規作成）
→ src/unknown_cv.py を使用
→ Unknown F1を確認

【ステップ4】閾値チューニング
→ Unknown F1を最大化する閾値を探索
→ src/threshold_optimizer.py を使用

【ステップ5】最終提出
→ 全データで再学習
→ 最適閾値で予測
→ 提出ファイル生成
"""

# ============================================================================
# FAQ
# ============================================================================

"""
Q1: なぜ2段階のCVが必要なのか？

A1: コンペの特性上、以下の2つの異なる能力が必要だから:
    1. 既知の選手を正確に識別する能力（11クラス分類）
    2. 未知の選手をunknownと判定する能力（異常検知）
    
    これらは別々に評価・最適化する必要がある。

Q2: Unknown判定の性能が悪い場合は？

A2: 以下を試す:
    - 閾値を上げる（より保守的にunknown判定）
    - ArcFaceのマージンmを大きくする（クラス間距離を広げる）
    - 埋め込み次元を増やす（識別能力向上）
    - TTAを使用（予測の安定性向上）

Q3: 5-fold CVと2-fold CVでどちらを信頼すべき？

A3: 用途が異なる:
    - 5-fold CV: モデルの基本性能・ハイパラ調整に使用
    - 2-fold CV: Unknown判定能力・閾値調整に使用
    
    最終的なLBスコアは両方のバランスで決まる。

Q4: 実装の優先度は？

A4: 
    1. まず5-fold CVで基本性能を確認（今すぐ）
    2. Unknown CVは閾値チューニング時に実装（後で可）
    3. LBフィードバックを見て調整
"""

if __name__ == "__main__":
    print("CV戦略ガイドを確認してください")
    print("詳細は本ファイルのコメントを参照")
