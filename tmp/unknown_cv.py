"""
Unknown判定評価用のCV実装

コンペの特性:
- テストデータで選手の入れ替えが発生（タイミング不明）
- 訓練データに存在しない選手が出現 → -1を予測する必要
- 訓練データの選手0/5の入れ替えは、この状況を模している

このモジュールは、「見たことのない選手」に対するunknown判定能力を評価する
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple

sys.path.append(os.path.abspath('..'))
from configs.config import *


class UnknownEvaluationCV:
    """Unknown判定評価用のCV戦略
    
    Q1/Q2を完全に分離して、一方の選手を見ずに他方を予測する能力をテスト
    """
    
    def __init__(self, df_train: pd.DataFrame, logger=None):
        """
        Args:
            df_train: 訓練データ（quarterカラム必須）
            logger: ロガー
        """
        self.df_train = df_train
        self.logger = logger
        
        # Q1/Q2のquarterリスト
        all_quarters = df_train['quarter'].unique()
        self.q1_quarters = [q for q in all_quarters if q.startswith('Q1')]
        self.q2_quarters = [q for q in all_quarters if q.startswith('Q2')]
        
        self.log(f"Unknown評価CV初期化: Q1={len(self.q1_quarters)}個, Q2={len(self.q2_quarters)}個")
    
    def log(self, msg):
        """ログ出力"""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def get_folds(self) -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
        """2つのfoldを生成
        
        Returns:
            [(train_df, val_df, unknown_player_id), ...]のリスト
        """
        folds = []
        
        # Fold 0: Q2で訓練 → Q1で検証（選手0がunknown）
        train_fold0 = self.df_train[self.df_train['quarter'].isin(self.q2_quarters)].copy()
        val_fold0 = self.df_train[self.df_train['quarter'].isin(self.q1_quarters)].copy()
        folds.append((train_fold0, val_fold0, 0))
        
        # Fold 1: Q1で訓練 → Q2で検証（選手5がunknown）
        train_fold1 = self.df_train[self.df_train['quarter'].isin(self.q1_quarters)].copy()
        val_fold1 = self.df_train[self.df_train['quarter'].isin(self.q2_quarters)].copy()
        folds.append((train_fold1, val_fold1, 5))
        
        return folds
    
    def evaluate_unknown_detection(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   unknown_player_id: int) -> dict:
        """Unknown判定の性能を評価
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            unknown_player_id: unknown判定すべき選手ID
        
        Returns:
            評価メトリクスの辞書
        """
        # unknown選手のマスク
        is_unknown = (y_true == unknown_player_id)
        
        # 予測がunknown (-1) かどうか
        pred_unknown = (y_pred == -1)
        
        # Unknown判定の評価
        tp_unknown = np.sum(is_unknown & pred_unknown)  # 正しくunknownと判定
        fp_unknown = np.sum(~is_unknown & pred_unknown)  # 誤ってunknownと判定
        fn_unknown = np.sum(is_unknown & ~pred_unknown)  # unknownを見逃し
        
        precision_unknown = tp_unknown / (tp_unknown + fp_unknown) if (tp_unknown + fp_unknown) > 0 else 0
        recall_unknown = tp_unknown / (tp_unknown + fn_unknown) if (tp_unknown + fn_unknown) > 0 else 0
        f1_unknown = 2 * precision_unknown * recall_unknown / (precision_unknown + recall_unknown) if (precision_unknown + recall_unknown) > 0 else 0
        
        # 既知選手の精度（unknownを除外）
        known_mask = ~is_unknown
        known_correct = np.sum((y_true[known_mask] == y_pred[known_mask]))
        known_total = known_mask.sum()
        known_accuracy = known_correct / known_total if known_total > 0 else 0
        
        return {
            'unknown_player_id': unknown_player_id,
            'unknown_precision': precision_unknown,
            'unknown_recall': recall_unknown,
            'unknown_f1': f1_unknown,
            'unknown_samples': is_unknown.sum(),
            'unknown_detected': pred_unknown.sum(),
            'tp_unknown': tp_unknown,
            'fp_unknown': fp_unknown,
            'fn_unknown': fn_unknown,
            'known_accuracy': known_accuracy,
            'known_samples': known_total,
        }
    
    def print_evaluation_summary(self, results: List[dict]):
        """評価結果のサマリーを表示"""
        self.log("\n" + "="*80)
        self.log("Unknown判定評価サマリー")
        self.log("="*80)
        
        for i, result in enumerate(results):
            self.log(f"\nFold {i} (Unknown選手: {result['unknown_player_id']})")
            self.log(f"  Unknown判定:")
            self.log(f"    Precision: {result['unknown_precision']:.4f}")
            self.log(f"    Recall:    {result['unknown_recall']:.4f}")
            self.log(f"    F1:        {result['unknown_f1']:.4f}")
            self.log(f"    TP/FP/FN:  {result['tp_unknown']}/{result['fp_unknown']}/{result['fn_unknown']}")
            self.log(f"  既知選手の精度: {result['known_accuracy']:.4f}")
        
        # 平均
        avg_precision = np.mean([r['unknown_precision'] for r in results])
        avg_recall = np.mean([r['unknown_recall'] for r in results])
        avg_f1 = np.mean([r['unknown_f1'] for r in results])
        avg_known_acc = np.mean([r['known_accuracy'] for r in results])
        
        self.log("\n【平均】")
        self.log(f"  Unknown Precision: {avg_precision:.4f}")
        self.log(f"  Unknown Recall:    {avg_recall:.4f}")
        self.log(f"  Unknown F1:        {avg_f1:.4f}")
        self.log(f"  既知選手精度:      {avg_known_acc:.4f}")
        self.log("="*80)


if __name__ == "__main__":
    # 使用例
    import pandas as pd
    from src.util import Logger
    
    logger = Logger()
    
    # 訓練データ読み込み
    df_train = pd.read_csv(f'{DIR_INPUT}/atmaCup22_2nd_meta/train_meta.csv')
    
    # Unknown評価CV
    unknown_cv = UnknownEvaluationCV(df_train, logger)
    
    # Fold取得
    folds = unknown_cv.get_folds()
    
    logger.info(f"\n生成されたfold数: {len(folds)}")
    for i, (tr, va, unknown_id) in enumerate(folds):
        logger.info(f"\nFold {i}:")
        logger.info(f"  訓練: {len(tr)}サンプル")
        logger.info(f"  検証: {len(va)}サンプル")
        logger.info(f"  Unknown判定対象: 選手{unknown_id}")
        logger.info(f"  訓練の選手: {sorted(tr['label_id'].unique())}")
        logger.info(f"  検証の選手: {sorted(va['label_id'].unique())}")
