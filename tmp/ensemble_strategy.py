"""
アンサンブル戦略モジュール
複数モデルの予測を組み合わせてスコア向上
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from collections import Counter

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.util import Metric


class EnsembleStrategy:
    """アンサンブル戦略クラス"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def log(self, msg):
        """ログ出力"""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    # ========================================================================
    # 1. Voting（投票）ベースのアンサンブル
    # ========================================================================
    
    def hard_voting(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Hard Voting（多数決）
        
        各モデルの予測ラベルで多数決を取る
        
        Args:
            predictions_list: 各モデルの予測ラベル [N,] のリスト
        
        Returns:
            アンサンブル予測 [N,]
        """
        self.log(f"Hard Voting: {len(predictions_list)}モデル")
        
        # 各サンプルごとに多数決
        ensemble_pred = []
        for i in range(len(predictions_list[0])):
            votes = [pred[i] for pred in predictions_list]
            # 最頻値を採用
            most_common = Counter(votes).most_common(1)[0][0]
            ensemble_pred.append(most_common)
        
        return np.array(ensemble_pred)
    
    def soft_voting(self, 
                   similarities_list: List[torch.Tensor],
                   threshold: float = 0.5) -> np.ndarray:
        """Soft Voting（確率平均）
        
        各モデルのcos類似度を平均してから予測
        
        Args:
            similarities_list: 各モデルの類似度行列 [N, num_classes] のリスト
            threshold: unknown判定閾値
        
        Returns:
            アンサンブル予測 [N,]
        """
        self.log(f"Soft Voting: {len(similarities_list)}モデル")
        
        # 類似度を平均
        avg_similarities = torch.stack(similarities_list).mean(dim=0)  # [N, num_classes]
        
        # 最大類似度で予測
        max_sims, max_indices = avg_similarities.max(dim=1)
        
        # 閾値判定
        predictions = []
        for sim, idx in zip(max_sims.tolist(), max_indices.tolist()):
            predictions.append(-1 if sim < threshold else idx)
        
        return np.array(predictions)
    
    # ========================================================================
    # 2. Weighted Ensemble（重み付きアンサンブル）
    # ========================================================================
    
    def weighted_soft_voting(self,
                            similarities_list: List[torch.Tensor],
                            weights: List[float],
                            threshold: float = 0.5) -> np.ndarray:
        """Weighted Soft Voting
        
        各モデルの類似度を重み付き平均
        
        Args:
            similarities_list: 各モデルの類似度行列
            weights: 各モデルの重み（OOFスコアなどから算出）
            threshold: unknown判定閾値
        
        Returns:
            アンサンブル予測
        """
        self.log(f"Weighted Soft Voting: {len(similarities_list)}モデル")
        self.log(f"  重み: {weights}")
        
        # 重みを正規化
        weights = np.array(weights) / np.sum(weights)
        
        # 重み付き平均
        weighted_avg = sum(w * sim for w, sim in zip(weights, similarities_list))
        
        # 予測
        max_sims, max_indices = weighted_avg.max(dim=1)
        predictions = []
        for sim, idx in zip(max_sims.tolist(), max_indices.tolist()):
            predictions.append(-1 if sim < threshold else idx)
        
        return np.array(predictions)
    
    # ========================================================================
    # 3. Rank Averaging（順位平均）
    # ========================================================================
    
    def rank_averaging(self,
                      similarities_list: List[torch.Tensor],
                      threshold: float = 0.5) -> np.ndarray:
        """Rank Averaging
        
        各モデルの類似度を順位に変換してから平均
        スケールの違いに頑健
        
        Args:
            similarities_list: 各モデルの類似度行列
            threshold: unknown判定閾値
        
        Returns:
            アンサンブル予測
        """
        self.log(f"Rank Averaging: {len(similarities_list)}モデル")
        
        # 各モデルで順位に変換（各サンプルごと）
        rank_list = []
        for sims in similarities_list:
            # argsortで順位を取得（降順）
            ranks = sims.argsort(dim=1, descending=True).argsort(dim=1).float()
            rank_list.append(ranks)
        
        # 順位を平均
        avg_ranks = torch.stack(rank_list).mean(dim=0)
        
        # 順位が最小（最も良い）のクラスを選択
        best_classes = avg_ranks.argmin(dim=1)
        
        # 元の類似度の平均でunknown判定
        avg_similarities = torch.stack(similarities_list).mean(dim=0)
        max_sims = avg_similarities.gather(1, best_classes.unsqueeze(1)).squeeze()
        
        # 予測
        predictions = []
        for sim, idx in zip(max_sims.tolist(), best_classes.tolist()):
            predictions.append(-1 if sim < threshold else idx)
        
        return np.array(predictions)
    
    # ========================================================================
    # 4. Stacking（スタッキング）
    # ========================================================================
    
    def simple_stacking(self,
                       train_predictions: List[np.ndarray],
                       test_predictions: List[np.ndarray],
                       train_labels: np.ndarray) -> np.ndarray:
        """Simple Stacking
        
        OOF予測を特徴量として2段目モデルを学習
        （簡易版: 投票ベースの決定木）
        
        Args:
            train_predictions: OOF予測のリスト [N,] x M
            test_predictions: テスト予測のリスト [N,] x M
            train_labels: 訓練ラベル [N,]
        
        Returns:
            アンサンブル予測
        """
        from sklearn.ensemble import RandomForestClassifier
        
        self.log(f"Simple Stacking: {len(train_predictions)}モデル")
        
        # OOF予測を特徴量に変換
        X_train = np.column_stack(train_predictions)  # [N, M]
        X_test = np.column_stack(test_predictions)    # [N_test, M]
        
        # 2段目モデル（Random Forest）
        meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        meta_model.fit(X_train, train_labels)
        
        # 予測
        ensemble_pred = meta_model.predict(X_test)
        
        return ensemble_pred
    
    # ========================================================================
    # 5. Confidence-based Ensemble（信頼度ベース）
    # ========================================================================
    
    def confidence_based_ensemble(self,
                                 similarities_list: List[torch.Tensor],
                                 threshold: float = 0.5) -> np.ndarray:
        """Confidence-based Ensemble
        
        各サンプルごとに最も信頼度が高いモデルの予測を採用
        
        Args:
            similarities_list: 各モデルの類似度行列
            threshold: unknown判定閾値
        
        Returns:
            アンサンブル予測
        """
        self.log(f"Confidence-based Ensemble: {len(similarities_list)}モデル")
        
        # 各モデルの最大類似度を取得
        max_sims_list = [sims.max(dim=1)[0] for sims in similarities_list]  # [N,] x M
        max_indices_list = [sims.max(dim=1)[1] for sims in similarities_list]
        
        # サンプルごとに最も信頼度が高いモデルを選択
        all_max_sims = torch.stack(max_sims_list, dim=1)  # [N, M]
        best_model_indices = all_max_sims.argmax(dim=1)  # [N,]
        
        # 選ばれたモデルの予測を採用
        predictions = []
        for i, model_idx in enumerate(best_model_indices.tolist()):
            sim = max_sims_list[model_idx][i].item()
            idx = max_indices_list[model_idx][i].item()
            predictions.append(-1 if sim < threshold else idx)
        
        return np.array(predictions)
    
    # ========================================================================
    # 6. アンサンブル評価
    # ========================================================================
    
    def evaluate_ensemble_strategies(self,
                                    similarities_list: List[torch.Tensor],
                                    predictions_list: List[np.ndarray],
                                    true_labels: np.ndarray,
                                    threshold: float = 0.5) -> pd.DataFrame:
        """各アンサンブル戦略を評価
        
        Args:
            similarities_list: 各モデルの類似度行列
            predictions_list: 各モデルの予測ラベル
            true_labels: 正解ラベル
            threshold: unknown判定閾値
        
        Returns:
            評価結果のDataFrame
        """
        self.log("\n" + "="*80)
        self.log("📊 アンサンブル戦略の比較")
        self.log("="*80)
        
        results = []
        
        # 1. Hard Voting
        pred = self.hard_voting(predictions_list)
        score = Metric.macro_f1(true_labels, pred)
        results.append({'strategy': 'Hard Voting', 'macro_f1': score})
        self.log(f"Hard Voting: {score:.5f}")
        
        # 2. Soft Voting
        pred = self.soft_voting(similarities_list, threshold)
        score = Metric.macro_f1(true_labels, pred)
        results.append({'strategy': 'Soft Voting', 'macro_f1': score})
        self.log(f"Soft Voting: {score:.5f}")
        
        # 3. Rank Averaging
        pred = self.rank_averaging(similarities_list, threshold)
        score = Metric.macro_f1(true_labels, pred)
        results.append({'strategy': 'Rank Averaging', 'macro_f1': score})
        self.log(f"Rank Averaging: {score:.5f}")
        
        # 4. Confidence-based
        pred = self.confidence_based_ensemble(similarities_list, threshold)
        score = Metric.macro_f1(true_labels, pred)
        results.append({'strategy': 'Confidence-based', 'macro_f1': score})
        self.log(f"Confidence-based: {score:.5f}")
        
        # 個別モデルの性能も表示
        self.log("\n【個別モデル性能】")
        for i, pred in enumerate(predictions_list):
            score = Metric.macro_f1(true_labels, pred)
            results.append({'strategy': f'Model {i}', 'macro_f1': score})
            self.log(f"Model {i}: {score:.5f}")
        
        df_results = pd.DataFrame(results).sort_values('macro_f1', ascending=False)
        
        self.log("\n" + "="*80)
        self.log("最良戦略: " + df_results.iloc[0]['strategy'])
        self.log(f"スコア: {df_results.iloc[0]['macro_f1']:.5f}")
        self.log("="*80)
        
        return df_results


# ============================================================================
# 実装ガイド
# ============================================================================

"""
【アンサンブル戦略の選び方】

1. Hard Voting
   - 最もシンプル
   - モデル数が少ない（2-3個）時に効果的
   - 予測ラベルのみで実装可能

2. Soft Voting
   - 最も一般的で効果的
   - モデル数が多い（3-5個）時に有効
   - 類似度スコアが必要

3. Weighted Soft Voting
   - 各モデルの性能差が大きい時に有効
   - OOFスコアで重みを設定
   - 重み最適化には時間がかかる

4. Rank Averaging
   - モデル間でスケールが大きく異なる時に有効
   - 頑健性が高い

5. Confidence-based
   - 各モデルが得意なサンプルが異なる時に有効
   - 推論時間は変わらない

6. Stacking
   - 最も高性能だが実装コストが高い
   - OOF予測が必要
   - 過学習に注意

【推奨アプローチ】
① まずSoft Votingで複数モデルを組み合わせる
② スコア差が大きい場合はWeighted Soft Votingを試す
③ さらに改善したい場合はRank AveragingやStacking

【モデルの多様性確保】
- 異なるバックボーン（EfficientNet, ResNet, ViT）
- 異なる画像サイズ（224, 384, 512）
- 異なるaugmentation設定
- 異なるloss関数（ArcFace, CosFace, Triplet）
"""

if __name__ == "__main__":
    # 使用例
    from src.util import Logger
    
    logger = Logger()
    ensemble = EnsembleStrategy(logger)
    
    # ダミーデータで動作確認
    n_samples = 1000
    n_models = 3
    n_classes = 11
    
    # ダミー類似度行列
    similarities_list = [
        torch.randn(n_samples, n_classes).softmax(dim=1) 
        for _ in range(n_models)
    ]
    
    # ダミー予測
    predictions_list = [
        sims.argmax(dim=1).numpy() 
        for sims in similarities_list
    ]
    
    # ダミー正解ラベル
    true_labels = np.random.randint(0, n_classes, n_samples)
    
    # 評価
    results = ensemble.evaluate_ensemble_strategies(
        similarities_list,
        predictions_list,
        true_labels,
        threshold=0.5
    )
    
    print("\n" + "="*80)
    print(results)
