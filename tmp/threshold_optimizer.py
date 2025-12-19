"""
閾値チューニングモジュール
cos類似度の閾値を最適化してMacro F1を最大化
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.util import Metric


class ThresholdOptimizer:
    """閾値最適化クラス
    
    埋め込みベクトルとプロトタイプから最適な閾値を探索
    """
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def log(self, msg):
        """ログ出力"""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def optimize_threshold(self,
                          embeddings: torch.Tensor,
                          prototypes: torch.Tensor,
                          true_labels: np.ndarray,
                          threshold_range: Tuple[float, float] = (0.0, 1.0),
                          n_steps: int = 100) -> Tuple[float, float]:
        """最適閾値を探索
        
        Args:
            embeddings: 埋め込みベクトル [N, embedding_dim]
            prototypes: プロトタイプベクトル [num_classes, embedding_dim]
            true_labels: 正解ラベル [N]
            threshold_range: 探索範囲 (min, max)
            n_steps: 探索ステップ数
        
        Returns:
            best_threshold: 最適閾値
            best_score: 最高スコア（Macro F1）
        """
        self.log("\n" + "="*80)
        self.log("🎯 閾値最適化を開始")
        self.log("="*80)
        
        # cos類似度を計算
        similarities = F.linear(embeddings, prototypes)  # [N, num_classes]
        max_sims, max_indices = similarities.max(dim=1)  # [N]
        
        # 閾値候補
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        scores = []
        
        self.log(f"探索範囲: {threshold_range[0]:.3f} ~ {threshold_range[1]:.3f}")
        self.log(f"ステップ数: {n_steps}")
        
        # 各閾値でスコア計算
        for threshold in tqdm(thresholds, desc="閾値探索"):
            # 閾値で予測を決定
            predictions = []
            for sim, idx in zip(max_sims.tolist(), max_indices.tolist()):
                predictions.append(-1 if sim < threshold else idx)
            
            # Macro F1を計算
            score = Metric.macro_f1(true_labels, np.array(predictions))
            scores.append(score)
        
        # 最適閾値を特定
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        self.log(f"\n最適閾値: {best_threshold:.4f}")
        self.log(f"Macro F1: {best_score:.5f}")
        
        # unknown予測の割合を表示
        test_sims = max_sims.numpy()
        unknown_ratio = (test_sims < best_threshold).mean()
        self.log(f"unknown予測割合: {unknown_ratio*100:.2f}%")
        
        return best_threshold, best_score
    
    def plot_threshold_curve(self,
                            embeddings: torch.Tensor,
                            prototypes: torch.Tensor,
                            true_labels: np.ndarray,
                            output_path: str,
                            threshold_range: Tuple[float, float] = (0.0, 1.0),
                            n_steps: int = 100):
        """閾値とスコアの関係をプロット
        
        Args:
            embeddings: 埋め込みベクトル
            prototypes: プロトタイプベクトル
            true_labels: 正解ラベル
            output_path: 出力ファイルパス
            threshold_range: 探索範囲
            n_steps: ステップ数
        """
        # cos類似度を計算
        similarities = F.linear(embeddings, prototypes)
        max_sims, max_indices = similarities.max(dim=1)
        
        # 閾値ごとのスコア
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        scores = []
        unknown_ratios = []
        
        for threshold in thresholds:
            predictions = []
            for sim, idx in zip(max_sims.tolist(), max_indices.tolist()):
                predictions.append(-1 if sim < threshold else idx)
            
            score = Metric.macro_f1(true_labels, np.array(predictions))
            scores.append(score)
            
            unknown_ratio = (max_sims.numpy() < threshold).mean()
            unknown_ratios.append(unknown_ratio * 100)
        
        # プロット
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Macro F1
        color = 'tab:blue'
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Macro F1', color=color, fontsize=12)
        ax1.plot(thresholds, scores, color=color, linewidth=2, label='Macro F1')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # 最適点をマーク
        best_idx = np.argmax(scores)
        ax1.scatter([thresholds[best_idx]], [scores[best_idx]], 
                   color='red', s=100, zorder=5, label=f'Best: {thresholds[best_idx]:.3f}')
        
        # unknown割合
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Unknown Ratio (%)', color=color, fontsize=12)
        ax2.plot(thresholds, unknown_ratios, color=color, linewidth=2, 
                linestyle='--', label='Unknown Ratio')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # タイトルと凡例
        plt.title('Threshold vs Macro F1 / Unknown Ratio', fontsize=14, fontweight='bold')
        
        # 凡例を統合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"閾値カーブ保存: {output_path}")
    
    def analyze_similarity_distribution(self,
                                       embeddings: torch.Tensor,
                                       prototypes: torch.Tensor,
                                       true_labels: np.ndarray,
                                       output_path: str):
        """類似度分布の可視化
        
        Args:
            embeddings: 埋め込みベクトル
            prototypes: プロトタイプベクトル
            true_labels: 正解ラベル
            output_path: 出力ファイルパス
        """
        # cos類似度を計算
        similarities = F.linear(embeddings, prototypes)
        max_sims, max_indices = similarities.max(dim=1)
        
        # 正解/不正解で分類
        correct_mask = (max_indices.numpy() == true_labels)
        correct_sims = max_sims[correct_mask].numpy()
        incorrect_sims = max_sims[~correct_mask].numpy()
        
        # プロット
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ヒストグラム
        axes[0].hist(correct_sims, bins=50, alpha=0.6, label='Correct', color='green', edgecolor='black')
        axes[0].hist(incorrect_sims, bins=50, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        axes[0].set_xlabel('Max Cosine Similarity', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Similarity Distribution', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 累積分布
        axes[1].hist(correct_sims, bins=50, alpha=0.6, label='Correct', 
                    color='green', cumulative=True, density=True, histtype='step', linewidth=2)
        axes[1].hist(incorrect_sims, bins=50, alpha=0.6, label='Incorrect', 
                    color='red', cumulative=True, density=True, histtype='step', linewidth=2)
        axes[1].set_xlabel('Max Cosine Similarity', fontsize=12)
        axes[1].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1].set_title('Cumulative Distribution', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"類似度分布保存: {output_path}")
        
        # 統計情報
        self.log("\n【類似度統計】")
        self.log(f"正解時: 平均={correct_sims.mean():.4f}, 中央値={np.median(correct_sims):.4f}, 標準偏差={correct_sims.std():.4f}")
        self.log(f"不正解時: 平均={incorrect_sims.mean():.4f}, 中央値={np.median(incorrect_sims):.4f}, 標準偏差={incorrect_sims.std():.4f}")


def optimize_threshold_from_oof(model_dir: str, 
                                train_df: pd.DataFrame,
                                logger=None) -> float:
    """OOF予測から最適閾値を探索
    
    Args:
        model_dir: モデルディレクトリ
        train_df: 訓練データ
        logger: ロガー
    
    Returns:
        最適閾値
    """
    from src.util import Util
    
    optimizer = ThresholdOptimizer(logger)
    
    # OOF予測とプロトタイプを読み込み（fold0のプロトタイプを使用）
    # 実際には全foldのembeddingを使うべきだが、簡易版としてfold0を使用
    oof_path = os.path.join(model_dir, 'va_pred.pkl')
    prototype_path = os.path.join(model_dir, f'{os.path.basename(model_dir)}_fold-0', 'prototypes.pth')
    
    if not os.path.exists(prototype_path):
        optimizer.log(f"プロトタイプが見つかりません: {prototype_path}")
        return 0.5
    
    # ここでは簡易実装として、埋め込みを再計算せずデフォルト値を返す
    # 実際には各foldで埋め込みを抽出して最適化すべき
    optimizer.log("閾値最適化はNotebookで実行してください")
    
    return 0.5


if __name__ == "__main__":
    # 使用例
    from src.util import Logger
    
    logger = Logger()
    logger.info("閾値最適化ツールのテスト")
    
    # ダミーデータで動作確認
    embeddings = torch.randn(1000, 512)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    prototypes = torch.randn(11, 512)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    true_labels = np.random.randint(0, 11, 1000)
    
    optimizer = ThresholdOptimizer(logger)
    best_threshold, best_score = optimizer.optimize_threshold(
        embeddings, prototypes, true_labels,
        threshold_range=(0.3, 0.8),
        n_steps=50
    )
    
    # プロット
    optimizer.plot_threshold_curve(
        embeddings, prototypes, true_labels,
        output_path='/workspace/atma_22_ca/data/figures/threshold_curve_test.png',
        threshold_range=(0.3, 0.8),
        n_steps=50
    )
    
    optimizer.analyze_similarity_distribution(
        embeddings, prototypes, true_labels,
        output_path='/workspace/atma_22_ca/data/figures/similarity_dist_test.png'
    )
