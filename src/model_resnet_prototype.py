"""
Prototype法によるResNet50特徴抽出モデル
各クラスの平均特徴量（プロトタイプ）との類似度で予測
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

sys.path.append(os.path.abspath('..'))
from src.model_resnet_base import ModelResNet50Base
from src.util import Util


class ModelResNet50Prototype(ModelResNet50Base):
    """Prototype法: クラス平均特徴量との類似度で予測"""

    def __init__(self, run_fold_name: str, params: dict, out_dir_name: str, logger) -> None:
        # Prototype法固有のパラメータ
        self.threshold = params.get('threshold', 0.5)
        self.min2_threshold = params.get('min2_threshold', 0.3)
        self.prototypes = None
        self.prototype_labels = None
        
        # 基底クラスの初期化
        super().__init__(run_fold_name, params, out_dir_name, logger)

    def train(self, tr: pd.DataFrame, va: Optional[pd.DataFrame] = None) -> None:
        """
        学習データから特徴抽出 + プロトタイプ計算
        
        Args:
            tr: 訓練データ
            va: 検証データ（未実装）
        """
        self.logger.info(f"Training Prototype method with {len(tr)} samples...")
        
        # 特徴抽出（基底クラスのメソッド + キャッシュ自動管理）
        self.train_features = self._extract_features_batch(tr, split='train')
        self.train_labels = tr['label_id'].values
        
        self.logger.info(f"Train features shape: {self.train_features.shape}")
        
        # プロトタイプ計算
        self.prototypes = self._compute_prototypes()
        
        # Validationがあれば評価
        if va is not None:
            self.logger.info("Validation not implemented yet")

    def _compute_prototypes(self) -> dict:
        """
        クラスごとの平均特徴量（プロトタイプ）を計算
        
        Returns:
            {label: prototype_vector} の辞書
        """
        unique_labels = np.unique(self.train_labels)
        prototypes = {}
        
        for label in unique_labels:
            mask = self.train_labels == label
            class_features = self.train_features[mask]
            
            # 平均ベクトルを計算
            prototype = class_features.mean(axis=0)
            
            # L2正規化
            prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
            
            prototypes[label] = prototype
        
        # ラベル順序を固定（類似度計算と予測で一貫性を保つ）
        self.prototype_labels = sorted(prototypes.keys())
        
        self.logger.info(f"Computed {len(prototypes)} prototypes: {self.prototype_labels}")
        return prototypes

    def _compute_similarities(self, test_features: np.ndarray) -> np.ndarray:
        """
        プロトタイプとの類似度を計算
        
        Args:
            test_features: テスト特徴量 (N, feature_dim)
        
        Returns:
            類似度行列 (N, n_classes)
        """
        # プロトタイプを行列に変換（固定されたラベル順序を使用）
        prototype_matrix = np.array([self.prototypes[label] for label in self.prototype_labels])
        
        # コサイン類似度 = 内積（L2正規化済みのため）
        similarities = test_features @ prototype_matrix.T
        
        return similarities

    def _predict_from_similarities(self, similarities: np.ndarray) -> np.ndarray:
        """
        類似度から予測（閾値ベース）
        
        Args:
            similarities: 類似度行列 (N, n_classes)
        
        Returns:
            予測ラベル (N,)
        """
        predictions = []
        
        for sims in similarities:
            # 最も類似度が高い2つを取得
            top2_idx = np.argsort(sims)[-2:][::-1]
            top1_sim = sims[top2_idx[0]]
            top2_sim = sims[top2_idx[1]]
            
            # 閾値判定
            if top1_sim >= self.threshold:
                # 1位が閾値以上 → 1位のクラス
                predictions.append(self.prototype_labels[top2_idx[0]])
            elif top2_sim >= self.min2_threshold:
                # 2位が閾値以上 → 2位のクラス
                predictions.append(self.prototype_labels[top2_idx[1]])
            else:
                # どちらも閾値未満 → unknown
                predictions.append(-1)
        
        return np.array(predictions)

    def predict_with_custom_threshold(self, threshold: float, min2_threshold: float = None) -> np.ndarray:
        """
        異なる閾値で予測（類似度キャッシュを再利用）
        
        Args:
            threshold: 新しい閾値
            min2_threshold: 新しいmin2閾値（Noneなら元の値）
        
        Returns:
            予測結果
        """
        if self.test_similarities is None:
            raise ValueError("No cached similarities. Run predict() first.")
        
        # 一時的に閾値を変更
        original_threshold = self.threshold
        original_min2 = self.min2_threshold
        
        self.threshold = threshold
        if min2_threshold is not None:
            self.min2_threshold = min2_threshold
        
        # 予測
        predictions = self._predict_from_similarities(self.test_similarities)
        
        # 元に戻す
        self.threshold = original_threshold
        self.min2_threshold = original_min2
        
        return predictions

    def save_model(self) -> None:
        """モデル保存"""
        model_path = os.path.join(self.base_dir, f'{self.run_fold_name}.pkl')
        model_data = {
            'train_features': self.train_features,
            'train_labels': self.train_labels,
            'prototypes': self.prototypes,
            'prototype_labels': self.prototype_labels
        }
        Util.dump(model_data, model_path)
        self.logger.info(f"Model saved: {model_path}")

    def load_model(self) -> None:
        """モデル読み込み"""
        model_path = os.path.join(self.base_dir, f'{self.run_fold_name}.pkl')
        model_data = Util.load(model_path)
        
        self.train_features = model_data['train_features']
        self.train_labels = model_data['train_labels']
        self.prototypes = model_data['prototypes']
        self.prototype_labels = model_data.get('prototype_labels')
        
        # prototype_labelsがない場合は再生成
        if self.prototype_labels is None and self.prototypes is not None:
            self.prototype_labels = sorted(self.prototypes.keys())
            self.logger.warning("prototype_labels not found, regenerated from prototypes")
        
        self.logger.info(f"Loaded features: {self.train_features.shape}")
        if self.prototype_labels:
            self.logger.info(f"Prototype labels: {self.prototype_labels}")
