"""
KNN法によるResNet50特徴抽出モデル
訓練データ全体との類似度からTop-k多数決で予測
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

sys.path.append(os.path.abspath('..'))
from src.model_resnet_base import ModelResNet50Base
from src.util import Util


class ModelResNet50KNN(ModelResNet50Base):
    """KNN法: 訓練データとの類似度Top-kで多数決予測"""

    def __init__(self, run_fold_name: str, params: dict, out_dir_name: str, logger) -> None:
        # KNN法固有のパラメータ
        self.k = params.get('k', 5)
        self.threshold = params.get('threshold', 0.5)
        self.min2_threshold = params.get('min2_threshold', 0.3)
        
        # 基底クラスの初期化
        super().__init__(run_fold_name, params, out_dir_name, logger)

    def train(self, tr: pd.DataFrame, va: Optional[pd.DataFrame] = None) -> None:
        """
        学習データから特徴抽出のみ（KNNは訓練データをそのまま使用）
        
        Args:
            tr: 訓練データ
            va: 検証データ（未実装）
        """
        self.logger.info(f"Training KNN method (k={self.k}) with {len(tr)} samples...")
        
        # 特徴抽出（基底クラスのメソッド + キャッシュ自動管理）
        self.train_features = self._extract_features_batch(tr, split='train')
        self.train_labels = tr['label_id'].values
        
        self.logger.info(f"Train features shape: {self.train_features.shape}")
        self.logger.info(f"KNN will use all {len(self.train_labels)} training samples for prediction")
        
        # Validationがあれば評価
        if va is not None:
            self.logger.info("Validation not implemented yet")

    def _compute_similarities(self, test_features: np.ndarray) -> np.ndarray:
        """
        全訓練データとの類似度を計算
        
        Args:
            test_features: テスト特徴量 (N, feature_dim)
        
        Returns:
            類似度行列 (N, n_train_samples)
        """
        # コサイン類似度 = 内積（L2正規化済みのため）
        similarities = test_features @ self.train_features.T
        
        return similarities

    def _predict_from_similarities(self, similarities: np.ndarray) -> np.ndarray:
        """
        類似度から予測（Top-k多数決 + 閾値）
        
        Args:
            similarities: 類似度行列 (N, n_train_samples)
        
        Returns:
            予測ラベル (N,)
        """
        predictions = []
        
        for sims in similarities:
            # Top-k取得
            topk_idx = np.argsort(sims)[-self.k:][::-1]
            topk_labels = self.train_labels[topk_idx]
            topk_sims = sims[topk_idx]
            
            # 閾値判定
            if topk_sims[0] >= self.threshold:
                # 1位が閾値以上 → Top-kで多数決
                unique, counts = np.unique(topk_labels, return_counts=True)
                predictions.append(unique[np.argmax(counts)])
            elif topk_sims[1] >= self.min2_threshold:
                # 2位が閾値以上 → 2位のラベル
                predictions.append(topk_labels[1])
            else:
                # どちらも閾値未満 → unknown
                predictions.append(-1)
        
        return np.array(predictions)

    def predict_with_custom_threshold(self, threshold: float, min2_threshold: float = None, k: int = None) -> np.ndarray:
        """
        異なる閾値・kで予測（類似度キャッシュを再利用）
        
        Args:
            threshold: 新しい閾値
            min2_threshold: 新しいmin2閾値（Noneなら元の値）
            k: 新しいk（Noneなら元の値）
        
        Returns:
            予測結果
        """
        if self.test_similarities is None:
            raise ValueError("No cached similarities. Run predict() first.")
        
        # 一時的にパラメータを変更
        original_threshold = self.threshold
        original_min2 = self.min2_threshold
        original_k = self.k
        
        self.threshold = threshold
        if min2_threshold is not None:
            self.min2_threshold = min2_threshold
        if k is not None:
            self.k = k
        
        # 予測
        predictions = self._predict_from_similarities(self.test_similarities)
        
        # 元に戻す
        self.threshold = original_threshold
        self.min2_threshold = original_min2
        self.k = original_k
        
        return predictions

    def save_model(self) -> None:
        """モデル保存"""
        model_path = os.path.join(self.base_dir, f'{self.run_fold_name}.pkl')
        model_data = {
            'train_features': self.train_features,
            'train_labels': self.train_labels,
            'k': self.k
        }
        Util.dump(model_data, model_path)
        self.logger.info(f"Model saved: {model_path}")

    def load_model(self) -> None:
        """モデル読み込み"""
        model_path = os.path.join(self.base_dir, f'{self.run_fold_name}.pkl')
        model_data = Util.load(model_path)
        
        self.train_features = model_data['train_features']
        self.train_labels = model_data['train_labels']
        self.k = model_data.get('k', self.k)
        
        self.logger.info(f"Loaded features: {self.train_features.shape}")
        self.logger.info(f"KNN k={self.k}")
