"""
ResNet50ベースの画像特徴抽出モデル
画像コンペ向けに改良: DataLoader + FeatureCache対応
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.util import Util
from src.dataset_image import BasketballDataset, get_transforms
from src.feature_cache import FeatureCache


class ModelResNet50(Model):
    """ResNet50を使った特徴抽出・KNN予測モデル"""

    def __init__(self, run_fold_name: str, params: dict, out_dir_name: str, logger) -> None:
        super().__init__(run_fold_name, params, out_dir_name, logger)
        
        # パラメータ
        self.model_name = params.get('model_name', 'resnet50')
        self.method = params.get('method', 'prototype')  # 'prototype' or 'knn'
        self.k = params.get('k', 5)
        self.threshold = params.get('threshold', 0.5)
        self.min2_threshold = params.get('min2_threshold', 0.3)
        self.batch_size = params.get('batch_size', 32)
        self.num_workers = params.get('num_workers', 4)
        self.use_cache = params.get('use_cache', True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデル関連
        self.feature_extractor = None
        self.train_features = None
        self.train_labels = None
        self.test_similarities = None  # 類似度キャッシュ
        self.prototypes = None
        
        # キャッシュ管理
        self.cache_manager = FeatureCache()
        
        # 出力ディレクトリ
        self.base_dir = os.path.join(out_dir_name, run_fold_name)
        os.makedirs(self.base_dir, exist_ok=True)
        
        # モデルのロード
        self._load_feature_extractor()


    def _load_feature_extractor(self):
        """特徴抽出器の初期化"""
        if self.model_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        elif self.model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()


    def _extract_features_batch(self, df: pd.DataFrame, split: str = 'train') -> np.ndarray:
        """
        DataLoaderを使った効率的な特徴抽出
        
        Args:
            df: メタデータ
            split: 'train' or 'test' (キャッシュキー生成用)
        """
        # キャッシュチェック
        if self.use_cache:
            cache_key = self.cache_manager.get_cache_key(df, self.model_name, self.params)

            print(cache_key)
            
            if self.cache_manager.exists(cache_key, split):
                self.logger.info(f"Loading features from cache ({split})...")
                features, labels = self.cache_manager.load(cache_key, split)
                return features
        
        # キャッシュがない場合は抽出
        self.logger.info(f"Extracting features from {len(df)} images ({split})...")
        
        # Dataset & DataLoader作成
        is_train = (split == 'train')
        transform = get_transforms(is_train=False)  # Phase 1では拡張なし
        dataset = BasketballDataset(df, transform=transform, is_train=is_train)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # バッチごとに特徴抽出
        features_list = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=f"Extracting {split}"):
                # is_trainによって返り値が異なる
                if is_train:
                    batch_imgs, _ = batch_data
                else:
                    batch_imgs = batch_data
                
                batch_imgs = batch_imgs.to(self.device)
                batch_features = self.feature_extractor(batch_imgs)
                batch_features = batch_features.squeeze()
                
                # 正規化
                batch_features = batch_features.cpu().numpy()
                norms = np.linalg.norm(batch_features, axis=1, keepdims=True) + 1e-8
                batch_features = batch_features / norms
                
                features_list.append(batch_features)
        
        features = np.vstack(features_list)
        
        # キャッシュに保存
        if self.use_cache:
            labels = df['label_id'].values if 'label_id' in df.columns else None
            self.cache_manager.save(cache_key, split, features, labels, 
                                   {'model': self.model_name, 'params': self.params})
            self.logger.info(f"Saved features to cache ({split})")
        
        return features


    def _compute_prototypes(self):
        """クラスごとの平均特徴量を計算"""
        unique_labels = np.unique(self.train_labels)
        prototypes = {}
        
        for label in unique_labels:
            mask = self.train_labels == label
            class_features = self.train_features[mask]
            prototype = class_features.mean(axis=0)
            prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
            prototypes[label] = prototype
        
        self.logger.info(f"Computed {len(prototypes)} prototypes")
        return prototypes


    def train(self, tr: pd.DataFrame, va: Optional[pd.DataFrame] = None) -> None:
        """学習データから特徴抽出 + キャッシュ対応"""
        self.logger.info(f"Training with {len(tr)} samples...")
        
        # 特徴抽出（キャッシュ自動管理）
        self.train_features = self._extract_features_batch(tr, split='train')
        self.train_labels = tr['label_id'].values
        
        self.logger.info(f"Train features shape: {self.train_features.shape}")
        
        # Prototype法の場合は事前計算
        if self.method == 'prototype':
            self.prototypes = self._compute_prototypes()
        
        # Validationがあれば評価
        if va is not None:
            self.logger.info("Validation not implemented yet")


    def predict(self, te: pd.DataFrame) -> pd.DataFrame:
        """テストデータを予測 + 類似度キャッシュ"""
        # テスト特徴抽出（キャッシュ自動管理）
        test_features = self._extract_features_batch(te, split='test')
        
        self.logger.info(f"Predicting with {self.method} method...")
        
        # 類似度計算（閾値調整用にキャッシュ）
        similarities = self._compute_similarities(test_features)
        
        # 閾値適用して予測
        predictions = self._apply_threshold(similarities)
        
        # 結果をDataFrameに
        result_df = pd.DataFrame({'label_id': predictions})
        
        # 類似度を保存（閾値チューニング用）
        self.test_similarities = similarities
        
        return result_df
    
    def _compute_similarities(self, test_features: np.ndarray) -> np.ndarray:
        """類似度計算（閾値適用前）"""
        if self.method == 'prototype':
            # プロトタイプとの類似度
            labels = sorted(self.prototypes.keys())
            prototype_matrix = np.array([self.prototypes[label] for label in labels])
            similarities = test_features @ prototype_matrix.T
        else:  # knn
            # 全訓練データとの類似度
            similarities = test_features @ self.train_features.T
        
        return similarities
    
    def _apply_threshold(self, similarities: np.ndarray) -> np.ndarray:
        """閾値を適用して予測（類似度から分離）"""
        if self.method == 'prototype':
            return self._predict_prototype_from_similarities(similarities)
        else:
            return self._predict_knn_from_similarities(similarities)

    def _predict_prototype_from_similarities(self, similarities: np.ndarray) -> np.ndarray:
        """Prototype法: 類似度から予測（閾値チューニング可能）"""
        predictions = []
        labels = sorted(self.prototypes.keys())
        
        for sims in similarities:
            # 最も類似度が高い2つ
            top2_idx = np.argsort(sims)[-2:][::-1]
            top1_sim = sims[top2_idx[0]]
            top2_sim = sims[top2_idx[1]]
            
            # 閾値判定
            if top1_sim >= self.threshold:
                predictions.append(labels[top2_idx[0]])
            elif top2_sim >= self.min2_threshold:
                predictions.append(labels[top2_idx[1]])
            else:
                predictions.append(-1)  # unknown
        
        return np.array(predictions)

    def _predict_knn_from_similarities(self, similarities: np.ndarray) -> np.ndarray:
        """KNN法: 類似度から予測（閾値チューニング可能）"""
        predictions = []
        
        for sims in similarities:
            # Top-k取得
            topk_idx = np.argsort(sims)[-self.k:][::-1]
            topk_labels = self.train_labels[topk_idx]
            topk_sims = sims[topk_idx]
            
            # 閾値判定
            if topk_sims[0] >= self.threshold:
                # 多数決
                unique, counts = np.unique(topk_labels, return_counts=True)
                predictions.append(unique[np.argmax(counts)])
            elif topk_sims[1] >= self.min2_threshold:
                predictions.append(topk_labels[1])
            else:
                predictions.append(-1)  # unknown
        
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
        predictions = self._apply_threshold(self.test_similarities)
        
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
            'prototypes': self.prototypes
        }
        Util.dump(model_data, model_path)
        self.logger.info(f"Model saved: {model_path}")
    
    def load_model(self) -> None:
        """モデル読み込み"""
        model_path = os.path.join(self.base_dir, f'{self.run_fold_name}.pkl')
        model_data = Util.load(model_path)
        
        if isinstance(model_data, dict):
            # 新形式（辞書）
            self.train_features = model_data['train_features']
            self.train_labels = model_data['train_labels']
            self.prototypes = model_data['prototypes']
        else:
            # 旧形式（特徴量のみ）- 互換性のため
            self.train_features = model_data
            self.logger.warning("Loaded old format model (features only). train_labels and prototypes are not available.")
        
        self.logger.info(f"Loaded features: {self.train_features.shape}")

