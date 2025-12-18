"""
ResNet50ベースの特徴抽出基底クラス
Prototype法とKNN法の共通部分を実装
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.util import Util
from src.dataset_image import BasketballDataset, get_transforms
import json


class ModelResNet50Base(Model):
    """ResNet50特徴抽出の抽象基底クラス"""

    def __init__(self, run_fold_name: str, params: dict, out_dir_name: str, logger) -> None:
        super().__init__(run_fold_name, params, out_dir_name, logger)
        
        # 共通パラメータ
        self.model_name = params.get('model_name', 'resnet50')
        self.batch_size = params.get('batch_size', 32)
        self.num_workers = params.get('num_workers', 4)
        self.use_cache = params.get('use_cache', True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデル関連
        self.feature_extractor = None
        self.feature_dim = None
        self.train_features = None
        self.train_labels = None
        self.test_similarities = None  # 類似度キャッシュ
        
        # 出力ディレクトリ（特徴量もここに保存）
        self.base_dir = os.path.join(out_dir_name, run_fold_name)
        self.features_dir = os.path.join(self.base_dir, 'features')
        os.makedirs(self.features_dir, exist_ok=True)
        
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


    # 抽象メソッド（子クラスで実装必須）
    def train(self, tr: pd.DataFrame, va: Optional[pd.DataFrame] = None) -> None:
        """学習処理（子クラスで実装）"""
        raise NotImplementedError("Subclass must implement train()")

    # 共通実装（テンプレートメソッドパターン）
    def predict(self, te: pd.DataFrame, split: str = 'test') -> pd.DataFrame:
        """
        テストデータを予測（共通処理）
        
        Args:
            te: テストデータ
            is_validation: 検証データかどうか（split名の決定に使用）
        
        Returns:
            予測結果DataFrame (label_id列)
        """
        # 特徴抽出（キャッシュがあれば読み込み、なければ抽出）
        test_features = self._get_or_extract_features(te, split=split)
        
        self.logger.info(f"{self.__class__.__name__}で予測中...")
        
        # 類似度計算（子クラスで実装）
        similarities = self._compute_similarities(test_features)
        
        # 閾値適用して予測（子クラスで実装）
        predictions = self._predict_from_similarities(similarities)
        
        # 類似度を保存（閾値チューニング用）
        self.test_similarities = similarities
        
        return pd.DataFrame({'label_id': predictions})
    
    def _get_or_extract_features(self, df: pd.DataFrame, split: str = 'train') -> np.ndarray:
        """
        特徴量取得（キャッシュがあれば読み込み、なければ抽出）
        
        Args:
            df: メタデータ
            split: 'train', 'valid', 'test'
        
        Returns:
            特徴量配列 (N, feature_dim)
        """
        if not self.use_cache:
            # キャッシュ無効の場合は毎回抽出
            return self._extract_features_with_dataloader(df, split)
        
        # キャッシュパス
        cache_path = os.path.join(self.features_dir, f"{split}_features.npy")
        
        # キャッシュがあれば読み込み
        if os.path.exists(cache_path):
            self.logger.info(f"特徴量をキャッシュから読み込み: {cache_path}")
            return np.load(cache_path)
        
        # キャッシュがない場合は抽出して保存
        features = self._extract_features_with_dataloader(df, split)
        np.save(cache_path, features)
        self.logger.info(f"特徴量を保存: {cache_path}")
        
        return features
    
    def _extract_features_with_dataloader(self, df: pd.DataFrame, split: str = 'train') -> np.ndarray:
        """
        DataLoaderを使った効率的な特徴抽出（純粋な抽出処理）
        
        Args:
            df: メタデータ
            split: 'train', 'valid', 'test'
        
        Returns:
            特徴量配列 (N, feature_dim)
        """
        self.logger.info(f"{len(df)}枚の画像から特徴抽出中 ({split})...")
        
        # Dataset & DataLoader作成
        is_train = (split == 'train')
        transform = get_transforms(is_train=False)  # Phase 1では拡張なし
        dataset = BasketballDataset(df, transform=transform, is_train=is_train)
        
        use_cuda = torch.cuda.is_available()
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_cuda,  # GPUメモリへの高速転送
            persistent_workers=True if self.num_workers > 0 else False,  # workerの再利用
            prefetch_factor=4 if self.num_workers > 0 else None  # 事前ロード（2→4に増加）
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
                
                # GPU転送（非ブロッキング）
                batch_imgs = batch_imgs.to(self.device, non_blocking=True)
                batch_features = self.feature_extractor(batch_imgs)
                
                # ResNet50の出力: (batch_size, 2048, 1, 1) → (batch_size, 2048)
                # flatten(1)でバッチ次元以外をフラット化
                batch_features = batch_features.flatten(1)
                
                # L2正規化（コサイン類似度計算のため）
                batch_features = batch_features.cpu().numpy()
                
                # バッチサイズが1の場合でも正しく動作するように確認
                if batch_features.ndim == 1:
                    batch_features = batch_features.reshape(1, -1)
                
                norms = np.linalg.norm(batch_features, axis=1, keepdims=True) + 1e-8
                batch_features = batch_features / norms
                
                features_list.append(batch_features)
        
        features = np.vstack(features_list)
        return features
    
    def _compute_similarities(self, test_features: np.ndarray) -> np.ndarray:
        """類似度計算（子クラスで実装）"""
        raise NotImplementedError("Subclass must implement _compute_similarities()")

    def _predict_from_similarities(self, similarities: np.ndarray) -> np.ndarray:
        """類似度から予測（子クラスで実装）"""
        raise NotImplementedError("Subclass must implement _predict_from_similarities()")

    def save_model(self) -> None:
        """モデル保存（子クラスでオーバーライド可能）"""
        raise NotImplementedError("Subclass must implement save_model()")

    def load_model(self) -> None:
        """モデル読み込み（子クラスでオーバーライド可能）"""
        raise NotImplementedError("Subclass must implement load_model()")
