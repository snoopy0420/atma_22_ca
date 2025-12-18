"""
特徴量キャッシュ管理クラス
画像コンペでは特徴抽出が重いため、キャッシュ戦略が重要
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import json

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.util import Util


class FeatureCache:
    """特徴量のキャッシュ管理"""
    
    def __init__(self, cache_dir: str = None):
        """
        Args:
            cache_dir: キャッシュディレクトリ（デフォルト: data/features/）
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(DIR_FEATURE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, df: pd.DataFrame, model_name: str, params: dict, run_fold_name: str = None) -> str:
        """
        データとパラメータからキャッシュキーを生成
        
        Args:
            df: メタデータ
            model_name: モデル名
            params: パラメータ辞書
            run_fold_name: run名_fold番号（指定すればrun/foldごとにキャッシュを分離）
        
        Returns:
            キャッシュキー文字列
        """
        return f"{run_fold_name}"
    
    def exists(self, cache_key: str, split: str = 'train') -> bool:
        """
        キャッシュが存在するか確認
        """
        feature_path = self.cache_dir / f"{cache_key}_{split}_features.npy"
        return feature_path.exists()
    
    def save(self, 
             cache_key: str, 
             features: np.ndarray,
             split: str = 'train'):
        """
        特徴量をキャッシュに保存
        
        Args:
            cache_key: キャッシュキー
            features: 特徴量 (N, feature_dim)
            split: 'train', 'valid', 'test'
        """
        feature_path = self.cache_dir / f"{cache_key}_{split}_features.npy"
        np.save(feature_path, features)
        
        print(f"✓ Cached to: {feature_path.name}")
    
    def load(self, cache_key: str, split: str = 'train') -> np.ndarray:
        """
        キャッシュから特徴量を読み込み
        
        Args:
            cache_key: キャッシュキー
            split: 'train', 'valid', 'test'
        
        Returns:
            features: 特徴量 (N, feature_dim)
        """
        feature_path = self.cache_dir / f"{cache_key}_{split}_features.npy"
        features = np.load(feature_path)
        
        print(f"✓ Loaded from cache: {feature_path.name}")
        return features
    
    def clear(self, cache_key: str = None):
        """キャッシュをクリア"""
        if cache_key:
            # 特定のキーのみクリア
            for file in self.cache_dir.glob(f"{cache_key}_*"):
                file.unlink()
            print(f"✓ Cleared cache: {cache_key}")
        else:
            # 全キャッシュクリア
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            print("✓ Cleared all cache")
    
    def list_caches(self):
        """利用可能なキャッシュ一覧を表示"""
        caches = {}
        for file in self.cache_dir.glob("*_features.npy"):
            key = file.stem.replace('_features', '')
            if key not in caches:
                size_mb = file.stat().st_size / (1024 * 1024)
                caches[key] = {'size_mb': size_mb, 'files': []}
            caches[key]['files'].append(file.name)
        
        if caches:
            print("=== Available Caches ===")
            for key, info in caches.items():
                print(f"  {key}: {info['size_mb']:.1f} MB, {len(info['files'])} files")
        else:
            print("No caches found.")
        
        return caches
