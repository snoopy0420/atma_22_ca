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
    
    def get_cache_key(self, df: pd.DataFrame, model_name: str, params: dict) -> str:
        """
        データとパラメータからキャッシュキーを生成
        
        Args:
            df: メタデータ
            model_name: モデル名
            params: パラメータ辞書
        
        Returns:
            キャッシュキー文字列
        """
        # データのハッシュ（行数、列名、最初と最後の行）
        data_hash = hashlib.md5(
            f"{len(df)}_{list(df.columns)}_{df.iloc[0].to_dict()}_{df.iloc[-1].to_dict()}".encode()
        ).hexdigest()[:8]
        
        # パラメータのハッシュ（キャッシュに影響するもののみ）
        cache_params = {
            'model_name': model_name,
            'batch_size': params.get('batch_size'),
        }
        param_hash = hashlib.md5(
            json.dumps(cache_params, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"{model_name}_{data_hash}_{param_hash}"
    
    def exists(self, cache_key: str, split: str = 'train') -> bool:
        """
        キャッシュが存在するか確認
        
        Args:
            cache_key: キャッシュキー
            split: 'train' or 'test'
        """
        feature_path = self.cache_dir / f"{cache_key}_{split}_features.npy"
        
        if split == 'train':
            label_path = self.cache_dir / f"{cache_key}_{split}_labels.npy"
            return feature_path.exists() and label_path.exists()
        else:
            return feature_path.exists()
    
    def save(self, 
             cache_key: str, 
             features: np.ndarray, 
             labels: Optional[np.ndarray] = None,
             split: str = 'train',
             metadata: dict = None):
        """
        特徴量をキャッシュに保存
        
        Args:
            cache_key: キャッシュキー
            features: 特徴量 (N, feature_dim)
            labels: ラベル (N,) - テストの場合はNone
            split: 'train' or 'test'
            metadata: メタデータ（保存時刻、データサイズなど）
        """
        feature_path = self.cache_dir / f"{cache_key}_{split}_features.npy"
        np.save(feature_path, features)
        
        if labels is not None:
            label_path = self.cache_dir / f"{cache_key}_{split}_labels.npy"
            np.save(label_path, labels)
        
        # メタデータ保存
        if metadata:
            meta_path = self.cache_dir / f"{cache_key}_{split}_meta.json"
            Util.dump_json(metadata, str(meta_path))
        
        print(f"✓ Cached to: {feature_path.name}")
    
    def load(self, cache_key: str, split: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        キャッシュから特徴量を読み込み
        
        Args:
            cache_key: キャッシュキー
            split: 'train' or 'test'
        
        Returns:
            (features, labels) - テストの場合labels=None
        """
        feature_path = self.cache_dir / f"{cache_key}_{split}_features.npy"
        features = np.load(feature_path)
        
        if split == 'train':
            label_path = self.cache_dir / f"{cache_key}_{split}_labels.npy"
            labels = np.load(label_path)
        else:
            labels = None
        
        print(f"✓ Loaded from cache: {feature_path.name}")
        return features, labels
    
    def get_metadata(self, cache_key: str, split: str = 'train') -> Optional[dict]:
        """キャッシュのメタデータを取得"""
        meta_path = self.cache_dir / f"{cache_key}_{split}_meta.json"
        if meta_path.exists():
            return Util.load_json(str(meta_path))
        return None
    
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
