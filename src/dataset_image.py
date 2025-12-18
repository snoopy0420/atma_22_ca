"""
画像コンペ用のDatasetクラス
PyTorchのDatasetを継承して効率的なデータ読み込み
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.abspath('..'))
from configs.config import *


class BasketballDataset(Dataset):
    """バスケットボール選手識別用Dataset"""
    
    def __init__(self, df: pd.DataFrame, transform=None, is_train=True):
        """
        Args:
            df: メタデータDataFrame
            transform: データ拡張・前処理
            is_train: 訓練データかどうか
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 画像読み込み・bbox切り出し
        image = self._load_and_crop(row)
        
        # 前処理・Data Augmentation
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            label = row['label_id']
            return image, label
        else:
            return image
    
    def _load_and_crop(self, row):
        """画像読み込みとbbox切り出し（最適化版）"""
        # 画像パス
        img_path = os.path.join(
            DIR_IMAGE,
            f"{row['quarter']}__{row['angle']}__{str(row['session']).zfill(2)}__{str(row['frame']).zfill(2)}.jpg"
        )
        
        # 読み込み（IMREAD_COLORで高速化）
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"⚠️ Image not found: {img_path}")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"⚠️ Failed to read image: {img_path}")
        
        # BGR→RGB変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # bbox切り出し
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"⚠️ Invalid bbox: x={x}, y={y}, w={w}, h={h}, img_shape={img.shape}")
        cropped = img[y1:y2, x1:x2]
        
        # リサイズ（INTER_AREAで縮小最適化）
        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
        
        return resized


class CachedFeatureDataset(Dataset):
    """特徴量キャッシュ版Dataset（2回目以降の高速化）"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        """
        Args:
            features: 抽出済み特徴量 (N, feature_dim)
            labels: ラベル (N,) - テストデータの場合はNone
        """
        self.features = features
        self.labels = labels
        self.is_train = labels is not None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.is_train:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


def get_transforms(is_train=True, augmentation_level='medium'):
    """
    画像の前処理・Data Augmentation
    
    Args:
        is_train: 訓練データかどうか
        augmentation_level: 'none', 'light', 'medium', 'heavy'
    """
    if is_train:
        if augmentation_level == 'none':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        elif augmentation_level == 'light':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        elif augmentation_level == 'medium':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:  # heavy
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    else:
        # テストデータは固定
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform
