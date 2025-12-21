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
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

sys.path.append(os.path.abspath('..'))
from configs.config import *


# ============================================================================
# テスト環境を模倣した拡張（注意事項のギャップを埋める）
# ============================================================================

class TestLikeAugmentation(A.ImageOnlyTransform):
    """テストデータの特性を模倣した拡張
    
    学習/テストのギャップ:
    1. BBox位置ずれ: テストのBBox精度が不正確
    2. 部分的な切り取り: IoU < 0.5のケース
    3. 照明変化: 時系列（Q1/Q2→Q4）による変化
    """
    
    def __init__(
        self,
        bbox_shift_prob: float = 0.3,
        bbox_shift_max: float = 0.15,
        partial_crop_prob: float = 0.2,
        partial_crop_ratio: float = 0.6,
        lighting_prob: float = 0.3,
        lighting_intensity: float = 0.3,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """
        Args:
            bbox_shift_prob: BBox位置ずれ適用確率
            bbox_shift_max: BBox位置ずれの最大量（画像サイズの割合）
            partial_crop_prob: 部分切り取り適用確率
            partial_crop_ratio: 切り取り後の残存割合
            lighting_prob: 照明変化適用確率
            lighting_intensity: 照明変化の強度
            p: この拡張全体の適用確率
        """
        super().__init__(always_apply, p)
        self.bbox_shift_prob = bbox_shift_prob
        self.bbox_shift_max = bbox_shift_max
        self.partial_crop_prob = partial_crop_prob
        self.partial_crop_ratio = partial_crop_ratio
        self.lighting_prob = lighting_prob
        self.lighting_intensity = lighting_intensity
    
    def apply(self, img, **params):
        """拡張を適用"""
        # 再現性確保のためnumpyのrandomを使用（Albumentationsが自動的にシードを管理）
        # (1) BBox位置ずれ
        if np.random.random() < self.bbox_shift_prob:
            img = self._bbox_shift(img)
        
        # (2) 部分的な切り取り
        if np.random.random() < self.partial_crop_prob:
            img = self._partial_crop(img)
        
        # (3) 照明変化
        if np.random.random() < self.lighting_prob:
            img = self._lighting_shift(img)
        
        return img
    
    def _bbox_shift(self, image: np.ndarray) -> np.ndarray:
        """BBox位置ずれを模倣"""
        h, w = image.shape[:2]
        
        # ランダムにシフト（numpy.random使用）
        shift_x = int(w * np.random.uniform(-self.bbox_shift_max, self.bbox_shift_max))
        shift_y = int(h * np.random.uniform(-self.bbox_shift_max, self.bbox_shift_max))
        
        # シフト後の範囲計算
        new_x1 = max(0, shift_x)
        new_y1 = max(0, shift_y)
        new_x2 = min(w, w + shift_x)
        new_y2 = min(h, h + shift_y)
        
        # 元画像からクロップ
        crop_x1 = max(0, -shift_x)
        crop_y1 = max(0, -shift_y)
        crop_x2 = crop_x1 + (new_x2 - new_x1)
        crop_y2 = crop_y1 + (new_y2 - new_y1)
        
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # リサイズして元サイズに戻す
        return cv2.resize(cropped, (w, h))
    
    def _partial_crop(self, image: np.ndarray) -> np.ndarray:
        """部分的な切り取り（IoU < 0.5を模倣）"""
        h, w = image.shape[:2]
        
        # ランダムに辺を選択（numpy.random使用）
        side = np.random.choice(['top', 'bottom', 'left', 'right'])
        crop_amount = 1 - self.partial_crop_ratio
        
        if side == 'top':
            crop_h = int(h * crop_amount)
            cropped = image[crop_h:, :]
        elif side == 'bottom':
            crop_h = int(h * crop_amount)
            cropped = image[:-crop_h, :]
        elif side == 'left':
            crop_w = int(w * crop_amount)
            cropped = image[:, crop_w:]
        else:  # right
            crop_w = int(w * crop_amount)
            cropped = image[:, :-crop_w]
        
        # リサイズして元サイズに戻す
        return cv2.resize(cropped, (w, h))
    
    def _lighting_shift(self, image: np.ndarray) -> np.ndarray:
        """照明変化（時系列ギャップを模倣）"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 明度変化（numpy.random使用）
        brightness_factor = 1 + np.random.uniform(-self.lighting_intensity, self.lighting_intensity)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        
        # 色温度変化（numpy.random使用）
        hue_shift = np.random.uniform(-10, 10)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ============================================================================
# Resnet用のDatasetクラス
# ============================================================================

class BasketballDataset(Dataset):
    """バスケットボール選手識別用Dataset"""
    
    def __init__(self, df: pd.DataFrame, transform=None, is_train=True):
        """
        Args:
            df: メタデータDataFrame
            transform: データ拡張・前処理
            is_train: 訓練データかどうか（False=テストデータ）
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 画像読み込み
        image = self._load_and_crop(row)
        
        # 前処理・Data Augmentation
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.is_train:
            label = torch.tensor(row['label_id'], dtype=torch.long)
            return image, label
        else:
            return image
    
    def _load_and_crop(self, row):
        """画像読み込みとbbox切り出し（最適化版）"""
        # 学習データ、検証データ: train_crops/から前処理済みデータを読み込み
        if self.is_train:
            from pathlib import Path
            train_crop_dir = Path(DIR_TRAIN_CROPS)
            if train_crop_dir.exists():
                idx = row.name
                crop_path = train_crop_dir / f"{idx}.jpg"
                if crop_path.exists():
                    img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        
        # テストデータ: crops/{rel_path}から読み込み
        if not self.is_train and 'rel_path' in row and pd.notna(row['rel_path']):
            img_path = os.path.join(DIR_CROPS, row['rel_path'])
            if os.path.exists(img_path):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        
        
        raise FileNotFoundError(f"画像が見つかりません: {row}")


# ============================================================================
# Redbullベースライン互換Datasetクラス
# ============================================================================

class PlayerDataset(Dataset):
    """プレイヤー再識別用Dataset（RedBullベースライン互換）
    
    機能:
    - 事前クロップ画像の読み込み（crop_dir指定時）
    - オンザフライクロップ（パディング10%付き）
    - 画像キャッシュ機能（cache_images=True時）
    - RGB変換を明示的に実施
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        cache_images: bool = False,
    ):
        self.original_indices = df.index.tolist()  # 元のインデックスを保持
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        self.crop_dir = Path(DIR_TRAIN_CROPS)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_image(self, img_path: Path) -> np.ndarray:
        """画像を読み込み（キャッシュ対応）"""
        if self.cache_images and str(img_path) in self.image_cache:
            return self.image_cache[str(img_path)]
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
        
        if self.cache_images:
            self.image_cache[str(img_path)] = img
        
        return img
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # 事前クロップ画像を直接読み込む（元のインデックスを使用）
        original_idx = self.original_indices[idx]
        crop_path = self.crop_dir / f"{original_idx}.jpg"
        crop = self._load_image(crop_path)
        
        # Albumentations変換を適用
        transformed = self.transform(image=crop)
        image = transformed['image']
        
        # 重要: rowはreset_index後のものなので、label_idは正しく対応している
        result = {
            'image': image,
            'angle': row['angle'],
            'label': torch.tensor(row['label_id'], dtype=torch.long),
        }
        
        return result


class TestDataset(Dataset):
    """テストデータ用Dataset"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
    ):
        """
        Args:
            df: test_meta.csv
            transform: Albumentations変換
        """
        self.df = df.reset_index(drop=True)
        self.base_dir = DIR_INPUT
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:

        # rel_pathから直接画像を読み込む
        row = self.df.iloc[idx]
        img_path = f"{self.base_dir}/{row['rel_path']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 変換を適用
        transformed = self.transform(image=img)
        image = transformed['image']
        
        return {'image': image}


def _create_dataloader(df: pd.DataFrame, split: str = 'train', batch_size: int=32, num_workers: int=8, img_size: int=224) -> DataLoader:
    """
    DataLoaderを作成
    
    Args:
        df: データフレーム
        split: 'train', 'valid', 'test'のいずれか
        batch_size: バッチサイズ
        num_workers: データローダーのワーカー数
        img_size: 画像サイズ
    
    注: split='valid'の場合、テスト環境を模倣した拡張が自動的に適用されます。
    """
    
    # カスタムcollate_fn: 辞書のリストをバッチ化
    def collate_fn(batch):
        if split == 'test':
            # テストデータは画像のみ
            return {'image': torch.stack([item['image'] for item in batch])}
        else:
            # 訓練/検証データは辞書形式
            return {
                'image': torch.stack([item['image'] for item in batch]),
                'label': torch.stack([item['label'] for item in batch]),
                'angle': [item['angle'] for item in batch],
            }
    
    # テストデータは専用Datasetを使用（rel_path対応）
    if split == 'test':
        dataset = TestDataset(
            df=df,
            transform=_get_transforms(split='test', img_size=img_size),
        )
    elif split == 'train':
        dataset = PlayerDataset(
            df=df,
            transform=_get_transforms(split='train', img_size=img_size),
            cache_images=False,  # 大規模データセットではメモリ不足に注意
        )
    elif split == 'valid':
        # 検証データ: テスト環境模倣拡張が自動適用
        dataset = PlayerDataset(
            df=df,
            transform=_get_transforms(split='valid', img_size=img_size),
            cache_images=False,  # 大規模データセットではメモリ不足に注意
        )
    
    # DataLoader作成（RedBullの設定を踏襲）
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train']),  # 訓練時のみシャッフル
        num_workers=num_workers,  # マルチプロセス読み込み
        pin_memory=True,  # GPU転送高速化
        drop_last=(split in ['train']),  # 訓練時のみ端数バッチを削除（安定性向上）
        persistent_workers=num_workers,  # ワーカープロセスの再利用（高速化）
        prefetch_factor=4,  # プリフェッチ数（I/O効率化）
        collate_fn=collate_fn,  # カスタムバッチ化関数
    )


def _get_transforms(split: str = 'train', img_size: int = 224) -> A.Compose:
    """
    画像変換パイプラインを取得
    
    訓練時 (split='train'):
    - データ拡張（Data Augmentation）で多様性を増やし過学習防止
    - 幾何学的変換: Flip, Shift, Scale, Rotate
    - 色変換: Brightness, Contrast, Saturation, Hue
    
    検証時 (split='valid'):
    - テスト環境を模倣した拡張を必ず適用
    - BBox位置ずれ、部分切り取り、照明変化を含む
    
    テスト時 (split='test'):
    - リサイズと正規化のみ
    
    Args:
        split: 'train', 'valid', 'test'のいずれか
        img_size: リサイズ後の画像サイズ
    
    Returns:
        Albumentations変換パイプライン
    """
    if split == 'train':
        # 訓練時: 通常のデータ拡張
        return A.Compose([
            A.Resize(img_size, img_size),  # リサイズ（例: 224x224）
            A.HorizontalFlip(p=0.5),  # 水平反転（50%確率）
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # 平行移動/拡大縮小/回転
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # 色調変換
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet統計で正規化
            ToTensorV2(),  # NumPy配列→PyTorchテンソル変換
        ])
    
    elif split == 'valid':
        # 検証時: テスト環境模倣拡張を必ず適用
        return A.Compose([
            # テスト環境模倣拡張
            TestLikeAugmentation(
                bbox_shift_prob=0.3,
                bbox_shift_max=0.15,
                partial_crop_prob=0.2,
                partial_crop_ratio=0.6,
                lighting_prob=0.3,
                lighting_intensity=0.3,
                p=1.0,
            ),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet正規化
            ToTensorV2(),
        ])
    
    else:  # split == 'test'
        # テスト時: 拡張なし（リサイズと正規化のみ）
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
