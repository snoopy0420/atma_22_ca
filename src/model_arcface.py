"""
ArcFace損失を用いた選手再識別モデル
手動PyTorch訓練ループで既存Runnerと完全互換
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.dataset_image import BasketballDataset


class ArcFaceHead(nn.Module):
    """
    ArcFace損失関数のヘッド
    
    参考文献: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698
    
    主な特徴:
    - 特徴量とクラス重みを単位球面上に正規化
    - コサイン類似度ベースの分類
    - 正解クラスに角度マージン(m)を追加
    - スケールファクター(s)で勾配調整
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5):
        """
        Args:
            in_features: 入力埋め込み次元数
            out_features: クラス数
            s: スケールファクター（デフォルト: 30.0）
            m: 角度マージン（デフォルト: 0.5 rad ≈ 28.6度）
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # クラス重み（学習可能パラメータ）
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 角度マージン計算用の定数
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力埋め込み [batch_size, in_features]
            labels: ターゲットラベル [batch_size] (推論時はNone)
            
        Returns:
            訓練時: ArcFaceロジット [batch_size, out_features]
            推論時: スケール済みコサイン類似度 [batch_size, out_features]
        """
        # 特徴量と重みを正規化（単位球面上に配置）
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # コサイン類似度計算
        cosine = F.linear(x_norm, w_norm)

        if labels is None:
            # 推論モード: スケール済みコサイン類似度を返す
            return cosine * self.s

        # 訓練モード: 角度マージンを適用
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        # 閾値処理
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # ワンホットエンコーディング
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # ターゲットクラスにのみマージンを適用
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class PlayerEmbeddingModel(nn.Module):
    """
    選手再識別のための埋め込みモデル
    
    アーキテクチャ:
        入力画像 → Backbone (CNN) → 埋め込み層 → ArcFaceヘッド
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        embedding_dim: int = 512,
        num_classes: int = 11,
        pretrained: bool = True,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
    ):
        """
        Args:
            model_name: バックボーンモデル名（timmライブラリ）
            embedding_dim: 埋め込みベクトル次元数
            num_classes: 訓練データのクラス数
            pretrained: ImageNet事前訓練重みを使用するか
            arcface_s: ArcFaceスケールパラメータ
            arcface_m: ArcFace角度マージン
        """
        super().__init__()

        # バックボーン（分類器なし）
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 分類器を削除
        )

        # バックボーンの出力特徴量数
        backbone_out = self.backbone.num_features

        # 埋め込み層（BN-Dropout-FC-BN構成）
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(backbone_out),
            nn.Dropout(0.3),
            nn.Linear(backbone_out, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        # 訓練用ArcFaceヘッド
        self.arcface = ArcFaceHead(
            in_features=embedding_dim,
            out_features=num_classes,
            s=arcface_s,
            m=arcface_m,
        )

        self.embedding_dim = embedding_dim

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """正規化された埋め込みベクトルを抽出"""
        features = self.backbone(x)
        embedding = self.embedding(features)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力画像 [batch_size, 3, H, W]
            labels: ターゲットラベル [batch_size] (推論時はNone)
            
        Returns:
            訓練時: ArcFaceロジット [batch_size, num_classes]
            推論時: 正規化埋め込み [batch_size, embedding_dim]
        """
        features = self.backbone(x)
        embedding = self.embedding(features)

        if labels is not None:
            # 訓練時: ArcFaceロジットを返す
            return self.arcface(embedding, labels)
        else:
            # 推論時: 正規化埋め込みを返す
            return F.normalize(embedding, p=2, dim=1)


class EMAHelper:
    """
    Exponential Moving Average (EMA) ヘルパー
    
    訓練中のモデル重みの移動平均を保持することで、
    より安定した汎化性能を得る
    """

    def __init__(self, model: nn.Module, decay: float = 0.995):
        """
        Args:
            model: 追跡するモデル
            decay: 減衰率（0.995 ~ 0.9998が一般的）
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初期重みをコピー
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """EMA重みを更新（各最適化ステップ後に呼ぶ）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        """モデルの重みをEMA重みに置き換え（推論前）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """モデルの重みを元に戻す（推論後）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class ModelArcFace(Model):
    """
    ArcFace損失を用いた選手再識別モデル
    既存Runnerと完全互換の実装
    
    【学習フェーズ】
    1. PyTorch標準の訓練ループでArcFace損失を最小化
    2. EMAで重みを平滑化
    3. 訓練データ全体からクラスプロトタイプ（平均埋め込み）を計算
    
    【推論フェーズ】
    1. テスト画像から埋め込みを抽出
    2. プロトタイプとのコサイン類似度を計算
    3. 最も類似度が高いクラスを予測
    4. 閾値判定でunknown（-1）を予測
    """

    def __init__(self, run_fold_name: str, params: dict, out_dir_name: str, logger) -> None:
        super().__init__(run_fold_name, params, out_dir_name, logger)
        
        # パラメータ
        self.model_name = params.get('model_name', 'efficientnet_b0')
        self.embedding_dim = params.get('embedding_dim', 512)
        self.img_size = params.get('img_size', 224)
        self.batch_size = params.get('batch_size', 64)
        self.epochs = params.get('epochs', 20)
        self.lr = params.get('lr', 1e-3)
        self.weight_decay = params.get('weight_decay', 1e-4)
        self.arcface_s = params.get('arcface_s', 30.0)
        self.arcface_m = params.get('arcface_m', 0.5)
        self.use_ema = params.get('use_ema', True)
        self.ema_decay = params.get('ema_decay', 0.995)
        self.threshold = params.get('threshold', 0.5)
        self.num_workers = params.get('num_workers', 4)
        
        # デバイス
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデル保存先
        self.model_dir = os.path.join(out_dir_name, run_fold_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初期化（trainで作成）
        self.model = None
        self.ema = None
        self.prototypes = None
        self.num_classes = None

    def _get_transforms(self, train: bool = True) -> A.Compose:
        """データ拡張の定義"""
        if train:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def _create_dataloader(self, df: pd.DataFrame, split: bool = 'train') -> torch.utils.data.DataLoader:
        """DataLoaderの作成"""

        # dataset
        treatment_is_train = split in ['train', 'valid']
        dataset = BasketballDataset(
            df=df,
            transform=self._get_transforms(train=treatment_is_train),
            is_train=treatment_is_train
        )
        
        # dataloader
        treatment_is_train = split in ['train']
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=treatment_is_train,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=treatment_is_train,
        )

    def train(self, tr: pd.DataFrame, va: Optional[pd.DataFrame] = None) -> None:
        """
        モデルの訓練
        
        Args:
            tr: 訓練データ
            va: 検証データ（オプション）
        """
        self.logger.info(f"ArcFace訓練開始: {len(tr)}サンプル")
        self.logger.info(f"  モデル: {self.model_name}, 埋め込み次元: {self.embedding_dim}")
        self.logger.info(f"  バッチサイズ: {self.batch_size}, エポック: {self.epochs}")
        self.logger.info(f"  ArcFace (s={self.arcface_s}, m={self.arcface_m})")
        
        # クラス数を取得
        self.num_classes = tr['label_id'].nunique()
        self.logger.info(f"  クラス数: {self.num_classes}")
        
        # モデル作成
        self.model = PlayerEmbeddingModel(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            pretrained=True,
            arcface_s=self.arcface_s,
            arcface_m=self.arcface_m,
        ).to(self.device)
        
        # EMA初期化
        if self.use_ema:
            self.ema = EMAHelper(self.model, decay=self.ema_decay)
            self.logger.info(f"  EMA有効 (decay={self.ema_decay})")
        
        # オプティマイザ・スケジューラ
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        
        # DataLoader
        train_loader = self._create_dataloader(tr, split='train')
        val_loader = self._create_dataloader(va, split='valid') if va is not None else None
        
        # 訓練ループ
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            # 訓練
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                images, labels = batch  # BasketballDatasetは(image, label)のタプルを返す
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 順伝播
                optimizer.zero_grad()
                outputs = self.model(images, labels)
                loss = criterion(outputs, labels)
                
                # 逆伝播
                loss.backward()
                optimizer.step()
                
                # EMA更新
                if self.use_ema:
                    self.ema.update(self.model)
                
                # 統計
                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
            
            train_loss /= train_total
            train_acc = 100. * train_correct / train_total
            
            # 検証
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                self.logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                               f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
                
                # ベストモデル保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model()
            else:
                self.logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%")
            
            scheduler.step()
        
        # 最終モデル保存（検証データがない場合）
        if val_loader is None:
            self.save_model()
        
        # プロトタイプ計算
        self.logger.info("クラスプロトタイプを計算中...")
        self._compute_prototypes(tr)
        self.logger.info("訓練完了")

    def _validate(self, val_loader, criterion) -> Tuple[float, float]:
        """検証"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # EMA重みを適用
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, labels)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 元の重みに戻す
        if self.use_ema:
            self.ema.restore(self.model)
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        return val_loss, val_acc

    def _compute_prototypes(self, df: pd.DataFrame):
        """クラスプロトタイプ（平均埋め込み）を計算"""
        self.model.eval()
        
        # EMA重みを適用
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        
        # 全データの埋め込みを抽出
        dataloader = self._create_dataloader(df, split='train')
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="埋め込み抽出"):
                images, labels = batch
                images = images.to(self.device)
                
                embeddings = self.model.get_embedding(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # クラスごとの平均を計算
        self.prototypes = torch.zeros(self.num_classes, self.embedding_dim)
        for c in range(self.num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                class_embeddings = all_embeddings[mask]
                prototype = class_embeddings.mean(dim=0)
                self.prototypes[c] = F.normalize(prototype, p=2, dim=0)
        
        self.logger.info(f"プロトタイプ形状: {self.prototypes.shape}")
        
        # 元の重みに戻す
        if self.use_ema:
            self.ema.restore(self.model)

    def predict(self, te: pd.DataFrame, split='test') -> np.ndarray:
        """
        予測
        
        Args:
            te: テストデータ
            
        Returns:
            予測ラベル（unknown は -1）
        """
        self.logger.info(f"ArcFace予測開始: {len(te)}サンプル")
        
        self.model.eval()
        
        # EMA重みを適用
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        
        # テストデータの埋め込みを抽出
        dataloader = self._create_dataloader(te, split=split)
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="推論中"):
                # trainがTrueの場合はタプル、Falseの場合は画像のみ
                if split != 'test':
                    images, _ = batch
                else:
                    images = batch
                images = images.to(self.device)
                embeddings = self.model.get_embedding(images)
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
                
        # プロトタイプとの類似度で予測
        similarities = F.linear(all_embeddings, self.prototypes)  # [N, num_classes]
        max_sims, max_indices = similarities.max(dim=1)
        
        # 閾値判定
        predictions = []
        for sim, idx in zip(max_sims.tolist(), max_indices.tolist()):
            if sim < self.threshold:
                predictions.append(-1)  # unknown
            else:
                predictions.append(idx)
        
        predictions = np.array(predictions)
        
        # 統計情報
        unique, counts = np.unique(predictions, return_counts=True)
        self.logger.info(f"予測分布: {dict(zip(unique, counts))}")
        
        # 元の重みに戻す
        if self.use_ema:
            self.ema.restore(self.model)
        
        # 元のDataFrameのインデックスを保持
        result = pd.DataFrame({'label_id': predictions}, index=te.index)
        return result

    def save_model(self) -> None:
        """モデルの保存"""
        model_path = os.path.join(self.model_dir, 'model.pth')
        proto_path = os.path.join(self.model_dir, 'prototypes.pth')
        
        # モデル保存
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow if self.use_ema else None,
            'num_classes': self.num_classes,
        }, model_path)
        
        # プロトタイプ保存（計算済みの場合）
        if self.prototypes is not None:
            torch.save(self.prototypes, proto_path)
        
        self.logger.info(f"モデル保存: {model_path}")

    def load_model(self) -> None:
        """モデルの読み込み"""
        model_path = os.path.join(self.model_dir, 'model.pth')
        proto_path = os.path.join(self.model_dir, 'prototypes.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
        
        # チェックポイント読み込み
        checkpoint = torch.load(model_path, map_location=self.device)
        self.num_classes = checkpoint['num_classes']
        
        # モデル作成
        self.model = PlayerEmbeddingModel(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            pretrained=False,
            arcface_s=self.arcface_s,
            arcface_m=self.arcface_m,
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # EMA復元
        if self.use_ema and checkpoint['ema_shadow'] is not None:
            self.ema = EMAHelper(self.model, decay=self.ema_decay)
            self.ema.shadow = checkpoint['ema_shadow']
        
        # プロトタイプ読み込み
        if os.path.exists(proto_path):
            self.prototypes = torch.load(proto_path, map_location='cpu')
        
        self.logger.info(f"モデル読み込み: {model_path}")
