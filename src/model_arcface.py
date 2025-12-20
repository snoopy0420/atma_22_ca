"""
ArcFace損失を用いた選手再識別モデル
RedBullのPyTorch Lightningコードをそのまま使用
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import F1Score


sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.dataset_image import _create_dataloader
from src.util import seed_everything



# ============================================================================
# ArcFace Model Components
# ============================================================================

class ArcFaceHead(nn.Module):
    """
    ArcFace損失関数のヘッド部分
    
    アーキテクチャ:
    - 重み行列 W: [num_classes, embedding_dim]
    - 入力埋め込みとWを正規化してcos類似度を計算
    - 訓練時: 正解クラスの角度にマージンmを追加してcos類似度を減少させる
    - 推論時: 単純なcos類似度にスケールsを掛ける
    
    数式:
    - cos(θ) = W^T * x / (||W|| * ||x||)  # 正規化された内積
    - φ = cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)  # 角度マージン追加
    - output = s * φ (正解クラス) or s * cos(θ) (その他)
    
    Args:
        in_features: 入力埋め込み次元（例: 512）
        out_features: 出力クラス数（選手数、例: 11）
        s: スケールパラメータ（類似度のスケーリング、通常30.0）
        m: マージンパラメータ（角度マージン、通常0.5 rad ≈ 28.6度）
        easy_margin: 簡単なマージン適用方法を使用するか
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # スケールパラメータ（logitのスケーリング）
        self.m = m  # 角度マージン（rad単位）
        self.easy_margin = easy_margin

        # クラスごとの重みベクトル（正規化して使用）
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 角度マージンの事前計算（cos/sin加法定理用）
        self.cos_m = math.cos(m)  # cos(m)
        self.sin_m = math.sin(m)  # sin(m)
        self.th = math.cos(math.pi - m)  # 閾値: cos(π-m)
        self.mm = math.sin(math.pi - m) * m  # マージン補正項

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        順伝播処理
        
        処理フロー:
        1. 入力xと重みWをL2正規化
        2. cos類似度を計算（正規化された内積）
        3. 訓練時: 正解クラスに角度マージンm追加
        4. スケールsを掛けてlogitとして出力
        
        Args:
            x: 埋め込みベクトル [batch_size, embedding_dim]
            labels: 正解ラベル [batch_size] (訓練時のみ)
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # 1. 正規化（埋め込みと重みを単位ベクトル化）
        x_norm = F.normalize(x, p=2, dim=1)  # [B, D] -> 各行をL2正規化
        w_norm = F.normalize(self.weight, p=2, dim=1)  # [C, D] -> 各行をL2正規化
        
        # 2. cos類似度計算（正規化された内積）
        cosine = F.linear(x_norm, w_norm)  # [B, C]: cos(θ_i,j)

        # 推論時: スケールのみ適用して返す
        if labels is None:
            return cosine * self.s

        # 3. 訓練時: 正解クラスに角度マージンを追加
        # sin(θ) = sqrt(1 - cos^2(θ))
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        # φ = cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # マージン適用の条件分岐
        if self.easy_margin:
            # cos(θ) > 0 の場合のみマージン適用
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # cos(θ) > cos(π-m) の場合にマージン適用、それ以外は線形補正
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4. 正解クラスのみφを使い、その他はcosineを使う
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)  # 正解位置に1
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # スケーリング
        return output


class PlayerModule(pl.LightningModule):
    """
    PyTorch Lightning統合版の訓練モジュール
    
    機能:
    - PlayerEmbeddingModelのラッパー
    - 訓練ループ（loss計算、最適化、ログ記録）
    - 検証ループ（F1スコア計算）
    - EMA（Exponential Moving Average）モデルの管理
    - 学習率スケジューリング（Cosine Annealing）
    
    EMA（指数移動平均）:
    - 訓練中のモデル重みの移動平均を保持
    - より安定した予測性能（推論時はEMAモデル使用）
    - 更新式: θ_ema = decay * θ_ema + (1-decay) * θ_current
    
    訓練プロセス:
    1. 画像 -> モデル -> ArcFace logits
    2. CrossEntropyLoss計算
    3. Backprop & 最適化
    4. EMAモデル更新
    5. F1スコア記録
    
    Args:
        model_name: Backboneモデル名
        num_classes: 選手クラス数
        pretrained: 事前学習済み重み使用
        embedding_dim: 埋め込み次元
        arcface_s/m: ArcFaceパラメータ
        lr: 学習率
        weight_decay: L2正則化係数
        epochs: エポック数
        ema_decay: EMA減衰率（0.9998推奨）
        use_ema: EMA使用フラグ
    """

    def __init__(self, model_name: str = "efficientnet_b0", num_classes: int = 11,
                 pretrained: bool = True, embedding_dim: int = 512,
                 arcface_s: float = 30.0, arcface_m: float = 0.5,
                 lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 20,
                 ema_decay: float = 0.9998, use_ema: bool = True):
        super().__init__()
        self.save_hyperparameters()  # チェックポイント保存用

        # Backbone: 事前学習済みCNN（最終分類層なし）
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone_out = self.backbone.num_features  # Backboneの出力次元

        # Embedding Layer: 特徴を低次元埋め込み空間に射影
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(backbone_out),  # バッチ正規化
            nn.Dropout(0.3),  # 過学習防止
            nn.Linear(backbone_out, embedding_dim, bias=False),  # 線形射影
            nn.BatchNorm1d(embedding_dim),  # 埋め込みの正規化
        )

        # ArcFace Head: 埋め込みをクラスlogitsに変換
        self.arcface = ArcFaceHead(in_features=embedding_dim, out_features=num_classes,
                                   s=arcface_s, m=arcface_m)
        self.embedding_dim = embedding_dim
        
        # 損失関数とメトリクス
        self.criterion = nn.CrossEntropyLoss()  # ArcFace logitsに対する交差エントロピー
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # EMA（Exponential Moving Average）設定
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.model_ema = None  # setup()で初期化

    def setup(self, stage: str = None):
        """訓練開始前にEMAモデルを初期化"""
        if self.use_ema and self.model_ema is None:
            self.model_ema = timm.utils.ModelEmaV3(self, decay=self.ema_decay, device=self.device)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        画像から正規化された埋め込みベクトルを取得
        
        処理: 画像 -> Backbone -> Embedding -> L2正規化
        用途: プロトタイプとのcos類似度計算に使用
        
        Args:
            x: 入力画像 [B, 3, H, W]
        Returns:
            正規化埋め込み [B, embedding_dim]、各ベクトルのL2ノルムは1
        """
        features = self.backbone(x)  # [B, backbone_dim]
        embedding = self.embedding(features)  # [B, embedding_dim]
        return F.normalize(embedding, p=2, dim=1)  # L2正規化

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        順伝播（訓練時と推論時で動作が異なる）
        
        訓練時（labels指定）: ArcFace損失計算用のlogitsを返す
        推論時（labels=None）: 正規化埋め込みを返す
        
        Args:
            x: 入力画像 [B, 3, H, W]
            labels: 正解ラベル [B]（訓練時のみ）
        
        Returns:
            訓練時: ArcFaceのlogits [B, num_classes]
            推論時: 正規化埋め込み [B, embedding_dim]
        """
        # Backbone特徴抽出
        features = self.backbone(x)
        
        # Embedding Layer: 特徴を低次元埋め込み空間に射影
        embedding = self.embedding(features)
        
        if labels is not None:
            # 訓練時: ArcFace損失用のlogits
            return self.arcface(embedding, labels)
        else:
            # 推論時: 正規化埋め込み
            return F.normalize(embedding, p=2, dim=1)
    
    def get_embedding_ema(self, x: torch.Tensor) -> torch.Tensor:
        """
        EMAモデルで埋め込み取得（推論時に使用）
        
        EMA有効時はより安定したEMAモデルで埋め込み抽出
        """
        if self.use_ema and self.model_ema is not None:
            return self.model_ema.module.get_embedding(x)
        return self.get_embedding(x)

    def forward_ema(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """EMAモデルの順伝播（検証時に使用）"""
        if self.model_ema is not None:
            return self.model_ema.module(x, labels)
        return self(x, labels)

    def on_before_zero_grad(self, optimizer):
        """
        勾配をゼロにする前にEMAモデルを更新
        
        各最適化ステップ後に自動的に呼ばれる
        θ_ema = decay * θ_ema + (1-decay) * θ_current
        """
        if self.use_ema and self.model_ema is not None:
            self.model_ema.update(self)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]
        # ArcFaceマージンのためにラベル付きで順伝播
        outputs = self(images, labels)
        loss = self.criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        self.train_f1(preds, labels)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]
        
        # 検証にはEMAモデルを使用（適切なマージンのためにラベル付き）
        if self.use_ema and self.model_ema is not None:
            outputs = self.forward_ema(images, labels)
        else:
            outputs = self(images, labels)
        
        loss = self.criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        
        self.val_f1(preds, labels)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


@torch.no_grad()
def compute_prototypes(model: PlayerModule, dataloader, num_classes: int, device) -> torch.Tensor:
    """
    各クラスのプロトタイプ（代表埋め込み）を計算
    
    プロトタイプベース分類:
    1. 訓練データの各クラスの埋め込みを抽出
    2. クラスごとに埋め込みの平均を計算
    3. 平均ベクトルをL2正規化してプロトタイプとする
    4. 推論時: テスト埋め込みと各プロトタイプのcos類似度で分類
    
    利点:
    - 新しい選手（クラス）を追加しやすい（Few-shot学習）
    - クラス不均衡に強い
    - cos類似度の閾値で「unknown」を判定可能
    
    Args:
        model: 訓練済みPlayerModule
        dataloader: 訓練データローダー
        num_classes: クラス数
        device: 計算デバイス
    
    Returns:
        prototypes: [num_classes, embedding_dim] 各クラスの正規化プロトタイプ
    """
    model.eval()
    # クラスごとの埋め込みリストを初期化
    class_embeddings = {i: [] for i in range(num_classes)}
    
    # 全訓練データの埋め込みを抽出
    for batch in tqdm(dataloader, desc="プロトタイプ計算"):
        images = batch["image"].to(device, non_blocking=True)  # 非同期転送で高速化
        labels = batch["label"]
        embeddings = model.get_embedding_ema(images)  # EMAモデル使用
        
        # クラスごとに埋め込みを蓄積
        for emb, label in zip(embeddings.cpu(), labels):
            class_embeddings[label.item()].append(emb)
    
    # クラスごとの平均埋め込み（プロトタイプ）を計算
    prototypes = torch.zeros(num_classes, embeddings.shape[1])
    for class_id in range(num_classes):
        if len(class_embeddings[class_id]) > 0:
            class_embs = torch.stack(class_embeddings[class_id])  # [N, D]
            prototype = class_embs.mean(dim=0)  # 平均: [D]
            prototypes[class_id] = F.normalize(prototype, p=2, dim=0)  # L2正規化
    return prototypes


class ModelArcFace(Model):
    """
    Runner統合版ArcFaceモデル
    
    全体フロー:
    1. 訓練: PlayerModule（PyTorch Lightning）でArcFace損失により学習
    2. プロトタイプ計算: 訓練データから各クラスの代表埋め込みを生成
    3. 推論: テスト画像の埋め込みとプロトタイプのcos類似度で分類
    4. 閾値判定: 類似度が閾値未満の場合は「unknown」(-1)と判定
    
    アーキテクチャの特徴:
    - Backbone: EfficientNet-B0（ImageNet事前学習済み）
    - Embedding: 512次元の正規化埋め込み空間
    - 損失: ArcFace（角度マージンによる識別性向上）
    - 正則化: BatchNorm + Dropout + EMA
    - 最適化: AdamW + Cosine Annealing LR
    """

    def __init__(self, run_fold_name: str, params: dict, out_dir_name: str, logger) -> None:
        """
        ArcFaceモデルの初期化
        """
        super().__init__(run_fold_name, params, out_dir_name, logger)
        
        # シード固定（再現性確保）
        seed = params.get('seed', 42)
        seed_everything(seed)
        self.logger.info(f"シード固定: {seed}")
        
        # モデルアーキテクチャ設定
        self.model_name = params.get('model_name', 'efficientnet_b0')  # Backboneモデル
        self.embedding_dim = params.get('embedding_dim', 512)  # 埋め込み次元
        self.img_size = params.get('img_size', 224)  # 入力画像サイズ
        # 訓練設定
        self.batch_size = params.get('batch_size', 64)
        self.epochs = params.get('epochs', 20)
        self.lr = params.get('lr', 1e-3)  # 学習率
        self.weight_decay = params.get('weight_decay', 1e-4)  # L2正則化
        # ArcFace損失パラメータ
        self.arcface_s = params.get('arcface_s', 30.0)  # スケール（logitの増幅）
        self.arcface_m = params.get('arcface_m', 0.5)  # 角度マージン（rad）
        # EMA（指数移動平均）設定
        self.use_ema = params.get('use_ema', True)
        self.ema_decay = params.get('ema_decay', 0.995)  # 減衰率（RedBull推奨値: 短期訓練に適切）
        # 推論設定
        self.threshold = params.get('threshold', 0.5)  # cos類似度閾値（unknown判定用）
        self.num_workers = params.get('num_workers', 8)  # DataLoaderワーカー数（GPU待ち時間を削減）
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # モデル保存ディレクトリ
        self.model_dir = os.path.join(out_dir_name, run_fold_name)
        os.makedirs(self.model_dir, exist_ok=True)
        # 訓練/推論時に使用する変数（初期化時はNone）
        self.pl_module = None  # PyTorch Lightningモジュール
        self.prototypes = None  # 各クラスの代表埋め込み [num_classes, embedding_dim]
        self.num_classes = None  # 選手クラス数


    def train(self, tr: pd.DataFrame, va: Optional[pd.DataFrame] = None) -> None:
        """
        モデル訓練の実行
        
        訓練フロー:
        1. ハイパーパラメータをログ出力
        2. PlayerModule（PyTorch Lightning）を初期化
        3. DataLoaderを作成
        4. ModelCheckpointコールバック設定（val_loss最小のモデルを保存）
        5. PyTorch Lightning Trainerで訓練実行
        6. ベストモデルをロード
        7. 訓練データからプロトタイプ（各クラスの代表埋め込み）を計算・保存
        
        Args:
            tr: 訓練データフレーム（画像パス、label_idを含む）
            va: 検証データフレーム（オプション）
        """
        # 訓練設定ログ
        self.logger.info(f"ArcFace訓練開始: {len(tr)}サンプル")
        self.logger.info(f"  モデル: {self.model_name}, 次元: {self.embedding_dim}")
        self.logger.info(f"  バッチ: {self.batch_size}, エポック: {self.epochs}")
        self.logger.info(f"  ArcFace (s={self.arcface_s}, m={self.arcface_m})")
        self.logger.info(f"  EMA: {self.use_ema} (decay={self.ema_decay})")
        self.num_classes = tr['label_id'].nunique()
        self.logger.info(f"  クラス数: {self.num_classes}")
        
        # PLモジュール
        self.pl_module = PlayerModule(
            model_name=self.model_name, 
            num_classes=self.num_classes, 
            pretrained=True,  # ImageNet事前学習済み重み使用
            embedding_dim=self.embedding_dim, 
            arcface_s=self.arcface_s, 
            arcface_m=self.arcface_m,
            lr=self.lr, 
            weight_decay=self.weight_decay, 
            epochs=self.epochs,
            ema_decay=self.ema_decay, 
            use_ema=self.use_ema
        )
        
        # コールバック
        callbacks = [
            ModelCheckpoint(
                dirpath=self.model_dir, 
                filename='best',
                monitor='val_f1',
                mode='max',  # 最大値を保存
                save_top_k=1,  # ベスト1つのみ保存
                verbose=True,
                enable_version_counter=False,
            )
        ]
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,  # GPU 1台使用（autoだとマルチGPU設定で遅延の可能性）
            callbacks=callbacks,
            enable_progress_bar=False,  # プログレスバー表示
            enable_model_summary=False,  # モデル概要表示
            logger=False,  # ロガー設定
            precision="16-mixed",  # 混合精度で高速化
            deterministic=True,  # 再現性確保（cudnnの決定性モード）
        )

        # DataLoader
        train_loader = _create_dataloader(tr, 'train', self.batch_size, self.num_workers, self.img_size)
        val_loader = _create_dataloader(va, 'valid', self.batch_size, self.num_workers, self.img_size) if va is not None else None
        
        # 訓練実行（自動でEpochループ、Backprop、最適化を実行）
        trainer.fit(self.pl_module, train_loader, val_loader)
        
        # ベストモデルをロード（val_lossが最小のエポックのモデル）
        best_model_path = os.path.join(self.model_dir, 'best.ckpt')
        if os.path.exists(best_model_path):
            self.pl_module = PlayerModule.load_from_checkpoint(
                best_model_path, 
                model_name=self.model_name, 
                num_classes=self.num_classes,
                embedding_dim=self.embedding_dim, 
                arcface_s=self.arcface_s, 
                arcface_m=self.arcface_m,
                strict=False
            )
            self.pl_module.to(self.device)
            self.logger.info(f"ベストモデルをロード: {best_model_path}")
        
        # プロトタイプ計算（各クラスの代表埋め込み）
        self.logger.info("プロトタイプ計算中...")
        self.prototypes = compute_prototypes(
            self.pl_module, 
            train_loader, 
            self.num_classes, 
            self.device
        )
        # プロトタイプを保存（推論時にロード）
        torch.save(self.prototypes, os.path.join(self.model_dir, 'prototypes.pth'))

        self.logger.info("訓練完了")

    def predict(self, te: pd.DataFrame, split='test') -> pd.DataFrame:
        """
        テストデータの予測実行
        
        推論フロー:
        1. モデルを評価モード（eval）に設定
        2. 各バッチの画像から埋め込みベクトルを抽出（EMAモデル使用）
        3. 全埋め込みとプロトタイプ間のcos類似度を計算
        4. 最大類似度のクラスを予測（閾値未満は-1=unknown）
        5. 予測分布をログ出力
        
        プロトタイプベース分類:
        - similarities[i, j] = cos(embedding_i, prototype_j)
        - 最大類似度が閾値以上 → そのクラスと予測
        - 最大類似度が閾値未満 → unknown (-1) と予測
        """
        self.logger.info(f"ArcFace予測: {len(te)}サンプル")
        
        # 元のインデックスを保存（OOF評価で正しい対応付けに必要）
        original_index = te.index.copy()
        
        # 評価モード（Dropout無効化、BatchNorm固定）
        self.pl_module.eval()
        
        # DataLoader
        dataloader = _create_dataloader(te, split, self.batch_size, self.num_workers, self.img_size)
        all_embeddings = []
        
        # 全データの埋め込みを抽出（勾配計算不要）
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="推論中"):

                images = batch['image'].to(self.device, non_blocking=True)  # 非同期転送
                
                # 埋め込み抽出（EMAモデルで安定した特徴量を取得）
                embeddings = self.pl_module.get_embedding_ema(images)  # [B, embedding_dim]
                all_embeddings.append(embeddings.cpu())
        
        # 全埋め込みを結合 [N, embedding_dim]
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # プロトタイプとのcos類似度を計算
        # similarities[i, j] = embedding_i · prototype_j （正規化済みなので内積=cos類似度）
        prototypes = self.prototypes.to(all_embeddings.device)  # [num_classes, embedding_dim]
        similarities = F.linear(all_embeddings, prototypes)  # [N, num_classes]
        
        # 各サンプルの最大類似度とそのクラスインデックスを取得
        max_sims, max_indices = similarities.max(dim=1)  # [N], [N]
        
        # 閾値判定: 類似度が閾値未満なら-1（unknown）
        predictions = []
        for sim, idx in zip(max_sims.tolist(), max_indices.tolist()):
            predictions.append(-1 if sim < self.threshold else idx)
        
        # DataFrameで返す（元のインデックスを使用してRunnerでの評価を正確に）
        return pd.DataFrame({'label_id': predictions}, index=original_index)

    def save_model(self) -> None:
        """
        モデルの保存
        
        注意:
        - PyTorch Lightningが自動で best.ckpt を保存するため、ここでは追加処理不要
        - プロトタイプも train() 内で保存済み
        - このメソッドはRunner仕様のインターフェース実装のみ
        """
        self.logger.info(f"モデル保存完了: {self.model_dir}")

    def load_model(self) -> None:
        """
        保存されたモデルとプロトタイプの読み込み
        
        読み込みファイル:
        1. best.ckpt: PyTorch Lightningチェックポイント（モデル重み、ハイパーパラメータ）
        2. prototypes.pth: 各クラスのプロトタイプ埋め込み
        
        用途:
        - CV予測時に各Foldのモデルをロード
        - 提出用予測時に全Foldのモデルをアンサンブル
        """
        # チェックポイントパスを構築
        best_model_path = os.path.join(self.model_dir, 'best.ckpt')
        proto_path = os.path.join(self.model_dir, 'prototypes.pth')
        
        # PyTorch Lightningチェックポイントからモデルをロード
        self.pl_module = PlayerModule.load_from_checkpoint(
            best_model_path,
            strict=False # strict=Falseで予期しないキー（EMA重み等）を無視
        )
        self.pl_module.to(self.device)
        self.num_classes = self.pl_module.hparams.num_classes
        
        # プロトタイプをロード（推論に必要）
        if os.path.exists(proto_path):
            self.prototypes = torch.load(proto_path, map_location='cpu')
        
        self.logger.info(f"モデル読み込み: {best_model_path}")
