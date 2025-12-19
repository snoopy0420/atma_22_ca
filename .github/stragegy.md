# atmaCup#22 戦略リスト

バスケットボール選手識別コンペティションの戦略を網羅的にリストアップします。

## 目次
1. [モデルアーキテクチャ](#1-モデルアーキテクチャ)
2. [損失関数・学習手法](#2-損失関数学習手法)
3. [データ拡張](#3-データ拡張)
4. [CV戦略](#4-cv戦略)
5. [後処理](#5-後処理)
6. [アンサンブル](#6-アンサンブル)
7. [特徴量エンジニアリング](#7-特徴量エンジニアリング)
8. [ドメイン知識の活用](#8-ドメイン知識の活用)
9. [推論時の工夫](#9-推論時の工夫)
10. [その他のテクニック](#10-その他のテクニック)

---

## 1. モデルアーキテクチャ

### 1.1 バックボーンモデル
- [ ] **EfficientNet系**: B0～B7、V2
- [ ] **ResNet系**: ResNet50, ResNet101, ResNeXt
- [ ] **ConvNeXt**: 最新のCNNアーキテクチャ
- [ ] **Swin Transformer**: Vision Transformer系
- [ ] **ViT (Vision Transformer)**: 大規模事前学習モデル
- [ ] **RegNet**: 効率的なCNN
- [ ] **MobileNet**: 軽量モデル（高速推論用）
- [ ] **DenseNet**: 密結合CNN
- [ ] **NFNet**: 正規化なしで訓練可能

### 1.2 埋め込み学習手法
- [x] **ArcFace**: 角度マージンによる分離（実装済み）
- [ ] **CosFace**: コサイン類似度ベース
- [ ] **SphereFace**: 球面上での分類
- [ ] **AdaCos**: 適応的スケーリング
- [ ] **CircleLoss**: 柔軟なマージン設計
- [ ] **TripletLoss**: 三つ組損失
- [ ] **ContrastiveLoss**: 対照学習
- [ ] **SupCon**: 教師ありContrastive Learning

### 1.3 Attention機構
- [ ] **SE (Squeeze-and-Excitation)**: チャネルAttention
- [ ] **CBAM**: チャネル+空間Attention
- [ ] **ECA**: 効率的チャネルAttention
- [ ] **Non-local**: 自己注意機構
- [ ] **Spatial Attention**: 空間的な重要度

### 1.4 マルチタスク学習
- [ ] **補助タスク追加**: 角度予測（side/top）
- [ ] **属性予測**: ユニフォーム色、番号（もし識別可能なら）
- [ ] **位置情報予測**: bbox座標の回帰

---

## 2. 損失関数・学習手法

### 2.1 損失関数の工夫
- [x] **ArcFace損失**: 実装済み
- [ ] **Focal Loss**: クラス不均衡対応
- [ ] **Label Smoothing**: 過学習防止
- [ ] **Class Balanced Loss**: サンプル数に応じた重み付け
- [ ] **複数損失の組み合わせ**: CrossEntropy + Triplet
- [ ] **Unknown用の損失**: -1クラスの特別扱い

### 2.2 学習率スケジューリング
- [x] **Cosine Annealing**: 実装済み
- [ ] **Warm-up + Cosine**: 初期学習率を徐々に上げる
- [ ] **OneCycleLR**: 1サイクル学習率スケジューリング
- [ ] **ReduceLROnPlateau**: 検証スコアに応じて調整
- [ ] **Exponential Decay**: 指数減衰

### 2.3 正則化
- [x] **Dropout**: 実装済み
- [x] **Weight Decay (L2)**: 実装済み
- [ ] **Stochastic Depth**: レイヤーをランダムにスキップ
- [ ] **MixUp**: 画像と教師ラベルを混合
- [ ] **CutMix**: 画像の一部を切り貼り
- [ ] **Cutout**: ランダムマスキング
- [ ] **GridMask**: グリッド状マスキング
- [ ] **SAM (Sharpness Aware Minimization)**: 損失の平坦性を改善

### 2.4 最適化
- [x] **AdamW**: 実装済み
- [ ] **SGD + Momentum**: 古典的だが効果的
- [ ] **RAdam**: Rectified Adam
- [ ] **Lookahead**: Lookahead optimizer
- [ ] **Ranger**: RAdam + Lookahead
- [ ] **LAMB**: Layer-wise Adaptive Moments

### 2.5 モデル平均化
- [x] **EMA (Exponential Moving Average)**: 実装済み
- [ ] **SWA (Stochastic Weight Averaging)**: 複数エポックの重み平均
- [ ] **SWALR**: SWA + Cyclic LR

---

## 3. データ拡張

### 3.1 幾何学的変換
- [x] **HorizontalFlip**: 実装済み
- [x] **ShiftScaleRotate**: 実装済み
- [ ] **VerticalFlip**: 垂直反転
- [ ] **RandomRotate90**: 90度回転
- [ ] **Perspective**: 射影変換
- [ ] **ElasticTransform**: 弾性変形
- [ ] **GridDistortion**: グリッド歪み
- [ ] **OpticalDistortion**: 光学歪み
- [ ] **RandomCrop**: ランダムクロップ
- [ ] **CenterCrop**: 中心クロップ

### 3.2 色変換
- [x] **ColorJitter**: 実装済み
- [ ] **RandomBrightnessContrast**: 明度・コントラスト
- [ ] **HueSaturationValue**: 色相・彩度・明度
- [ ] **RGBShift**: RGB値のシフト
- [ ] **ChannelShuffle**: チャネル順序シャッフル
- [ ] **RandomGamma**: ガンマ補正
- [ ] **CLAHE**: 適応的ヒストグラム均等化
- [ ] **ToGray**: グレースケール化（一部確率で）
- [ ] **Invert**: 色反転

### 3.3 ノイズ・ぼかし
- [ ] **GaussNoise**: ガウシアンノイズ
- [ ] **ISONoise**: ISOノイズ
- [ ] **GaussianBlur**: ガウシアンぼかし
- [ ] **MotionBlur**: モーションブラー
- [ ] **MedianBlur**: メディアンぼかし
- [ ] **Blur**: 一般的なぼかし

### 3.4 その他の拡張
- [ ] **CoarseDropout**: 粗いドロップアウト
- [ ] **RandomShadow**: 影の追加
- [ ] **RandomRain**: 雨の効果
- [ ] **RandomFog**: 霧の効果
- [ ] **RandomSnow**: 雪の効果
- [ ] **RandomSunFlare**: レンズフレア

### 3.5 テスト時拡張（TTA）
- [ ] **HorizontalFlip TTA**: 左右反転した画像も予測
- [ ] **MultiScale TTA**: 複数スケールで予測
- [ ] **Rotation TTA**: 複数角度で予測
- [ ] **Crop TTA**: 複数クロップで予測

---

## 4. CV戦略

### 4.1 基本的なCV
- [x] **Group K-Fold**: quarter_sessionでグループ化（実装済み）
- [ ] **Stratified K-Fold**: 選手分布を保持
- [ ] **Time Series Split**: 時系列を考慮した分割

### 4.2 CV改善
- [ ] **5-Fold → 10-Fold**: より細かい分割
- [ ] **Leave-One-Group-Out**: 1つのquarterを完全に除外
- [ ] **Double Validation**: 2段階の検証

### 4.3 リーク防止
- [x] **同一シーンの分離**: 実装済み
- [ ] **同一フレーム付近の分離**: 近いフレームを同じFoldに
- [ ] **同一選手の連続フレーム**: 時系列考慮

### 4.4 Pseudo Labeling
- [ ] **テストデータへの擬似ラベル付与**: 高信頼度予測を訓練に追加
- [ ] **閾値調整**: 信頼度スコアで選別
- [ ] **反復学習**: 複数回の擬似ラベル更新

---

## 5. 後処理

### 5.1 閾値チューニング
- [x] **Unknown判定閾値**: 実装済み（threshold=0.5）
- [ ] **CVで最適閾値を探索**: グリッドサーチ
- [ ] **選手ごとの閾値**: 各選手で異なる閾値
- [ ] **動的閾値**: 予測分布に応じて調整

### 5.2 時系列スムージング
- [ ] **移動平均フィルタ**: 連続フレームで平滑化
- [ ] **メディアンフィルタ**: 外れ値を除去
- [ ] **ガウシアンフィルタ**: 重み付き平均
- [ ] **Kalmanフィルタ**: 状態推定

### 5.3 選手交代の検出
- [ ] **予測の急変検知**: 同一bboxで選手IDが変わる
- [ ] **確信度の低下**: 類似度が低い区間を検出
- [ ] **ルールベース**: quarterやtimeに基づく

### 5.4 空間的整合性
- [ ] **同一フレーム内の重複除去**: 同じ選手が複数予測されないように
- [ ] **BBox IoU考慮**: 重なりが大きいbboxは同一人物の可能性
- [ ] **位置情報の利用**: 前後フレームでの移動量を考慮

### 5.5 予測の補正
- [ ] **多数決**: 複数モデルで多数決
- [ ] **確信度加重平均**: 類似度スコアで重み付け
- [ ] **上位K個の平均**: Top-K予測の平均

---

## 6. アンサンブル

### 6.1 単純アンサンブル
- [ ] **多数決**: 複数モデルの多数決
- [ ] **平均**: 予測確率の平均
- [ ] **重み付き平均**: CVスコアで重み付け

### 6.2 高度なアンサンブル
- [ ] **Stacking**: 2段目モデルで予測
- [ ] **Blending**: ホールドアウトで2段目学習
- [ ] **Rank Averaging**: 順位の平均

### 6.3 多様性の確保
- [ ] **異なるバックボーン**: EfficientNet + ResNet + ViT
- [ ] **異なる損失関数**: ArcFace + CosFace + Triplet
- [ ] **異なるCV分割**: 複数のシード値
- [ ] **異なる画像サイズ**: 224x224 + 384x384
- [ ] **異なるデータ拡張**: 強・弱の拡張

---

## 7. 特徴量エンジニアリング

### 7.1 画像特徴
- [x] **CNN埋め込み**: 実装済み
- [ ] **複数層の特徴結合**: 中間層 + 最終層
- [ ] **Multi-Scale特徴**: 異なるスケールで抽出
- [ ] **局所特徴**: SIFT, ORB, HOG

### 7.2 メタ特徴
- [ ] **BBoxサイズ**: 幅・高さ・面積
- [ ] **BBox位置**: 画像内の相対座標
- [ ] **BBoxアスペクト比**: 幅/高さ
- [ ] **角度情報**: side/topのone-hot
- [ ] **Quarter/Session/Frame**: 時系列情報
- [ ] **選手の出現頻度**: 統計情報

### 7.3 統計特徴
- [ ] **色ヒストグラム**: RGB分布
- [ ] **テクスチャ特徴**: Gabor, LBP
- [ ] **エッジ特徴**: Cannyエッジ

### 7.4 時系列特徴
- [ ] **前後フレームの差分**: モーション情報
- [ ] **軌跡情報**: bbox中心の移動経路
- [ ] **速度・加速度**: 移動量から計算

---

## 8. ドメイン知識の活用

### 8.1 バスケットボールのルール
- [ ] **選手交代タイミング**: quarterの切り替わり
- [ ] **コート内の人数制限**: 各チーム5人
- [ ] **ポジション情報**: ガード、フォワード、センター

### 8.2 カメラ・撮影環境
- [ ] **画角の違い**: side/topで見え方が異なる
- [ ] **照明条件**: 明るさの変動
- [ ] **オクルージョン**: 選手同士の重なり

### 8.3 選手の特徴
- [ ] **体格差**: 身長・体型の違い
- [ ] **ユニフォーム**: 色・番号（もし見える場合）
- [ ] **プレースタイル**: 動き方の特徴

### 8.4 Unknown判定の基準
- [ ] **新規登場選手**: 訓練データにいない選手
- [ ] **BBoxの不正確さ**: 選手の一部だけ
- [ ] **低品質画像**: ぼけ・ノイズ

---

## 9. 推論時の工夫

### 9.1 高速化
- [ ] **Mixed Precision (FP16)**: GPU高速化
- [ ] **TensorRT**: 推論エンジン最適化
- [ ] **ONNX変換**: 軽量化
- [ ] **バッチサイズ最適化**: スループット向上
- [ ] **モデル蒸留**: 小型モデルに知識転移

### 9.2 精度向上
- [x] **EMA重み使用**: 実装済み
- [ ] **Test Time Augmentation (TTA)**: 複数拡張で予測
- [ ] **Multi-Crop**: 複数クロップで予測
- [ ] **Multi-Scale**: 複数解像度で予測

### 9.3 Unknown判定の改善
- [ ] **アウトライア検出**: マハラノビス距離
- [ ] **信頼度スコア**: エントロピーベース
- [ ] **OOD (Out-of-Distribution)検出**: 分布外サンプル検出

---

## 10. その他のテクニック

### 10.1 事前学習
- [x] **ImageNet事前学習**: 実装済み
- [ ] **別のPerson Re-IDデータセット**: Market-1501, DukeMTMC
- [ ] **スポーツデータセット**: スポーツ選手の画像
- [ ] **自己教師あり学習**: SimCLR, MoCo

### 10.2 データの前処理
- [x] **事前クロップ**: 実装済み
- [ ] **パディング調整**: 10% → 5%, 15%で実験
- [ ] **アスペクト比維持**: リサイズ方法の変更
- [ ] **Super Resolution**: 低解像度画像の高解像度化

### 10.3 Hard Example Mining
- [ ] **誤分類サンプルの重視**: 難しいサンプルを重点的に学習
- [ ] **類似選手のペア学習**: 見分けにくい選手を集中訓練
- [ ] **Online Hard Example Mining**: 学習中に動的に選択

### 10.4 マルチモーダル
- [ ] **複数画角の統合**: side + topの両方を使用
- [ ] **時系列情報の統合**: 前後フレームを入力
- [ ] **LSTM/Transformer**: 時系列モデルで追跡

### 10.5 外部データ
- [ ] **類似コンペのデータ**: Person Re-IDコンペ
- [ ] **合成データ**: Data Augmentationで生成
- [ ] **Webスクレイピング**: バスケ選手の画像（注意: 規約確認）

### 10.6 デバッグ・分析
- [ ] **Grad-CAM**: モデルの注目領域可視化
- [ ] **Confusion Matrix**: 誤分類パターン分析
- [ ] **Error Analysis**: 失敗ケースの詳細分析
- [ ] **Embedding可視化**: t-SNE, UMAP

### 10.7 実装最適化
- [ ] **num_workers調整**: DataLoader並列化
- [ ] **pin_memory**: GPU転送高速化
- [ ] **persistent_workers**: ワーカー再利用
- [ ] **prefetch_factor**: プリフェッチ数調整
- [ ] **AMP (Automatic Mixed Precision)**: 自動混合精度

---

## 実装優先度

### 🔴 高優先度（すぐに試す価値あり）
1. 異なるバックボーン（ResNet50, ConvNeXt）
2. TTA（HorizontalFlip, MultiScale）
3. 閾値の最適化（CVでグリッドサーチ）
4. MixUp/CutMix
5. 時系列スムージング（移動平均）
6. 複数モデルのアンサンブル

### 🟡 中優先度（時間があれば試す）
7. CosFace/CircleLoss
8. データ拡張の追加
9. Pseudo Labeling
10. Stacking/Blending
11. メタ特徴の追加
12. SAM optimizer

### 🟢 低優先度（余裕があれば試す）
13. Attention機構の追加
14. マルチタスク学習
15. 外部データの利用
16. モデル蒸留
17. LSTM/Transformerでの時系列追跡

---

## 実験管理

### 実験記録フォーマット
```markdown
## 実験 XX: [実験名]
- **日付**: YYYY-MM-DD
- **変更点**: 
  - 箇条書きで記載
- **結果**: 
  - CV: X.XXXX
  - LB: X.XXXX (提出した場合)
- **考察**: 
  - なぜこの結果になったか
  - 次に試すこと
```

### 優先的に追跡すべき指標
- CV Macro F1
- Unknown (-1) の精度
- 選手ごとのF1スコア
- 訓練時間
- 推論速度

---

## 参考リソース

### Papers
- ArcFace: https://arxiv.org/abs/1801.07698
- CosFace: https://arxiv.org/abs/1801.09414
- CircleLoss: https://arxiv.org/abs/2002.10857
- EfficientNet: https://arxiv.org/abs/1905.11946
- ConvNeXt: https://arxiv.org/abs/2201.03545

### Competitions
- Kaggle Person Re-Identification
- Google Landmark Recognition
- Shopee Product Matching

### Libraries
- timm: PyTorch Image Models
- albumentations: Data Augmentation
- pytorch-metric-learning: Metric Learning

---

**更新日**: 2025-12-19
**ステータス**: 継続的に更新中



