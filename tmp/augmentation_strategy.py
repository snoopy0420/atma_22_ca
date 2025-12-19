"""
データ拡張戦略ガイド
バスケットボール選手再識別タスクに最適化されたaugmentation設定
"""

# ============================================================================
# 現状のaugmentation設定の分析
# ============================================================================

"""
現在の実装（RedBullベースライン）:

訓練時:
- HorizontalFlip (p=0.5)
- ShiftScaleRotate (shift=0.05, scale=0.05, rotate=10度, p=0.5)
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
- Resize (224x224)
- Normalize (ImageNet統計)

推論時:
- Resize (224x224)
- Normalize (ImageNet統計)
"""

# ============================================================================
# 問題点と改善提案
# ============================================================================

"""
【問題点1】HorizontalFlipの不適切性
- バスケットボールの選手番号は左右反転すると読めなくなる
- ユニフォームのロゴも反転すると識別困難
→ HorizontalFlipは削除すべき

【問題点2】回転角度が大きすぎる
- 10度の回転は選手の姿勢を大きく変えてしまう
- テストデータは正立した画像のみ
→ 回転は3-5度程度に抑えるべき

【問題点3】色変換が弱い
- 照明条件の変化に対応するには不十分
- コントラスト・明度の変動をもっと許容すべき
→ 範囲を拡大（brightness=0.3, contrast=0.3）

【問題点4】オクルージョン（隠蔽）対策がない
- 他の選手に重なる状況が多い
- CoarseDropoutやCutoutで頑健性向上
→ CoarseDropout追加を推奨

【問題点5】ぼかし・ノイズ対策がない
- 遠距離からの撮影でぼやけた画像がある
- GaussianBlurやGaussianNoiseで対応
→ 軽度のBlur/Noise追加を推奨
"""

# ============================================================================
# 推奨データ拡張設定（レベル別）
# ============================================================================

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation_light(img_size=224):
    """軽度のaugmentation（安全策・ベースライン改善版）
    
    変更点:
    - HorizontalFlipを削除
    - 回転を5度に制限
    - 色変換を強化
    """
    return A.Compose([
        # 幾何学変換（控えめ）
        A.ShiftScaleRotate(
            shift_limit=0.05,  # 5%シフト
            scale_limit=0.05,  # 5%スケール
            rotate_limit=5,    # ±5度回転（削減）
            border_mode=0,     # ゼロパディング
            p=0.5
        ),
        # 色変換（強化）
        A.ColorJitter(
            brightness=0.3,    # ±30%明度
            contrast=0.3,      # ±30%コントラスト
            saturation=0.2,    # ±20%彩度
            hue=0.1,           # ±10%色相
            p=0.6
        ),
        # リサイズと正規化
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_augmentation_medium(img_size=224):
    """中程度のaugmentation（推奨設定）
    
    追加要素:
    - 軽度のぼかし
    - CoarseDropout（オクルージョン対策）
    """
    return A.Compose([
        # 幾何学変換
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,   # スケールを10%に拡大
            rotate_limit=5,
            border_mode=0,
            p=0.5
        ),
        # 色変換
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
            p=0.6
        ),
        # ぼかし（軽度）
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        # オクルージョン対策
        A.CoarseDropout(
            max_holes=3,           # 最大3個の穴
            max_height=30,         # 高さ30px
            max_width=30,          # 幅30px
            fill_value=0,          # 黒で埋める
            p=0.3
        ),
        # リサイズと正規化
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_augmentation_heavy(img_size=224):
    """強力なaugmentation（過学習が深刻な場合）
    
    追加要素:
    - より強い色変換
    - ノイズ追加
    - Cutout
    """
    return A.Compose([
        # 幾何学変換
        A.ShiftScaleRotate(
            shift_limit=0.08,  # シフト拡大
            scale_limit=0.15,  # スケール拡大
            rotate_limit=7,    # 回転拡大（ただし慎重に）
            border_mode=0,
            p=0.6
        ),
        # 色変換（強化）
        A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.15,
            p=0.7
        ),
        # ぼかし
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        # ノイズ
        A.GaussianNoise(var_limit=(10.0, 50.0), p=0.2),
        # オクルージョン
        A.OneOf([
            A.CoarseDropout(
                max_holes=5,
                max_height=40,
                max_width=40,
                fill_value=0,
                p=1.0
            ),
            A.Cutout(
                num_holes=8,
                max_h_size=20,
                max_w_size=20,
                fill_value=0,
                p=1.0
            ),
        ], p=0.4),
        # リサイズと正規化
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_augmentation_test_time(img_size=224):
    """Test Time Augmentation (TTA)設定
    
    推論時に複数の変換でアンサンブル
    """
    return [
        # オリジナル
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 軽度の明度調整
        A.Compose([
            A.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0, p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 軽度のスケール変更
        A.Compose([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.05, rotate_limit=0, p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]


# ============================================================================
# 実装のガイドライン
# ============================================================================

"""
【実装手順】

1. まずは軽度版でベースライン改善
   → HorizontalFlip削除だけでも効果あり

2. 中程度版で性能向上を狙う
   → CoarseDropoutがオクルージョン対策に効果的

3. 過学習が見られたら強力版を試す
   → ただしaugmentationが強すぎると逆効果の可能性

4. TTAで最終スコア向上
   → 推論時間は3倍になるが、通常0.5-1%の改善が見込める

【注意点】
- augmentationを変更したら必ず複数foldで検証
- 単一foldのみだと偶然の改善/悪化を判断できない
- 学習曲線を確認して過学習の兆候をチェック
- テストデータとの分布の違いを意識
"""

# ============================================================================
# 使用例: model_arcface.pyでの適用方法
# ============================================================================

"""
# src/model_arcface.py の _get_transforms を以下のように変更:

def _get_transforms(train: bool = True, img_size: int = 224, aug_level: str = 'medium') -> A.Compose:
    '''
    画像変換パイプラインを取得
    
    Args:
        train: 訓練モードかどうか
        img_size: 画像サイズ
        aug_level: augmentationレベル ('light', 'medium', 'heavy')
    '''
    if train:
        if aug_level == 'light':
            return get_augmentation_light(img_size)
        elif aug_level == 'medium':
            return get_augmentation_medium(img_size)
        elif aug_level == 'heavy':
            return get_augmentation_heavy(img_size)
        else:
            raise ValueError(f"Unknown aug_level: {aug_level}")
    else:
        # 推論時は標準的な前処理のみ
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# Notebookでは以下のようにパラメータ指定:
params = {
    'model_name': 'efficientnet_b0',
    'img_size': 224,
    'aug_level': 'medium',  # ← 新規追加
    ...
}
"""

if __name__ == "__main__":
    # 可視化例
    print("="*80)
    print("データ拡張戦略ガイド")
    print("="*80)
    print("\n【推奨設定】")
    print("1. Light: HorizontalFlip削除、回転5度制限、色変換強化")
    print("2. Medium: Light + CoarseDropout + 軽度Blur（推奨）")
    print("3. Heavy: Medium強化版（過学習対策）")
    print("\n【実装ステップ】")
    print("① Lightでベースライン改善を確認")
    print("② Mediumで性能向上を狙う")
    print("③ 必要に応じてHeavyまたはTTA適用")
    print("\n詳細は本ファイルのコメントを参照")
