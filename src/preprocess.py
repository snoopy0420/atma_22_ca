"""
学習データの前処理（bbox切り出し）
RedBullのベースラインを参考に、パディング付きクロップを実装
"""
import os
import sys
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.abspath('..'))
from configs.config import *


def get_image_path(row: pd.Series, image_dir: Path) -> Path:
    """データフレームの行から画像パスを生成
    
    Args:
        row: メタデータの行（quarter, angle, session, frameを含む）
        image_dir: 画像ディレクトリのパス
        
    Returns:
        画像ファイルのパス
        
    Note:
        ファイル名形式: {quarter}__{angle}__{session:02d}__{frame:02d}.jpg
    """
    fname = f"{row['quarter']}__{row['angle']}__{row['session']:02d}__{row['frame']:02d}.jpg"
    return image_dir / fname


def process_single_crop(args: tuple) -> tuple:
    """単一画像のクロップ処理
    
    Args:
        args: (idx, row, image_dir, output_dir, padding_ratio)のタプル
            - idx: データフレームのインデックス
            - row: メタデータの行
            - image_dir: 画像ディレクトリのパス
            - output_dir: 出力ディレクトリのパス
            - padding_ratio: パディング比率（bboxサイズに対する割合）
            
    Returns:
        (インデックス, 成功フラグ)のタプル
        
    Note:
        - バウンディングボックスにパディングを追加してクロップ
        - JPEG品質95%で保存
        - エラー発生時はFalseを返す
    """
    idx, row, image_dir, output_dir, padding_ratio = args
    
    try:
        # 画像読み込み
        img_path = get_image_path(row, image_dir)
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"⚠️ 画像読み込み失敗: {img_path}")
            return idx, False
        
        # BBox座標取得
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        img_h, img_w = img.shape[:2]
        
        # パディング計算（bboxサイズの指定%）
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        
        # クロップ範囲計算（画像境界内に制限）
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        # クロップ
        crop = img[y1:y2, x1:x2]
        
        # 保存（JPEG品質95%）
        output_path = output_dir / f"{idx}.jpg"
        cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return idx, True
    
    except Exception as e:
        print(f"⚠️ インデックス {idx} の処理中にエラー: {e}")
        return idx, False


def preprocess_train_crops(
    csv_path: Path,
    image_dir: Path,
    output_dir: Path,
    padding_ratio: float = 0.1,
    num_workers: int = None,
):
    """学習データの全bboxを事前クロップして保存
    
    Args:
        csv_path: train_meta.csvのパス
        image_dir: 元画像ディレクトリ（images/）
        output_dir: クロップ画像保存先
        padding_ratio: パディング比率（デフォルト: 0.1 = 10%）
        num_workers: 並列処理ワーカー数（Noneの場合はCPU数）
        
    Note:
        - ProcessPoolExecutorで並列化して高速処理
        - 出力ディレクトリは自動作成
        - 失敗したサンプルのインデックスを記録・表示
        - クロップ画像は{idx}.jpg形式で保存
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # メタデータ読み込み
    df = pd.read_csv(str(csv_path))
    print(f"📋 {len(df)}サンプルを{num_workers}ワーカーで処理開始...")
    print(f"   入力: {image_dir}")
    print(f"   出力: {output_dir}")
    print(f"   パディング: {padding_ratio*100:.1f}%")
    
    # 並列処理用引数リスト
    args_list = [
        (idx, row, image_dir, output_dir, padding_ratio)
        for idx, row in df.iterrows()
    ]
    
    # 並列処理実行
    success_count = 0
    failed_indices = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 全タスク投入
        futures = {
            executor.submit(process_single_crop, args): args[0]
            for args in args_list
        }
        
        # 結果取得
        for future in tqdm(as_completed(futures), total=len(futures), desc="クロッピング中"):
            idx, success = future.result()
            if success:
                success_count += 1
            else:
                failed_indices.append(idx)
    
    # 結果サマリー
    print(f"\n✅ 処理完了:")
    print(f"   成功: {success_count}/{len(df)} ({100*success_count/len(df):.2f}%)")
    if failed_indices:
        print(f"   ⚠️ 失敗: {len(failed_indices)}サンプル")
        print(f"   失敗インデックス: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")


