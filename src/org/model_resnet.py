"""
ResNet50ベースの画像特徴抽出モデル（後方互換性ラッパー）

【重要】このファイルは後方互換性のために残されています。
新規コードでは以下のクラスを直接使用してください：
- src.model_resnet_prototype.ModelResNet50Prototype (Prototype法)
- src.model_resnet_knn.ModelResNet50KNN (KNN法)

使い方:
    # 旧方式（このファイル経由で自動振り分け）
    from src.model_resnet import ModelResNet50
    model = ModelResNet50(run_fold_name, params, out_dir, logger)
    
    # 新方式（推奨）
    from src.model_resnet_prototype import ModelResNet50Prototype
    model = ModelResNet50Prototype(run_fold_name, params, out_dir, logger)
"""
import os
import sys

sys.path.append(os.path.abspath('..'))
from src.model_resnet_prototype import ModelResNet50Prototype
from src.model_resnet_knn import ModelResNet50KNN


class ModelResNet50:
    """
    後方互換性のためのラッパークラス
    
    params['method'] に応じて適切なクラスに振り分ける:
    - 'prototype' → ModelResNet50Prototype
    - 'knn' → ModelResNet50KNN
    """
    
    def __new__(cls, run_fold_name: str, params: dict, out_dir_name: str, logger):
        """
        インスタンス生成時に適切なクラスに振り分け
        
        Args:
            run_fold_name: 実行名（fold情報含む）
            params: パラメータ辞書
            out_dir_name: 出力ディレクトリ
            logger: ロガー
        
        Returns:
            ModelResNet50Prototype または ModelResNet50KNN のインスタンス
        """
        method = params.get('method', 'prototype')
        
        if method == 'prototype':
            logger.info(f"Creating ModelResNet50Prototype instance (method='{method}')")
            return ModelResNet50Prototype(run_fold_name, params, out_dir_name, logger)
        elif method == 'knn':
            logger.info(f"Creating ModelResNet50KNN instance (method='{method}')")
            return ModelResNet50KNN(run_fold_name, params, out_dir_name, logger)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'prototype' or 'knn'.")

