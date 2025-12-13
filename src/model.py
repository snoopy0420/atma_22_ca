"""
モデルの抽象クラス
過去コンペの構成に従った設計
"""
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional


class Model(metaclass=ABCMeta):
    """
    model_xxのスーパークラス
    abcモジュールにより抽象メソッドを定義
    """

    def __init__(self, 
                 run_fold_name: str, 
                 params: dict,
                 out_dir_name: str,
                 logger
                 ) -> None:
        """コンストラクタ
        Args:
            run_fold_name: runの名前とfoldの番号を組み合わせた名前
            params: ハイパーパラメータ
            out_dir_name: 出力ディレクトリ
            logger: ロガー
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.out_dir_name = out_dir_name
        self.logger = logger

    @abstractmethod
    def train(self, 
              tr: pd.DataFrame,
              va: Optional[pd.DataFrame] = None
              ) -> None:
        """モデルの学習を行い、学習済のモデルを保存する
        Args:
            tr: 学習データ
            va: バリデーションデータ
        """
        pass

    @abstractmethod
    def predict(self, te: pd.DataFrame) -> pd.DataFrame:
        """学習済のモデルでの予測値を返す
        Args:
            te: テストデータ
        Returns:
            予測結果のDataFrame
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """モデルの保存を行う"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """モデルの読み込みを行う"""
        pass
