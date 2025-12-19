"""
ユーティリティクラス群
過去コンペの構成に従った設計
"""
import datetime
import logging
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import (
    KFold, 
    StratifiedKFold, 
    GroupKFold, 
    StratifiedGroupKFold
)
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath('..'))
from configs.config import *


class Util:
    """ファイル操作ユーティリティ"""

    @classmethod
    def dump(cls, value, path):
        """オブジェクトを保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        """オブジェクトを読み込み"""
        return joblib.load(path)

    @classmethod
    def dump_json(cls, value, path):
        """JSONを保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_json(cls, path):
        """JSONを読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def dump_df_pickle(cls, df, path):
        """DataFrameをpickleで保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        """DataFrameをpickleから読み込み"""
        return pd.read_pickle(path)

    @classmethod
    def load_feature(cls, file_name):
        """特徴量ファイルを読み込み"""
        file_name = file_name if file_name.endswith('.pkl') else file_name + ".pkl"
        return pd.read_pickle(os.path.join(DIR_FEATURE, file_name))
    
    @classmethod
    def save_submission(cls, submission: pd.DataFrame, run_name: str, suffix: str = "", logger=None) -> str:
        """提出ファイル保存（後方互換性のため残す）
        
        Note: Submissionクラスの使用を推奨
        """
        return Submission.save(submission, run_name, suffix, logger)


class Submission:
    """提出ファイル管理クラス"""
    
    @staticmethod
    def save(submission: pd.DataFrame, run_name: str, logger=None) -> str:
        """提出ファイル保存
        
        Args:
            submission: 提出用DataFrame（label_id列を持つ）
            run_name: 実行名
            suffix: ファイル名のサフィックス（オプション）
            logger: ロガー（オプション）
        
        Returns:
            保存先のパス
        
        Examples:
            >>> submission = pd.DataFrame({'label_id': predictions})
            >>> Submission.save(submission, 'resnet50_knn', 'tuned')
            'data/submission/submission_resnet50_knn_tuned_20251218_143022.csv'
        """
        from datetime import datetime
        
        if Submission.validate(submission):

            # ファイル名生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"submission_{run_name}_{timestamp}.csv"
            save_path = os.path.join(DIR_SUBMISSIONS, filename)
            
            # 保存
            os.makedirs(DIR_SUBMISSIONS, exist_ok=True)
            submission.to_csv(save_path, index=False, header=True)
            
            logger.info(f"提出ファイル保存: {save_path}")

            return save_path
            
    
    @staticmethod
    def validate(submission: pd.DataFrame) -> bool:
        """提出ファイルのバリデーション
        
        Args:
            submission: 提出用DataFrame
            expected_length: 期待される行数（オプション）
        
        Returns:
            検証結果（True: OK, False: NG）
        """
        sample_submission = pd.read_csv(FILE_SAMPLE_SUBMISSION)
        
        # カラムチェック
        if submission.columns != sample_submission.columns:
            print(f"❌ カラム名エラー: {submission.columns}（期待: {sample_submission.columns}）")
            return False
        
        # # 長さチェック
        # expected_length = sample_submission.shape[0]
        # if len(submission) != expected_length:
        #     print(f"❌ 行数エラー: {len(submission)}行（期待: {expected_length}行）")
        #     return False
        
        # データ型チェック
        if not pd.api.types.is_integer_dtype(submission['label_id']):
            print(f"❌ データ型エラー: label_id列は整数型である必要があります")
            return False
        
        
        print(f"✅ バリデーション成功: {len(submission)}行")
        return True


class Logger:
    """ロギングクラス"""

    def __init__(self, path):
        """
        Args:
            path: ログ出力ディレクトリ
        """
        os.makedirs(path, exist_ok=True)
        
        self.general_logger = logging.getLogger(os.path.join(path, 'general'))
        self.result_logger = logging.getLogger(os.path.join(path, 'result'))
        
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, 'general.log'))
        file_result_handler = logging.FileHandler(os.path.join(path, 'result.log'))
        
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        """時刻付きでコンソールとログに出力"""
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        """結果ログに出力"""
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        """結果をLTSV形式で出力"""
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        """CVスコアを出力"""
        dic = dict()
        dic['run_name'] = run_name
        dic['score_mean'] = np.mean(scores)
        dic['score_std'] = np.std(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def section_start(self, title: str, width: int = 80):
        """セクション開始ログ"""
        self.info("")
        self.info("="*width)
        self.info(title)
        self.info("="*width)

    def section_end(self, title: str = "Completed", width: int = 80):
        """セクション終了ログ"""
        self.info("="*width)
        self.info(title)
        self.info("="*width)

    def fold_start(self, fold_idx: int, n_folds: int, width: int = 80):
        """Fold開始ログ"""
        self.info("")
        self.info("="*width)
        self.info(f"Fold {fold_idx} / {n_folds}")
        self.info("="*width)

    def fold_result(self, fold_idx: int, score: float, metric_name: str = "Macro F1", train_size: int = None, valid_size: int = None):
        """Fold結果ログ"""
        if train_size and valid_size:
            self.info(f"  Train: {train_size:,}, Valid: {valid_size:,}")
        self.info(f"  {metric_name}: {score:.6f}")

    def cv_summary(self, scores: list, width: int = 80):
        """CVサマリーログ"""
        self.info("")
        self.info("="*width)
        self.info("CV Results Summary")
        self.info("="*width)
        for i, score in enumerate(scores):
            self.info(f"  Fold {i}: {score:.6f}")
        self.info(f"  Mean: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")

    def now_string(self):
        """現在時刻の文字列"""
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        """辞書をLTSV形式に変換"""
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Metric:
    """評価指標クラス"""

    @classmethod
    def macro_f1(cls, y_true, y_pred, labels: Optional[List[int]] = None):
        """
        Macro F1スコアの計算
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            labels: 評価対象のラベルリスト（Noneの場合は自動推定）
        Returns:
            Macro F1スコア
        """
        if labels is None:
            # -1（unknown）も含めて評価
            labels = sorted(set(y_true) | set(y_pred))
        
        score = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        return score

    @classmethod
    def my_metric(cls, y_true, y_pred):
        """コンペ用の評価指標（Macro F1）"""
        return cls.macro_f1(y_true, y_pred)
    
    @classmethod
    def unknown_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray, 
                       unknown_player_id: int) -> dict:
        """Unknown判定の評価指標を計算
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            unknown_player_id: unknown判定すべき選手ID
        
        Returns:
            dict: 評価指標の辞書
                - unknown_precision: Unknown判定の精度
                - unknown_recall: Unknown判定の再現率
                - unknown_f1: Unknown判定のF1
                - known_macro_f1: 既知選手のMacro F1
                - unknown_samples: unknown選手のサンプル数
                - unknown_detected: unknownと予測したサンプル数
        """
        # unknown選手のマスク
        is_unknown = (y_true == unknown_player_id)
        
        # 予測がunknown (-1) かどうか
        pred_unknown = (y_pred == -1)
        
        # Unknown判定の評価
        tp = np.sum(is_unknown & pred_unknown)
        fp = np.sum(~is_unknown & pred_unknown)
        fn = np.sum(is_unknown & ~pred_unknown)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 既知選手のMacro F1
        known_mask = ~is_unknown
        known_macro_f1 = cls.macro_f1(y_true[known_mask], y_pred[known_mask]) if known_mask.sum() > 0 else 0.0
        
        return {
            'unknown_precision': float(precision),
            'unknown_recall': float(recall),
            'unknown_f1': float(f1),
            'known_macro_f1': float(known_macro_f1),
            'unknown_samples': int(is_unknown.sum()),
            'unknown_detected': int(pred_unknown.sum()),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }


class Validation:
    """CV分割とリークチェックを管理するクラス"""
    
    @staticmethod
    def create_validator(method: str, n_splits: int = 5, **kwargs):
        """
        CV手法を選択してvalidatorを生成
        
        Args:
            method: CV手法の種類
                - 'kfold': KFold
                - 'stratified': StratifiedKFold
                - 'group': GroupKFold (推奨: リーク防止)
                - 'stratified_group': StratifiedGroupKFold
            n_splits: Fold数
            **kwargs: 各validator固有のパラメータ
                - shuffle: シャッフルの有無 (default: True)
                - random_state: 乱数シード (default: 42)
        
        Returns:
            sklearn cross-validator
        """
        shuffle = kwargs.get('shuffle', True)
        random_state = kwargs.get('random_state', 42)
        
        if method == 'kfold':
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        elif method == 'stratified':
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        elif method == 'group':
            # GroupKFold: グループ間でリークなし（層化なし）
            return GroupKFold(n_splits=n_splits)
        
        elif method == 'stratified_group':
            # StratifiedGroupKFold: グループ間でリークなし + 層化を試みる
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        else:
            raise ValueError(f"Unknown CV method: {method}. Choose from ['kfold', 'stratified', 'group', 'stratified_group']")
    
    
    @staticmethod
    def split_by_index(df: pd.DataFrame, train_idx: np.ndarray, 
                      valid_idx: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        DataFrameをインデックスで分割
        
        Args:
            df: 分割対象のDataFrame
            train_idx: 訓練データのインデックス
            valid_idx: 検証データのインデックス
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, valid_df)
        """
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        return train_df, valid_df
    

class Q1Q2Validator:
    """Q1/Q2クォーター分割用のValidator
    
    Unknown判定評価用のカスタムCV戦略
    - Fold 0: Q2で訓練 → Q1で検証（選手0がunknown）
    - Fold 1: Q1で訓練 → Q2で検証（選手5がunknown）
    """
    
    def __init__(self, quarter_col: str = 'quarter'):
        """
        Args:
            quarter_col: クォーター列名
        """
        self.quarter_col = quarter_col
        self.n_splits = 2
    
    def split(self, X, y=None, groups=None):
        """CV分割を生成
        
        Args:
            X: DataFrameまたは配列（quarter列を含む必要あり）
            y: ターゲット（未使用）
            groups: グループ（未使用）
        
        Yields:
            (train_indices, valid_indices)のタプル
        """
        if isinstance(X, pd.DataFrame):
            quarters = X[self.quarter_col]
        else:
            raise ValueError("Q1Q2ValidatorはDataFrameが必要です")
        
        # Q1/Q2のマスクを作成
        q1_mask = quarters.astype(str).str.startswith('Q1')
        q2_mask = quarters.astype(str).str.startswith('Q2')
        
        q1_indices = X[q1_mask].index.values
        q2_indices = X[q2_mask].index.values
        
        # Fold 0: Q2訓練 → Q1検証
        yield q2_indices, q1_indices
        
        # Fold 1: Q1訓練 → Q2検証
        yield q1_indices, q2_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Fold数を返す"""
        return self.n_splits
