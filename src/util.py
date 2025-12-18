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
    def check_group_leak(validator, X: pd.DataFrame, y: np.ndarray, 
                        groups: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """
        グループ間のリークをチェック
        
        Args:
            validator: sklearn cross-validator
            X: 特徴量データフレーム
            y: ターゲット配列
            groups: グループ配列
            verbose: 詳細出力の有無
        
        Returns:
            Dict containing:
                - has_leak: bool (リークの有無)
                - fold_results: List[Dict] (各Foldの結果)
        """
        fold_results = []
        has_leak = False
        
        if verbose:
            print("="*80)
            print("🔍 CV Group Leak Check")
            print("="*80)
        
        for fold_idx, (train_idx, valid_idx) in enumerate(validator.split(X, y, groups)):
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            
            # 重複チェック
            overlap = train_groups & valid_groups
            fold_has_leak = len(overlap) > 0
            has_leak = has_leak or fold_has_leak
            
            leak_status = "❌ LEAK!" if fold_has_leak else "✅ No leak"
            
            # 選手分布
            train_labels = set(y[train_idx])
            valid_labels = set(y[valid_idx])
            
            # 各選手のサンプル数
            train_label_counts = pd.Series(y[train_idx]).value_counts()
            valid_label_counts = pd.Series(y[valid_idx]).value_counts()
            
            fold_result = {
                'fold': fold_idx,
                'train_samples': len(train_idx),
                'valid_samples': len(valid_idx),
                'train_groups': len(train_groups),
                'valid_groups': len(valid_groups),
                'overlap_groups': len(overlap),
                'has_leak': fold_has_leak,
                'train_labels': len(train_labels),
                'valid_labels': len(valid_labels),
                'overlap_labels': len(train_labels & valid_labels),
                'train_label_min': train_label_counts.min(),
                'train_label_max': train_label_counts.max(),
                'valid_label_min': valid_label_counts.min(),
                'valid_label_max': valid_label_counts.max(),
            }
            fold_results.append(fold_result)
            
            if verbose:
                print(f"\nFold {fold_idx}: {leak_status}")
                print(f"  Train: {len(train_idx):5,} samples, {len(train_groups):3d} groups")
                print(f"  Valid: {len(valid_idx):5,} samples, {len(valid_groups):3d} groups")
                print(f"  Overlap groups: {len(overlap)}")
                print(f"  Players - Train: {len(train_labels)}, Valid: {len(valid_labels)}, Overlap: {len(train_labels & valid_labels)}")
                print(f"  Label balance (train): min={train_label_counts.min()}, max={train_label_counts.max()}")
                print(f"  Label balance (valid): min={valid_label_counts.min()}, max={valid_label_counts.max()}")
        
        if verbose:
            print("\n" + "="*80)
            if has_leak:
                print("❌ LEAK DETECTED!")
            else:
                print("✅ No Leakage - CV Strategy is Valid")
            print("="*80)
        
        return {
            'has_leak': has_leak,
            'fold_results': fold_results
        }
    
    
    @staticmethod
    def get_cv_statistics(validator, X: pd.DataFrame, y: np.ndarray, 
                         groups: np.ndarray) -> pd.DataFrame:
        """
        CV分割の統計情報をDataFrameとして取得
        
        Args:
            validator: sklearn cross-validator
            X: 特徴量データフレーム
            y: ターゲット配列
            groups: グループ配列
        
        Returns:
            pd.DataFrame: 各Foldの統計情報
        """
        fold_stats = []
        
        for fold_idx, (train_idx, valid_idx) in enumerate(validator.split(X, y, groups)):
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            
            fold_stats.append({
                'fold': fold_idx,
                'train_samples': len(train_idx),
                'valid_samples': len(valid_idx),
                'train_groups': len(train_groups),
                'valid_groups': len(valid_groups),
                'overlap_groups': len(train_groups & valid_groups),
            })
        
        return pd.DataFrame(fold_stats)
    
    
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
    
    
    @staticmethod
    def log_fold_result(fold_idx: int, train_size: int, valid_size: int, 
                       score: float, metric_name: str = "Macro F1"):
        """
        Fold結果のログ出力
        
        Args:
            fold_idx: Fold番号
            train_size: 訓練データサイズ
            valid_size: 検証データサイズ
            score: スコア
            metric_name: メトリクス名
        """
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} Results")
        print(f"{'='*60}")
        print(f"  Train samples: {train_size:,}")
        print(f"  Valid samples: {valid_size:,}")
        print(f"  {metric_name}: {score:.6f}")
    
    
    @staticmethod
    def log_cv_summary(scores: List[float], metric_name: str = "Macro F1"):
        """
        CV全体のサマリーログ出力
        
        Args:
            scores: 各Foldのスコアリスト
            metric_name: メトリクス名
        """
        print(f"\n{'='*60}")
        print(f"Cross Validation Summary")
        print(f"{'='*60}")
        print(f"  {metric_name} - Mean: {np.mean(scores):.6f}")
        print(f"  {metric_name} - Std:  {np.std(scores):.6f}")
        print(f"  Fold scores: {[f'{s:.6f}' for s in scores]}")
        print(f"{'='*60}")
