"""
学習・予測・評価を管理するRunnerクラス
過去コンペの構成に従った設計
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Callable, Optional
from datetime import datetime

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.util import Util, Metric, Validation


class Runner:
    """学習・予測・評価を担うクラス"""

    def __init__(self,
                 run_name: str,
                 model_cls: Callable,
                 params: dict,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 cv_setting: dict,
                 logger
                 ):
        """
        Args:
            run_name: 実行名
            model_cls: モデルクラス
            params: パラメータ
            df_train: 訓練データ
            df_test: テストデータ
            cv_setting: CV設定
            logger: ロガー
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.logger = logger
        
        # CV設定
        self.group_col = cv_setting.get("group_col")
        self.n_splits = cv_setting.get("n_splits", 5)
        self.cv_method = cv_setting.get("method", "group")  # "group" or "stratified_group"
        
        # Validation を使用してvalidator作成
        self.validator = Validation.create_validator(
            method=self.cv_method,
            n_splits=self.n_splits,
            shuffle=cv_setting.get("shuffle", True),
            random_state=cv_setting.get("random_state", 42)
        )
        self.logger.info(f"Using {self.cv_method}")
        
        # データ
        self.df_train = df_train
        self.df_test = df_test
        self.target_col = 'label_id'
        
        # 出力ディレクトリ
        self.out_dir_name = os.path.join(DIR_MODEL, run_name)
        os.makedirs(self.out_dir_name, exist_ok=True)
        
        self.logger.info(f"Runner initialized: {run_name}")
        self.logger.info(f"  Train: {len(df_train)}, Test: {len(df_test)}")
        self.logger.info(f"  CV: {self.n_splits}-fold {self.cv_method}, group_col: {self.group_col}")
        
        # グループ数の確認（リークチェック用）
        if self.group_col:
            n_groups = df_train[self.group_col].nunique()
            self.logger.info(f"  Total groups: {n_groups}")
            if n_groups < self.n_splits:
                self.logger.warning(f"  ⚠️ Warning: Only {n_groups} groups for {self.n_splits}-fold CV!")


    def metric(self, true: np.ndarray, pred: np.ndarray) -> float:
        """評価指標（Macro F1）"""
        return Metric.macro_f1(true, pred)


    def build_model(self, i_fold: int) -> Model:
        """モデルのインスタンス作成"""
        run_fold_name = f'{self.run_name}_fold-{i_fold}'
        model = self.model_cls(
            run_fold_name,
            self.params.copy(),
            self.out_dir_name,
            self.logger
        )
        return model


    def create_train_valid_dataset(self, i_fold: int):
        """訓練・検証データの分割"""
        groups = self.df_train[self.group_col].values
        y = self.df_train[self.target_col].values
        
        splits = list(self.validator.split(self.df_train, y, groups))
        tr_idx, va_idx = splits[i_fold]
        
        # Validationのユーティリティを使用
        tr, va = Validation.split_by_index(self.df_train, tr_idx, va_idx)
        
        return tr, va


    def train_fold(self, i_fold: int):
        """指定foldでの学習・評価"""
        self.logger.fold_start(i_fold, self.n_splits)
        
        # データ分割
        tr, va = self.create_train_valid_dataset(i_fold)
        self.logger.info(f"  Train: {len(tr):,}, Valid: {len(va):,}")
        
        # モデル構築
        model = self.build_model(i_fold)
        
        # 学習
        self.logger.info("Training...")
        model.train(tr, va)
        
        # 検証データで評価
        self.logger.info("Predicting on validation...")
        va_pred = model.predict(va)
        
        # スコア計算
        va_score = self.metric(va[self.target_col].values, va_pred['label_id'].values)
        self.logger.fold_result(i_fold, va_score)
        
        # モデル保存
        model.save_model()
        
        # 予測分布を確認
        pred_counts = va_pred['label_id'].value_counts().sort_index()
        self.logger.info(f"  Prediction distribution:\n{pred_counts}")
        
        return model, va_pred, va_score


    def train_cv(self):
        """Cross Validationでの学習"""
        self.logger.section_start(f"Starting {self.n_splits}-Fold Cross Validation")
        
        scores = []
        va_preds_all = []
        
        for i_fold in range(self.n_splits):
            model, va_pred, score = self.train_fold(i_fold)
            scores.append(score)
            va_preds_all.append(va_pred)
        
        # CV結果のサマリー
        self.logger.cv_summary(scores)
        
        # 結果保存
        self.logger.result_scores(self.run_name, scores)
        
        # パラメータの保存
        try:
            path_output = os.path.join(self.out_dir_name, f'params.yaml')
            Util.jump_json(self.params, path_output)
        except:
            self.logger.info("パラメータは保存しません")
        
        return scores


    def train_all(self):
        """全データでの学習（CV不要の場合）"""
        self.logger.section_start("Training on all data")
        
        # モデル構築
        model = self.build_model(i_fold='all')
        
        # 学習
        self.logger.info(f"Training with {len(self.df_train):,} samples...")
        model.train(self.df_train)
        
        # モデル保存
        model.save_model()
        
        self.logger.section_end("Training completed!")
        
        return model


    def predict_cv(self):
        """CV学習済みモデルでの予測（アンサンブル）"""
        self.logger.section_start(f"Predicting with CV ensemble ({self.n_splits} folds)")
        
        # 各foldで予測
        all_predictions = []
        
        for i_fold in range(self.n_splits):
            self.logger.info(f"Predicting with fold {i_fold}...")
            
            # モデル読み込み
            model = self.build_model(i_fold)
            model.load_model()
            
            # 予測
            te_pred = model.predict(self.df_test)
            all_predictions.append(te_pred['label_id'].values)
        
        # アンサンブル（多数決）
        self.logger.info("Ensembling predictions...")
        all_predictions = np.array(all_predictions)  # (n_folds, n_samples)
        
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            fold_preds = all_predictions[:, i]
            # 多数決
            unique, counts = np.unique(fold_preds, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        final_predictions = np.array(final_predictions)
        
        # 結果DataFrame作成
        submission = pd.DataFrame({'label_id': final_predictions})
        
        # 予測分布
        pred_counts = submission['label_id'].value_counts().sort_index()
        self.logger.info(f"Final prediction distribution:\n{pred_counts}")
        
        return submission


    def predict_all(self):
        """全データ学習済みモデルでの予測"""
        self.logger.section_start("Predicting with all-data model")
        
        # モデル読み込み
        model = self.build_model(i_fold='all')
        model.load_model()
        
        # 予測
        self.logger.info(f"Predicting {len(self.df_test)} samples...")
        submission = model.predict(self.df_test)
        
        # 予測分布
        pred_counts = submission['label_id'].value_counts().sort_index()
        self.logger.info(f"Prediction distribution:\n{pred_counts}")
        
        return submission


    def save_submission(self, submission: pd.DataFrame, suffix: str = ""):
        """提出ファイル保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"submission_{self.run_name}_{suffix}_{timestamp}.csv" if suffix else f"submission_{self.run_name}_{timestamp}.csv"
        save_path = os.path.join(DIR_SUBMISSIONS, filename)
        
        submission.to_csv(save_path, index=False, header=False)
        self.logger.info(f"Submission saved: {save_path}")
        
        return save_path
    
    
    def load_model_cv(self, i_fold: int) -> Model:
        """指定foldのモデルを読み込み"""
        model = self.build_model(i_fold)
        model.load_model()
        return model


    def metric_cv(self):
        """CVでの評価を行う（OOF評価）
        
        学習済みの各foldモデルで検証データを予測し、全体のOOFスコアを算出
        
        Returns:
            scores: 各foldのスコアリスト
            oof_score: OOF全体のスコア
        """
        self.logger.section_start(f"Evaluating {self.n_splits}-Fold CV (OOF)")

        scores = []
        va_preds_all = []

        # fold毎の検証データの予測・評価
        for i_fold in range(self.n_splits):
            self.logger.info(f"Evaluating fold {i_fold}...")
            
            # データ分割
            _, va = self.create_train_valid_dataset(i_fold)
            
            # 学習済みモデル読み込み
            model = self.build_model(i_fold)
            model.load_model()
            
            # 検証データで予測
            va_pred = model.predict(va)
            
            # スコア計算
            va_score = self.metric(va[self.target_col].values, va_pred['label_id'].values)
            scores.append(va_score)
            va_preds_all.append(va_pred)
            
            self.logger.info(f"  Fold {i_fold} score: {va_score:.5f}")
        
        # 全foldの予測を結合してOOFスコア計算
        df_va_preds = pd.concat(va_preds_all, axis=0).reset_index(drop=True)
        
        # 訓練データと結合して正解ラベルと比較
        # インデックスが維持されているので直接比較可能
        oof_score = self.metric(
            self.df_train['label_id'].values,
            df_va_preds['label_id'].values
        )

        # 結果サマリー
        self.logger.info("\n" + "="*80)
        self.logger.info(f"OOF Score (Macro F1): {oof_score:.5f}")
        self.logger.info(f"Fold Scores Mean: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
        self.logger.info("="*80)
        
        # 結果保存
        self.logger.result_scores(self.run_name, scores)
        
        # OOF予測結果の保存
        path_output = os.path.join(self.out_dir_name, 'va_pred.pkl')
        Util.dump_df_pickle(df_va_preds, path_output)
        self.logger.info(f'OOF predictions saved: {path_output}')

        return scores, oof_score
