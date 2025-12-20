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
from src.util import Util, Metric, Validation, seed_everything


class Runner:
    """学習・予測・評価を担うクラス"""

    def __init__(self,
                 run_name: str,
                 model_cls: Callable,
                 params: dict,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 cv_setting: dict,
                 logger,
                 seed: int = 42
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
            seed: ランダムシード（デフォルト: 42）
        """
        # シード固定（再現性確保）
        seed_everything(seed)
        self.seed = seed
        
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.logger = logger
        
        # CV設定（validatorは必須）
        if 'validator' not in cv_setting:
            raise ValueError("cv_setting に 'validator' が必要です。Validation.create_validator()で作成してください。")
        self.validator = cv_setting['validator']
        self.group_col = cv_setting.get("group_col")
        self.n_splits = getattr(self.validator, 'n_splits', getattr(self.validator, 'get_n_splits', lambda: 5)())
        
        # データ
        self.df_train = df_train
        self.df_test = df_test
        self.target_col = 'label_id'
        
        # 出力ディレクトリ
        self.out_dir_name = os.path.join(DIR_MODEL, run_name)
        os.makedirs(self.out_dir_name, exist_ok=True)
        
        self.logger.info(f"Runner初期化完了: {run_name}")
        self.logger.info(f"  学習データ: {len(df_train)}, テストデータ: {len(df_test)}")
        self.logger.info(f"  グループ列: {self.group_col if self.group_col else 'なし'}")
        
        # グループ数の確認（リークチェック用）
        if self.group_col:
            n_groups = df_train[self.group_col].nunique()
            self.logger.info(f"  総グループ数: {n_groups}")
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
        # group_colがある場合のみgroupsを使用
        if self.group_col and self.group_col in self.df_train.columns:
            groups = self.df_train[self.group_col].values
        else:
            groups = None
        
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
        self.logger.info(f"  学習: {len(tr):,}, 検証: {len(va):,}")
        
        # モデル構築
        model = self.build_model(i_fold)
        
        # 学習
        self.logger.info("学習開始...")
        model.train(tr, va)
        
        # モデル保存
        model.save_model()
        
        return model


    def train_cv(self):
        """Cross Validationでの学習"""
        self.logger.section_start(f"Starting {self.n_splits}-Fold Cross Validation")
        
        for i_fold in range(self.n_splits):
            model = self.train_fold(i_fold)
        
        # パラメータの保存
        try:
            path_output = os.path.join(self.out_dir_name, f'params.yaml')
            Util.jump_json(self.params, path_output)
        except:
            self.logger.info("パラメータは保存しません")


    def train_all(self):
        """全データでの学習（CV不要の場合）"""
        self.logger.section_start("Training on all data")
        
        # モデル構築
        model = self.build_model(i_fold='all')
        
        # 学習
        self.logger.info(f"{len(self.df_train):,}サンプルで学習中...")
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
            self.logger.info(f"Fold {i_fold}で予測中...")
            
            # モデル読み込み
            model = self.build_model(i_fold)
            model.load_model()
            
            # 予測
            te_pred = model.predict(self.df_test, split='test')
            all_predictions.append(te_pred['label_id'].values)
        
        # アンサンブル（多数決）
        self.logger.info("予測結果をアンサンブル中（多数決）...")
        # 各foldの予測長が一致しているか確認
        pred_lengths = [len(p) for p in all_predictions]
        self.logger.info(f"各foldの予測サンプル数: {pred_lengths}")
        if len(set(pred_lengths)) > 1:
            raise ValueError(f"Prediction lengths are inconsistent: {pred_lengths}")
        
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
        self.logger.info(f"最終予測分布:\n{pred_counts}")
        
        return submission


    def predict_all(self):
        """全データ学習済みモデルでの予測"""
        self.logger.section_start("Predicting with all-data model")
        
        # モデル読み込み
        model = self.build_model(i_fold='all')
        model.load_model()
        
        # 予測
        self.logger.info(f"{len(self.df_test)}サンプルを予測中...")
        pred = model.predict(self.df_test)
        
        # 予測分布
        pred_counts = pred['label_id'].value_counts().sort_index()
        self.logger.info(f"予測分布:\n{pred_counts}")

        submission = pd.DataFrame({'label_id': pred['label_id'].values})
        
        return submission

    
    
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
            self.logger.info(f"Fold {i_fold}を評価中...")
            
            # データ分割
            _, va = self.create_train_valid_dataset(i_fold)
            
            # 学習済みモデル読み込み
            model = self.build_model(i_fold)
            model.load_model()
            
            # 検証データで予測（元のインデックスを保持）
            va_pred = model.predict(va, split='valid')
            
            # スコア計算
            va_score = self.metric(va[self.target_col].values, va_pred['label_id'].values)
            scores.append(va_score)
            va_preds_all.append(va_pred)
            
            self.logger.info(f"  Fold {i_fold} スコア: {va_score:.5f}")
        
        # 全foldの予測を結合してOOFスコア計算
        df_va_preds = pd.concat(va_preds_all, axis=0)
        
        # インデックスでソートして元の順序に並べる
        df_va_preds = df_va_preds.sort_index()
        
        # 訓練データと同じ順序でOOFスコア計算
        # インデックスが一致することを確認
        if not df_va_preds.index.equals(self.df_train.index):
            self.logger.info("⚠️ Warning: OOF予測のインデックスが訓練データと一致しません！")
            self.logger.info(f"  訓練データ: {len(self.df_train)}行, OOF予測: {len(df_va_preds)}行")
        
        oof_score = self.metric(
            self.df_train.loc[df_va_preds.index, 'label_id'].values,
            df_va_preds['label_id'].values
        )

        # 結果サマリー
        self.logger.info("\n" + "="*80)
        self.logger.info(f"OOFスコア (Macro F1): {oof_score:.5f}")
        self.logger.info(f"Foldスコア平均: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
        self.logger.info("="*80)
        
        # 結果保存
        self.logger.result_scores(self.run_name, scores)
        
        # OOF予測結果の保存
        path_output = os.path.join(self.out_dir_name, 'va_pred.pkl')
        Util.dump_df_pickle(df_va_preds, path_output)
        self.logger.info(f'OOF予測結果保存: {path_output}')

        return scores, oof_score