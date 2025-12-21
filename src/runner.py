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
from src import post_processing


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

    def load_model_cv(self, i_fold: int) -> Model:
        """指定foldのモデルを読み込み"""
        model = self.build_model(i_fold)
        model.load_model()
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


    def predict_cv_with_postprocessing(
        self,
        unknown_prob: Optional[float] = None,
        max_labels: int = 10,
    ):
        """
        CV予測に個数制約付き最適化を適用
        
        処理フロー:
        各Foldで確率予測:
        ├─ Fold 0: テストデータ → 確率予測
        ├─ Fold 1: テストデータ → 確率予測
        └─ Fold 2: テストデータ → 確率予測
        ↓
        確率を平均
        ↓
        平均確率に最適化適用
        
        Args:
            unknown_prob: Unknownラベル（index=11）に割り当てる確率値
                         ※params['threshold']（類似度閾値）とは別物
            max_labels: 全体で使用可能な最大ラベル数
        """
        self.logger.section_start(
            f"CV Prediction with Optimization (unknown_prob={unknown_prob})"
        )
        
        # 各foldで確率予測
        probs_list = []
        
        for i_fold in range(self.n_splits):
            self.logger.info(f"Fold {i_fold}のモデルで確率予測中...")
            model = self.load_model_cv(i_fold)
            test_probs = model.predict_proba(self.df_test, split='test')
            probs_list.append(test_probs)
            self.logger.info(f"  Fold {i_fold}: shape={test_probs.shape}")
        
        # 各foldの確率を平均
        probs_mean = np.mean(probs_list, axis=0)
        self.logger.info(f"確率の平均完了: {probs_mean.shape}")
        
        # 最適化を適用
        self.logger.info("個数制約付き最適化を実行中...")
        df_test_with_probs = self.df_test.copy()
        df_test_with_probs['probs'] = probs_mean.tolist()
        
        optimized_pred = post_processing.apply_post_processing(df_test_with_probs)

        # optimized_pred = post_processing.apply_postprocessing(
        #     test_meta=self.df_test,
        #     probs=probs_mean,
        #     unknown_prob=unknown_prob,
        #     max_labels=max_labels,
        #     verbose=True
        # )

        submission = pd.DataFrame({'label_id': optimized_pred})
        
        return submission



    def metric_cv_with_postprocessing(
        self,
        unknown_prob: Optional[float] = None,
        max_labels: int = 10
    ):
        """
        CVでの評価を行う（OOF評価 + 後処理）

        処理フロー:
        各Foldごとに個別に後処理:
        ├─ Fold 0: 検証データ → 確率予測 → 最適化 → スコア計算
        ├─ Fold 1: 検証データ → 確率予測 → 最適化 → スコア計算
        └─ Fold 2: 検証データ → 確率予測 → 最適化 → スコア計算
        ↓
        全Fold結合してOOFスコア算出
        
        Args:
            unknown_prob: Unknownラベル（index=11）に割り当てる確率値
                         ※params['threshold']（類似度閾値）とは別物
            max_labels: 最大使用ラベル数
        """
        self.logger.section_start(f"Evaluating {self.n_splits}-Fold CV with Postprocessing (Per-Fold)")

        scores_optimized = []

        # 各foldごとに個別に最適化
        for i_fold in range(self.n_splits):
            self.logger.info(f"\nFold {i_fold}の処理中...")
            
            # データ分割
            _, va = self.create_train_valid_dataset(i_fold)
            df_va = va.copy()
            
            # 学習済みモデル読み込み
            model = self.load_model_cv(i_fold)
            
            # 検証データで確率予測
            va_probs = model.predict_proba(va, split='valid')
            df_va['probs'] = va_probs.tolist()
            
            # 検証データの確率予測に対して最適化を適用しラベルを取得
            self.logger.info(f"  Fold {i_fold}の検証データに後処理を適用中...")
            va_pred_optimized = post_processing.apply_post_processing(df_va)
            
            # スコア
            self.logger.info(f'最適化：{va_pred_optimized}')
            self.logger.info(f'最適化：{va_pred_optimized}')
            va_score_optimized = self.metric(df_va['label_id'].values, va_pred_optimized)
            scores_optimized.append(va_score_optimized)
            # self.logger.info(f"  最適化後の予測分布:\n{va_pred_optimized.value_counts().sort_index()}")
            self.logger.info(f"  Fold {i_fold} スコア: {va_score_optimized:.5f}")
            
        # 結果サマリー
        self.logger.info("\n" + "="*80)
        self.logger.info("後処理ありの評価結果:")
        self.logger.info(f"Foldスコア平均: {np.mean(scores_optimized):.5f} ± {np.std(scores_optimized):.5f}")
        self.logger.info("="*80)
        
        # 結果保存
        self.logger.result_scores(self.run_name + '_optimized', scores_optimized)
        
        return scores_optimized


    def optimize_unknown_threshold(
        self,
        threshold_range: tuple = (0.01, 0.5),
        num_trials: int = 20
    ) -> float:
        """
        Unknown閾値を最適化（検証データでF1スコアを最大化）
        
        Args:
            threshold_range: 探索する閾値の範囲 (min, max)
            num_trials: 試行回数
        
        Returns:
            best_threshold: 最適な閾値
        """
        self.logger.section_start(f"Optimizing Unknown Threshold ({num_trials} trials)")
        
        all_probs = []
        all_labels = []
        
        # 各foldのOOF確率を収集
        for i_fold in range(self.n_splits):
            self.logger.info(f"Fold {i_fold}のOOF確率を取得中...")
            
            # データ分割
            _, va = self.create_train_valid_dataset(i_fold)
            
            # モデル読み込み
            model = self.load_model_cv(i_fold)
            
            # 検証データの確率予測
            va_probs = model.predict_proba(va, split='valid')
            va_labels = va[self.target_col].values
            
            all_probs.append(va_probs)
            all_labels.append(va_labels)
        
        # 各foldのデータを結合
        probs_combined = np.concatenate(all_probs, axis=0)
        labels_combined = np.concatenate(all_labels, axis=0)
        
        self.logger.info(f"OOFデータ: {probs_combined.shape[0]}サンプル, {probs_combined.shape[1]}クラス")
        
        # 最適化実行
        self.logger.info("グリッドサーチで最適閾値を探索")
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_trials)
        best_threshold = threshold_range[0]
        best_f1 = 0.0
        
        for threshold in thresholds:
            # 予測（argmax）
            pred_labels = probs_combined.argmax(axis=1)
            
            # 最大確率を取得
            max_probs = probs_combined.max(axis=1)
            
            # 閾値以下の場合はunknown (-1) に変更
            pred_labels_with_threshold = pred_labels.copy()
            pred_labels_with_threshold[max_probs < threshold] = -1
            
            # F1計算（-1を含めてMacro F1）
            f1 = Metric.macro_f1(labels_combined, pred_labels_with_threshold)
            
            self.logger.info(f"  threshold={threshold:.3f} → F1={f1:.4f} (unknown数: {(pred_labels_with_threshold == -1).sum()})")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.logger.info(f"最適閾値: {best_threshold:.3f} (F1={best_f1:.4f})")
        
        return best_threshold


########### 以下現時点で不使用　 #####################################################################


    # def train_all(self):
    #     """全データでの学習（CV不要の場合）"""
    #     self.logger.section_start("Training on all data")
        
    #     # モデル構築
    #     model = self.build_model(i_fold='all')
        
    #     # 学習
    #     self.logger.info(f"{len(self.df_train):,}サンプルで学習中...")
    #     model.train(self.df_train)
        
    #     # モデル保存
    #     model.save_model()
        
    #     self.logger.section_end("Training completed!")
        
    #     return model


    # def predict_all(self):
    #     """全データ学習済みモデルでの予測"""
    #     self.logger.section_start("Predicting with all-data model")
        
    #     # モデル読み込み
    #     model = self.build_model(i_fold='all')
    #     model.load_model()
        
    #     # 予測
    #     self.logger.info(f"{len(self.df_test)}サンプルを予測中...")
    #     pred = model.predict(self.df_test)
        
    #     # 予測分布
    #     pred_counts = pred['label_id'].value_counts().sort_index()
    #     self.logger.info(f"予測分布:\n{pred_counts}")

    #     submission = pd.DataFrame({'label_id': pred['label_id'].values})
        
    #     return submission



    # def predict_cv(self):
    #     """CV学習済みモデルでの予測（アンサンブル）"""
    #     self.logger.section_start(f"Predicting with CV ensemble ({self.n_splits} folds)")
        
    #     # 各foldで予測
    #     all_predictions = []
        
    #     for i_fold in range(self.n_splits):
    #         self.logger.info(f"Fold {i_fold}で予測中...")
            
    #         # モデル読み込み
    #         model = self.build_model(i_fold)
    #         model.load_model()
            
    #         # 予測
    #         te_pred = model.predict(self.df_test, split='test')
    #         all_predictions.append(te_pred['label_id'].values)
        
    #     # アンサンブル（多数決）
    #     self.logger.info("予測結果をアンサンブル中（多数決）...")
    #     # 各foldの予測長が一致しているか確認
    #     pred_lengths = [len(p) for p in all_predictions]
    #     self.logger.info(f"各foldの予測サンプル数: {pred_lengths}")
    #     if len(set(pred_lengths)) > 1:
    #         raise ValueError(f"Prediction lengths are inconsistent: {pred_lengths}")
        
    #     all_predictions = np.array(all_predictions)  # (n_folds, n_samples)
        
    #     final_predictions = []
    #     for i in range(all_predictions.shape[1]):
    #         fold_preds = all_predictions[:, i]
    #         # 多数決
    #         unique, counts = np.unique(fold_preds, return_counts=True)
    #         final_predictions.append(unique[np.argmax(counts)])
        
    #     final_predictions = np.array(final_predictions)
        
    #     # 結果DataFrame作成
    #     submission = pd.DataFrame({'label_id': final_predictions})
        
    #     # 予測分布
    #     pred_counts = submission['label_id'].value_counts().sort_index()
    #     self.logger.info(f"最終予測分布:\n{pred_counts}")
        
    #     return submission



    # def metric_cv(self):
    #     """CVでの評価を行う（OOF評価、後処理なし）
        
    #     学習済みの各foldモデルで検証データを予測し、全体のOOFスコアを算出
    #     ベースライン性能の確認用
        
    #     Returns:
    #         scores: 各foldのスコアリスト
    #         oof_score: OOF全体のスコア
    #     """
    #     self.logger.section_start(f"Evaluating {self.n_splits}-Fold CV (OOF, no postprocessing)")

    #     scores = []
    #     va_preds_all = []

    #     # fold毎の検証データの予測・評価
    #     for i_fold in range(self.n_splits):
    #         self.logger.info(f"Fold {i_fold}を評価中...")
            
    #         # データ分割
    #         _, va = self.create_train_valid_dataset(i_fold)
            
    #         # 学習済みモデル読み込み
    #         model = self.build_model(i_fold)
    #         model.load_model()
            
    #         # 検証データで予測（元のインデックスを保持）
    #         va_pred = model.predict(va, split='valid')
            
    #         # スコア計算
    #         va_score = self.metric(va[self.target_col].values, va_pred['label_id'].values)
    #         scores.append(va_score)
    #         va_preds_all.append(va_pred)
            
    #         self.logger.info(f"  Fold {i_fold} スコア: {va_score:.5f}")
        
    #     # 全foldの予測を結合してOOFスコア計算
    #     df_va_preds = pd.concat(va_preds_all, axis=0)
        
    #     # インデックスでソートして元の順序に並べる
    #     df_va_preds = df_va_preds.sort_index()
        
    #     # 訓練データと同じ順序でOOFスコア計算
    #     # インデックスが一致することを確認
    #     if not df_va_preds.index.equals(self.df_train.index):
    #         self.logger.info("⚠️ Warning: OOF予測のインデックスが訓練データと一致しません！")
    #         self.logger.info(f"  訓練データ: {len(self.df_train)}行, OOF予測: {len(df_va_preds)}行")
        
    #     oof_score = self.metric(
    #         self.df_train.loc[df_va_preds.index, 'label_id'].values,
    #         df_va_preds['label_id'].values
    #     )

    #     # 結果サマリー
    #     self.logger.info("\n" + "="*80)
    #     self.logger.info(f"OOFスコア (Macro F1): {oof_score:.5f}")
    #     self.logger.info(f"Foldスコア平均: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    #     self.logger.info("="*80)
        
    #     # 結果保存
    #     self.logger.result_scores(self.run_name, scores)
        
    #     # OOF予測結果の保存
    #     path_output = os.path.join(self.out_dir_name, 'va_pred.pkl')
    #     Util.dump_df_pickle(df_va_preds, path_output)
    #     self.logger.info(f'OOF予測結果保存: {path_output}')

    #     return scores, oof_score