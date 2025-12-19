"""
複数のRunnerをまとめてアンサンブル予測を行うクラス
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Literal

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.runner import Runner


class RunnerEnsemble:
    """複数のRunnerをまとめてアンサンブル予測"""
    
    def __init__(self, runners: List[Runner], logger):
        """
        Args:
            runners: Runnerインスタンスのリスト
            logger: ロガー
        """
        if not runners:
            raise ValueError("runners リストが空です")
        
        self.runners = runners
        self.logger = logger
        self.df_test = runners[0].df_test  # 全Runnerで共通と仮定
        
        # 総モデル数の確認
        self.total_models = sum(runner.n_splits for runner in runners)
        self.logger.info(f"RunnerEnsemble初期化: {len(runners)}個のRunner, 合計{self.total_models}モデル")
        for i, runner in enumerate(runners):
            self.logger.info(f"  Runner {i}: {runner.run_name} ({runner.n_splits}-fold)")
    
    
    def predict_ensemble(self, method: Literal['voting', 'soft'] = 'voting') -> pd.DataFrame:
        """全Runnerのモデルでアンサンブル予測
        
        Args:
            method: アンサンブル手法
                - 'voting': 多数決（ハードボーティング）
                - 'soft': 確率の平均（ソフトボーティング、未実装）
        
        Returns:
            submission: 提出用DataFrame
        """
        self.logger.section_start(f"Ensemble Prediction ({self.total_models} models)")
        
        if method == 'soft':
            raise NotImplementedError("soft votingは未実装です")
        
        all_predictions = []
        model_info = []
        
        # 各Runnerの各foldで予測
        for runner_idx, runner in enumerate(self.runners):
            self.logger.info(f"\nRunner {runner_idx}: {runner.run_name}")
            
            for i_fold in range(runner.n_splits):
                self.logger.info(f"  Fold {i_fold}で予測中...")
                
                # モデル読み込み
                model = runner.build_model(i_fold)
                model.load_model()
                
                # 予測
                pred = model.predict(self.df_test, split='test')
                all_predictions.append(pred['label_id'].values)
                
                model_info.append({
                    'runner_idx': runner_idx,
                    'runner_name': runner.run_name,
                    'fold': i_fold
                })
        
        # アンサンブル（多数決）
        self.logger.info(f"\n{self.total_models}モデルでアンサンブル中（多数決）...")
        
        # 各予測長が一致しているか確認
        pred_lengths = [len(p) for p in all_predictions]
        if len(set(pred_lengths)) > 1:
            raise ValueError(f"予測長が一致しません: {pred_lengths}")
        
        all_predictions = np.array(all_predictions)  # (n_models, n_samples)
        
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
        self.logger.info(f"\n最終予測分布:")
        for label, count in pred_counts.items():
            label_name = f"選手{label}" if label != -1 else "Unknown"
            self.logger.info(f"  {label_name}: {count:,}サンプル")
        
        self.logger.section_end(f"アンサンブル予測完了")
        
        return submission
    
    
    def save_submission(self, submission: pd.DataFrame, 
                       filename: str = None) -> str:
        """提出ファイル保存
        
        Args:
            submission: 提出用DataFrame
            filename: ファイル名（省略時は自動生成）
        
        Returns:
            保存先パス
        """
        from datetime import datetime
        
        # ファイル名生成
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            runner_names = '_'.join([r.run_name.split('_')[-1] for r in self.runners[:2]])  # 最初の2つのみ
            filename = f"ensemble_{runner_names}_{self.total_models}models_{timestamp}.csv"
        
        # 保存
        output_path = os.path.join(DIR_SUBMISSIONS, filename)
        os.makedirs(DIR_SUBMISSIONS, exist_ok=True)
        
        # label_idのみをヘッダーなしで保存
        submission.to_csv(output_path, index=False, header=False)
        
        self.logger.info(f"提出ファイル保存: {output_path}")
        self.logger.info(f"  総サンプル数: {len(submission):,}")
        
        return output_path
    
    
    def analyze_agreement(self) -> pd.DataFrame:
        """モデル間の予測一致度を分析
        
        Returns:
            分析結果のDataFrame
        """
        self.logger.info("モデル間の予測一致度を分析中...")
        
        all_predictions = []
        
        # 各Runnerの各foldで予測
        for runner in self.runners:
            for i_fold in range(runner.n_splits):
                model = runner.build_model(i_fold)
                model.load_model()
                pred = model.predict(self.df_test, split='test')
                all_predictions.append(pred['label_id'].values)
        
        all_predictions = np.array(all_predictions)  # (n_models, n_samples)
        
        # 一致度を計算
        agreement_scores = []
        for i in range(all_predictions.shape[1]):
            fold_preds = all_predictions[:, i]
            # 最頻値の出現回数 / 総モデル数
            unique, counts = np.unique(fold_preds, return_counts=True)
            max_count = counts.max()
            agreement = max_count / len(fold_preds)
            agreement_scores.append(agreement)
        
        agreement_scores = np.array(agreement_scores)
        
        # サマリー統計
        self.logger.info(f"\n予測一致度:")
        self.logger.info(f"  平均: {agreement_scores.mean():.4f}")
        self.logger.info(f"  中央値: {np.median(agreement_scores):.4f}")
        self.logger.info(f"  最小: {agreement_scores.min():.4f}")
        self.logger.info(f"  最大: {agreement_scores.max():.4f}")
        
        # 一致度の分布
        bins = [0, 0.4, 0.6, 0.8, 1.0]
        labels = ['低(~0.4)', '中(0.4~0.6)', '高(0.6~0.8)', '非常に高(0.8~1.0)']
        
        self.logger.info(f"\n一致度の分布:")
        for i in range(len(bins)-1):
            count = np.sum((agreement_scores > bins[i]) & (agreement_scores <= bins[i+1]))
            self.logger.info(f"  {labels[i]}: {count:,}サンプル ({count/len(agreement_scores)*100:.1f}%)")
        
        return pd.DataFrame({
            'agreement_score': agreement_scores,
            'all_predictions': list(all_predictions.T)
        })
