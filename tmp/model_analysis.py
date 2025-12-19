"""
モデル性能分析ツール
混同行列、クラス別F1、誤分類パターンの可視化
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from typing import Optional

sys.path.append(os.path.abspath('..'))
from configs.config import *


class ModelAnalyzer:
    """モデル性能の詳細分析クラス"""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def log(self, msg):
        """ログ出力"""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def analyze_predictions(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          output_dir: str,
                          prefix: str = ""):
        """予測結果の包括的な分析
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            output_dir: 出力ディレクトリ
            prefix: ファイル名プレフィックス
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 基本統計
        self._basic_statistics(y_true, y_pred)
        
        # 2. クラス別F1スコア
        self._class_wise_f1(y_true, y_pred, output_dir, prefix)
        
        # 3. 混同行列
        self._plot_confusion_matrix(y_true, y_pred, output_dir, prefix)
        
        # 4. 誤分類パターン分析
        self._error_analysis(y_true, y_pred, output_dir, prefix)
        
        # 5. unknown予測の分析
        self._unknown_analysis(y_true, y_pred)
    
    def _basic_statistics(self, y_true, y_pred):
        """基本統計情報"""
        self.log("\n" + "="*80)
        self.log("📊 予測統計")
        self.log("="*80)
        
        # 全体精度
        accuracy = (y_true == y_pred).mean()
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        self.log(f"総サンプル数: {len(y_true):,}")
        self.log(f"正解率: {accuracy:.4f}")
        self.log(f"Macro F1: {macro_f1:.4f}")
        
        # クラス分布
        self.log("\n【正解ラベルの分布】")
        unique, counts = np.unique(y_true, return_counts=True)
        for label, count in zip(unique, counts):
            self.log(f"  クラス {label:2d}: {count:5d} ({count/len(y_true)*100:5.2f}%)")
        
        self.log("\n【予測ラベルの分布】")
        unique, counts = np.unique(y_pred, return_counts=True)
        for label, count in zip(unique, counts):
            self.log(f"  クラス {label:2d}: {count:5d} ({count/len(y_pred)*100:5.2f}%)")
    
    def _class_wise_f1(self, y_true, y_pred, output_dir, prefix):
        """クラス別F1スコア"""
        self.log("\n" + "="*80)
        self.log("📈 クラス別性能")
        self.log("="*80)
        
        # クラス別F1を計算
        labels = np.unique(np.concatenate([y_true, y_pred]))
        report = classification_report(
            y_true, y_pred, 
            labels=labels,
            target_names=[f"Class {l}" for l in labels],
            output_dict=True,
            zero_division=0
        )
        
        # DataFrame化して保存
        df_report = pd.DataFrame(report).transpose()
        csv_path = os.path.join(output_dir, f"{prefix}class_report.csv")
        df_report.to_csv(csv_path)
        self.log(f"クラス別レポート保存: {csv_path}")
        
        # F1スコアの低いクラスTOP5を表示
        class_f1 = {label: report[f"Class {label}"]["f1-score"] 
                    for label in labels if f"Class {label}" in report}
        sorted_f1 = sorted(class_f1.items(), key=lambda x: x[1])
        
        self.log("\n【F1スコアが低いクラス TOP5】")
        for label, f1 in sorted_f1[:5]:
            support = report[f"Class {label}"]["support"]
            self.log(f"  クラス {label:2d}: F1={f1:.4f} (サンプル数: {support})")
        
        self.log("\n【F1スコアが高いクラス TOP5】")
        for label, f1 in sorted_f1[-5:][::-1]:
            support = report[f"Class {label}"]["support"]
            self.log(f"  クラス {label:2d}: F1={f1:.4f} (サンプル数: {support})")
    
    def _plot_confusion_matrix(self, y_true, y_pred, output_dir, prefix):
        """混同行列の可視化"""
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # 正規化版も作成
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # ゼロ除算対策
        
        # プロット
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 生の混同行列
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('True', fontsize=12)
        
        # 正規化版
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('True', fontsize=12)
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{prefix}confusion_matrix.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"\n混同行列保存: {fig_path}")
    
    def _error_analysis(self, y_true, y_pred, output_dir, prefix):
        """誤分類パターン分析"""
        self.log("\n" + "="*80)
        self.log("🔍 誤分類パターン分析")
        self.log("="*80)
        
        # 誤分類のみ抽出
        errors = y_true != y_pred
        if errors.sum() == 0:
            self.log("誤分類なし（完璧な予測）")
            return
        
        self.log(f"誤分類数: {errors.sum():,} / {len(y_true):,} ({errors.mean()*100:.2f}%)")
        
        # 誤分類ペアの頻度
        error_pairs = list(zip(y_true[errors], y_pred[errors]))
        from collections import Counter
        pair_counts = Counter(error_pairs)
        
        self.log("\n【頻出誤分類パターン TOP10】")
        for (true_label, pred_label), count in pair_counts.most_common(10):
            self.log(f"  {true_label:2d} → {pred_label:2d}: {count:4d}回")
        
        # 誤分類ペアをDataFrameで保存
        df_errors = pd.DataFrame({
            'true_label': y_true[errors],
            'pred_label': y_pred[errors]
        })
        csv_path = os.path.join(output_dir, f"{prefix}error_pairs.csv")
        df_errors['true_label'].value_counts().to_csv(csv_path.replace('.csv', '_by_true.csv'))
        df_errors['pred_label'].value_counts().to_csv(csv_path.replace('.csv', '_by_pred.csv'))
        
        self.log(f"誤分類データ保存: {csv_path}")
    
    def _unknown_analysis(self, y_true, y_pred):
        """unknown (-1) 予測の分析"""
        self.log("\n" + "="*80)
        self.log("❓ unknown予測の分析")
        self.log("="*80)
        
        # unknown予測
        pred_unknown = y_pred == -1
        true_unknown = y_true == -1
        
        self.log(f"unknown予測数: {pred_unknown.sum():,} ({pred_unknown.mean()*100:.2f}%)")
        self.log(f"真のunknown数: {true_unknown.sum():,} ({true_unknown.mean()*100:.2f}%)")
        
        if pred_unknown.sum() > 0:
            # unknown予測時の正解ラベル分布
            self.log("\n【unknown予測時の真のラベル分布】")
            true_when_pred_unknown = y_true[pred_unknown]
            unique, counts = np.unique(true_when_pred_unknown, return_counts=True)
            for label, count in zip(unique, counts):
                self.log(f"  クラス {label:2d}: {count:4d} ({count/pred_unknown.sum()*100:5.2f}%)")
        
        if true_unknown.sum() > 0:
            # 真のunknown時の予測ラベル分布
            self.log("\n【真のunknownに対する予測分布】")
            pred_when_true_unknown = y_pred[true_unknown]
            unique, counts = np.unique(pred_when_true_unknown, return_counts=True)
            for label, count in zip(unique, counts):
                self.log(f"  クラス {label:2d}: {count:4d} ({count/true_unknown.sum()*100:5.2f}%)")


def analyze_oof_predictions(oof_path: str, train_df: pd.DataFrame, logger=None):
    """OOF予測結果を分析
    
    Args:
        oof_path: OOF予測結果のpklファイルパス
        train_df: 訓練データ（正解ラベル含む）
        logger: ロガー
    """
    from src.util import Util
    
    # OOF予測読み込み
    df_oof = Util.load_df_pickle(oof_path)
    
    # 正解ラベルとマージ
    df_merged = train_df.loc[df_oof.index, ['label_id']].copy()
    df_merged['pred'] = df_oof['label_id'].values
    
    # 分析実行
    analyzer = ModelAnalyzer(logger)
    output_dir = os.path.dirname(oof_path)
    analyzer.analyze_predictions(
        y_true=df_merged['label_id'].values,
        y_pred=df_merged['pred'].values,
        output_dir=output_dir,
        prefix="oof_"
    )


if __name__ == "__main__":
    # 使用例
    import pandas as pd
    from src.util import Logger
    
    logger = Logger()
    
    # 訓練データ読み込み
    train_df = pd.read_csv(f'{DIR_INPUT}/atmaCup22_2nd_meta/train_meta.csv')
    
    # 最新のOOF予測を分析
    import glob
    oof_files = glob.glob(f'{DIR_MODEL}/*/va_pred.pkl')
    if oof_files:
        latest_oof = max(oof_files, key=os.path.getmtime)
        logger.info(f"最新OOF予測を分析: {latest_oof}")
        analyze_oof_predictions(latest_oof, train_df, logger)
    else:
        logger.warning("OOF予測ファイルが見つかりません")
