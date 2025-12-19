# Unknown評価用Runner設計案

## 現状の課題
- Runnerはシンプルになったが、Unknown評価（Q1/Q2分割CV）の機能が削除された
- Unknown評価では特殊な評価指標（Unknown Precision/Recall/F1）が必要
- テストデータの「選手入れ替え」を模擬した評価が必要

## 設計案

### 【案A】Q1/Q2Validatorを作成（推奨）

**コンセプト:**
- Runnerはそのまま使用
- Q1/Q2分割専用のvalidatorクラスを作成
- Unknown評価指標は別途`UnknownMetric`クラスで計算

**実装:**

```python
# src/util.py に追加
class Q1Q2Validator:
    """Q1/Q2クォーター分割用のValidator
    
    - Fold 0: Q2で訓練 → Q1で検証（選手0がunknown）
    - Fold 1: Q1で訓練 → Q2で検証（選手5がunknown）
    """
    def __init__(self, quarter_col='quarter'):
        self.quarter_col = quarter_col
        self.n_splits = 2
    
    def split(self, X, y=None, groups=None):
        """CV分割を生成"""
        quarters = X[self.quarter_col]
        q1_mask = quarters.str.startswith('Q1')
        q2_mask = quarters.str.startswith('Q2')
        
        q1_indices = X[q1_mask].index.values
        q2_indices = X[q2_mask].index.values
        
        # Fold 0: Q2訓練 → Q1検証
        yield q2_indices, q1_indices
        
        # Fold 1: Q1訓練 → Q2検証
        yield q1_indices, q2_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# src/util.py の Metric クラスに追加
class Metric:
    # ... 既存のメソッド ...
    
    @staticmethod
    def unknown_metrics(y_true, y_pred, unknown_player_id):
        """Unknown判定の評価指標を計算
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            unknown_player_id: unknown判定すべき選手ID
        
        Returns:
            dict: precision, recall, f1, known_macro_f1
        """
        is_unknown = (y_true == unknown_player_id)
        pred_unknown = (y_pred == -1)
        
        tp = np.sum(is_unknown & pred_unknown)
        fp = np.sum(~is_unknown & pred_unknown)
        fn = np.sum(is_unknown & ~pred_unknown)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 既知選手のMacro F1
        known_mask = ~is_unknown
        known_macro_f1 = Metric.macro_f1(y_true[known_mask], y_pred[known_mask])
        
        return {
            'unknown_precision': precision,
            'unknown_recall': recall,
            'unknown_f1': f1,
            'known_macro_f1': known_macro_f1
        }
```

**Runnerの修正（カスタムvalidatorサポート）:**

```python
class Runner:
    def __init__(self, run_name, model_cls, params, df_train, df_test, cv_setting, logger):
        # ...
        
        # validatorを直接渡せるようにする
        if 'validator' in cv_setting:
            self.validator = cv_setting['validator']
            self.n_splits = cv_setting.get('n_splits', self.validator.n_splits)
        else:
            self.validator = Validation.create_validator(...)
            self.n_splits = cv_setting.get('n_splits', 3)
```

**Notebookでの使用例:**

```python
# ========== 通常の3-fold CV ==========
runner = Runner(
    run_name='exp_normal_cv',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={
        'n_splits': 3,
        'method': 'stratified_group',
        'group_col': 'quarter'
    },
    logger=logger
)

runner.train_cv()
scores, oof_score = runner.metric_cv()
logger.info(f"通常CV OOF: {oof_score:.5f}")


# ========== Unknown評価用2-fold CV ==========
from src.util import Q1Q2Validator, Metric

# Q1/Q2分割validator
q1q2_validator = Q1Q2Validator(quarter_col='quarter')

runner_unknown = Runner(
    run_name='exp_unknown_cv',
    model_cls=ModelArcFace,
    params=params,
    df_train=df_train,
    df_test=df_test,
    cv_setting={
        'validator': q1q2_validator,  # カスタムvalidatorを渡す
        'n_splits': 2
    },
    logger=logger
)

runner_unknown.train_cv()

# Unknown評価指標の計算
# OOF予測を取得
oof_pred_path = os.path.join(runner_unknown.out_dir_name, 'va_pred.pkl')
oof_df = pd.read_pickle(oof_pred_path)

# Fold 0: 選手0がunknown
fold0_mask = oof_df['quarter'].str.startswith('Q1')
metrics_fold0 = Metric.unknown_metrics(
    oof_df[fold0_mask]['label_id'].values,
    oof_df[fold0_mask]['pred'].values,
    unknown_player_id=0
)

# Fold 1: 選手5がunknown
fold1_mask = oof_df['quarter'].str.startswith('Q2')
metrics_fold1 = Metric.unknown_metrics(
    oof_df[fold1_mask]['label_id'].values,
    oof_df[fold1_mask]['pred'].values,
    unknown_player_id=5
)

logger.info(f"Unknown Fold 0: {metrics_fold0}")
logger.info(f"Unknown Fold 1: {metrics_fold1}")
```

**メリット:**
- Runnerはシンプルなまま
- validatorを差し替えるだけで任意のCV戦略に対応
- 拡張性が高い（他のカスタムCVも簡単に追加可能）

**デメリット:**
- Notebook側で少しコードが増える
- Unknown評価指標の計算を明示的に書く必要がある


---

### 【案B】RunnerUnknownクラスを作成

**コンセプト:**
- Runnerを継承した`RunnerUnknown`クラスを作成
- Unknown評価専用のメソッドを追加

```python
# src/runner_unknown.py
class RunnerUnknown(Runner):
    """Unknown判定評価専用のRunner"""
    
    def __init__(self, run_name, model_cls, params, df_train, df_test, logger):
        # Q1/Q2分割validator自動生成
        q1q2_validator = Q1Q2Validator()
        cv_setting = {'validator': q1q2_validator, 'n_splits': 2}
        
        super().__init__(run_name, model_cls, params, df_train, df_test, cv_setting, logger)
    
    def metric_unknown_cv(self):
        """Unknown評価指標を自動計算"""
        # OOF予測からunknown指標を計算
        pass
```

**メリット:**
- Notebook側がシンプル
- Unknown評価専用の機能を集約

**デメリット:**
- クラスが増える
- 拡張性は案Aより低い


---

## 推奨案

**案A（Q1Q2Validator + カスタムvalidatorサポート）** を推奨します。

理由：
1. Runnerの責務がシンプルに保たれる
2. 他のカスタムCV戦略にも対応できる拡張性
3. validatorの差し替えだけで対応可能
4. テスタビリティが高い

## 次のステップ

1. `src/util.py`に`Q1Q2Validator`クラスを追加
2. `Metric.unknown_metrics()`メソッドを追加
3. `Runner.__init__()`を修正してカスタムvalidatorをサポート
4. Notebookで動作確認

実装しますか？
