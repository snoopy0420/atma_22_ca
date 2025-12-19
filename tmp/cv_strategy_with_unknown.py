"""
コンペ特性を考慮した最適CV戦略の設計

重要な発見：
- 訓練データの選手0/5の入れ替えは、テストデータの状況を模している
- テストデータでも選手の入れ替えが発生（どの選手がいつ入れ替わるか不明）
- → CV戦略では「見たことのない選手をunknownと判定する能力」をテストすべき
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')

print('='*80)
print('【コンペ特性を考慮したCV戦略の再評価】')
print('='*80)

# 選手0/5の入れ替えパターンを確認
q1_quarters = [q for q in train['quarter'].unique() if q.startswith('Q1')]
q2_quarters = [q for q in train['quarter'].unique() if q.startswith('Q2')]

print('\n【訓練データの選手入れ替えパターン】')
print(f'Q1 ({len(q1_quarters)}個): 選手0あり、選手5なし')
print(f'Q2 ({len(q2_quarters)}個): 選手5あり、選手0なし')
print('\n→ これはテストデータでの「途中の選手入れ替え」を模している！')

print('\n' + '='*80)
print('【CV戦略の3つの選択肢】')
print('='*80)

# オプション1: StratifiedGroupKFold（推奨）
print('\n■ オプション1: StratifiedGroupKFold（全選手を均等に配分）')
print('-'*80)
print('【メリット】')
print('  ✅ 各foldで全選手が訓練・検証の両方に含まれる')
print('  ✅ Macro F1の計算が安定')
print('  ✅ 11クラス全てでモデルの性能を評価できる')
print('【デメリット】')
print('  ⚠️ 「見たことのない選手」のunknown判定能力をテストできない')
print('  ⚠️ テストデータの状況（選手入れ替え）を模していない')
print('【推奨度】★★★☆☆')
print('【適用場面】通常の学習・モデル選択時')

# オプション2: Leave-One-Out CV（選手0または5を完全に除外）
print('\n■ オプション2: Leave-One-Player-Out CV（unknown判定特化）')
print('-'*80)
print('【戦略】')
print('  - Fold A: Q2のみ使用（選手5を学習、選手0を検証でunknown判定）')
print('  - Fold B: Q1のみ使用（選手0を学習、選手5を検証でunknown判定）')
print('【メリット】')
print('  ✅ 「見たことのない選手」のunknown判定能力を直接テスト')
print('  ✅ テストデータの状況に最も近い')
print('  ✅ 閾値チューニングに最適')
print('【デメリット】')
print('  ⚠️ 2-fold CVのみ（統計的に不安定）')
print('  ⚠️ 通常のMacro F1評価ができない（unknownクラスのF1のみ）')
print('【推奨度】★★★★☆')
print('【適用場面】閾値チューニング専用')

# オプション3: ハイブリッド戦略
print('\n■ オプション3: ハイブリッド戦略（推奨）')
print('-'*80)
print('【戦略】')
print('  1. 通常の5-fold CV: StratifiedGroupKFoldで全選手を使用')
print('     → モデルの基本性能評価・ハイパラ調整')
print('  2. Unknown評価用2-fold: Q1/Q2を完全分離')
print('     → unknown判定の閾値チューニング')
print('【メリット】')
print('  ✅ 両方の目的を達成')
print('  ✅ 最も現実的で包括的')
print('【デメリット】')
print('  ⚠️ 実装が複雑')
print('【推奨度】★★★★★')
print('【適用場面】最終調整・提出前')

# 具体的なunknown評価CVの実装
print('\n' + '='*80)
print('【Unknown判定評価用CVの実装】')
print('='*80)

for fold_name, train_q, val_q in [
    ('Fold A (選手5→選手0)', q2_quarters, q1_quarters),
    ('Fold B (選手0→選手5)', q1_quarters, q2_quarters)
]:
    train_df = train[train['quarter'].isin(train_q)]
    val_df = train[train['quarter'].isin(val_q)]
    
    train_players = sorted(train_df['label_id'].unique())
    val_players = sorted(val_df['label_id'].unique())
    unknown_players = set(val_players) - set(train_players)
    
    print(f'\n{fold_name}:')
    print(f'  訓練: {len(train_df):5,}サンプル, 選手{train_players}')
    print(f'  検証: {len(val_df):5,}サンプル, 選手{val_players}')
    print(f'  Unknown判定すべき選手: {sorted(unknown_players)}')

print('\n' + '='*80)
print('【最終推奨】')
print('='*80)
print('1. 通常学習: StratifiedGroupKFold (5-fold)')
print('   → cv_setting = {"method": "stratified_group", "group_col": "quarter", "n_splits": 5}')
print('')
print('2. 閾値チューニング: Q1/Q2分離CV (2-fold)')
print('   → 別途実装が必要（Notebookで実行）')
print('')
print('3. 評価指標:')
print('   - 5-fold CV: 通常のMacro F1（11クラス）')
print('   - 2-fold CV: Unknown判定のRecall/Precision')
print('               + 既知選手のMacro F1')
