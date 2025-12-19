"""
最適CV戦略の設計と検証
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')

print('='*80)
print('【提案: 改良版CV戦略】')
print('='*80)

# 選手0と5の問題を解決するため、Q1/Q2を均等に配分するカスタムCV
quarters = sorted(train['quarter'].unique())
q1_quarters = [q for q in quarters if q.startswith('Q1')]  # 8個
q2_quarters = [q for q in quarters if q.startswith('Q2')]  # 21個

print(f'\nQ1 quarters: {len(q1_quarters)}個')
print(f'Q2 quarters: {len(q2_quarters)}個')

# 各foldにQ1を1-2個、Q2を4-5個配分
# Q1: 8個 → 5foldで [2,2,1,1,2]個ずつ
# Q2: 21個 → 5foldで [4,4,4,4,5]個ずつ

print('\n【案1: 手動で均等配分】')
np.random.seed(42)
q1_shuffled = np.random.permutation(q1_quarters)
q2_shuffled = np.random.permutation(q2_quarters)

fold_assignment = {
    0: list(q1_shuffled[0:2]) + list(q2_shuffled[0:4]),   # 2 Q1 + 4 Q2
    1: list(q1_shuffled[2:4]) + list(q2_shuffled[4:8]),   # 2 Q1 + 4 Q2
    2: list(q1_shuffled[4:5]) + list(q2_shuffled[8:12]),  # 1 Q1 + 4 Q2
    3: list(q1_shuffled[5:6]) + list(q2_shuffled[12:16]), # 1 Q1 + 4 Q2
    4: list(q1_shuffled[6:8]) + list(q2_shuffled[16:21]), # 2 Q1 + 5 Q2
}

for fold, val_quarters in fold_assignment.items():
    val_df = train[train['quarter'].isin(val_quarters)]
    player_counts = val_df['label_id'].value_counts().sort_index()
    
    q1_count = sum(1 for q in val_quarters if q.startswith('Q1'))
    q2_count = sum(1 for q in val_quarters if q.startswith('Q2'))
    
    print(f'\nFold {fold}:')
    print(f'  Q1/Q2: {q1_count}/{q2_count} quarters')
    print(f'  選手0: {player_counts.get(0, 0):3d}, 選手5: {player_counts.get(5, 0):3d}')
    print(f'  総サンプル: {len(val_df):,}')
    print(f'  サンプル数: min={player_counts.min():3d}, max={player_counts.max():3d}')

# 統計
player0_counts = []
player5_counts = []
total_counts = []

for val_quarters in fold_assignment.values():
    val_df = train[train['quarter'].isin(val_quarters)]
    player_counts = val_df['label_id'].value_counts()
    player0_counts.append(player_counts.get(0, 0))
    player5_counts.append(player_counts.get(5, 0))
    total_counts.append(len(val_df))

print('\n【均等配分の統計】')
print(f'  選手0のばらつき: std={np.std(player0_counts):.1f}')
print(f'  選手5のばらつき: std={np.std(player5_counts):.1f}')
print(f'  総サンプル数のばらつき: std={np.std(total_counts):.1f}')

print('\n' + '='*80)
print('【比較: 現状 vs 改良版】')
print('='*80)
print('                      GroupKFold  | 手動均等配分')
print('選手0のばらつき(std)      41.8    |    %.1f' % np.std(player0_counts))
print('選手5のばらつき(std)      15.3    |    %.1f' % np.std(player5_counts))
print('総サンプル数のばらつき    346.8   |    %.1f' % np.std(total_counts))

if np.std(player0_counts) < 41.8:
    print('\n✅ 改良版の方が選手0のばらつきが小さい → 推奨')
else:
    print('\n⚠️ 改良版でも改善が限定的 → 現状のStratifiedGroupKFoldで十分')
