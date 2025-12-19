"""
現在のCV戦略の問題点を分析
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')
quarters = train['quarter'].values
labels = train['label_id'].values

print('='*80)
print('【問題1: 現在のGroupKFold (n_splits=5)】')
print('='*80)

gkf = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkf.split(train, labels, groups=quarters)):
    val_quarters = train.iloc[val_idx]['quarter'].unique()
    val_labels = train.iloc[val_idx]['label_id'].unique()
    
    print(f'\nFold {fold}:')
    print(f'  検証Quarters ({len(val_quarters)}個): {sorted(val_quarters)[:5]}...')
    print(f'  検証に含まれる選手: {sorted(val_labels)}')
    
    train_labels = set(train.iloc[train_idx]['label_id'].unique())
    val_labels_set = set(val_labels)
    
    only_train = train_labels - val_labels_set
    only_val = val_labels_set - train_labels
    
    if only_train:
        print(f'  ⚠️ 訓練のみ: 選手{sorted(only_train)} → 検証で予測不可能！')
    if only_val:
        print(f'  ⚠️ 検証のみ: 選手{sorted(only_val)} → 訓練データなし！')

print('\n' + '='*80)
print('【問題の本質】')
print('='*80)
print('- Q1の8 quarters（選手0あり、選手5なし）')
print('- Q2の21 quarters（選手5あり、選手0なし）')
print('- 5-fold CVで分割すると、一部のfoldで選手0または5が片方しか現れない')
print('- → Macro F1で11クラス全てを評価するのに、データが偏っている')

# StratifiedGroupKFoldを試す
print('\n' + '='*80)
print('【解決策: StratifiedGroupKFold】')
print('='*80)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

all_ok = True
for fold, (train_idx, val_idx) in enumerate(sgkf.split(train, labels, groups=quarters)):
    train_labels = set(train.iloc[train_idx]['label_id'].unique())
    val_labels = set(train.iloc[val_idx]['label_id'].unique())
    
    print(f'\nFold {fold}:')
    print(f'  訓練の選手: {sorted(train_labels)}')
    print(f'  検証の選手: {sorted(val_labels)}')
    
    if len(train_labels) != 11 or len(val_labels) != 11:
        print(f'  ⚠️ 問題あり！')
        all_ok = False
    else:
        print(f'  ✅ 全選手が両方に含まれる')

if all_ok:
    print('\n✅ StratifiedGroupKFoldで問題解決！')
else:
    print('\n⚠️ StratifiedGroupKFoldでも完全には解決しない')
    print('→ さらなる工夫が必要（カスタムCV戦略）')
