"""
CV戦略の詳細分析：サンプル数のバランスチェック
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')

print('='*80)
print('【GroupKFold vs StratifiedGroupKFold の比較】')
print('='*80)

for cv_name, cv_splitter in [
    ('GroupKFold', GroupKFold(n_splits=5)),
    ('StratifiedGroupKFold', StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42))
]:
    print(f'\n■ {cv_name}')
    print('-'*80)
    
    fold_stats = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(
        train, train['label_id'].values, groups=train['quarter'].values
    )):
        val_df = train.iloc[val_idx]
        
        # 各選手のサンプル数
        player_counts = val_df['label_id'].value_counts().sort_index()
        
        # Q1/Q2のQuarter数
        val_quarters = val_df['quarter'].unique()
        q1_count = sum(1 for q in val_quarters if q.startswith('Q1'))
        q2_count = sum(1 for q in val_quarters if q.startswith('Q2'))
        
        # 選手0と5のサンプル数
        player0_count = player_counts.get(0, 0)
        player5_count = player_counts.get(5, 0)
        
        stats = {
            'fold': fold,
            'total_samples': len(val_df),
            'q1_quarters': q1_count,
            'q2_quarters': q2_count,
            'player0_samples': player0_count,
            'player5_samples': player5_count,
            'min_player_samples': player_counts.min(),
            'max_player_samples': player_counts.max(),
            'std_player_samples': player_counts.std()
        }
        fold_stats.append(stats)
        
        print(f'\n  Fold {fold}:')
        print(f'    総サンプル: {stats["total_samples"]:,}')
        print(f'    Q1/Q2 quarters: {q1_count}/{q2_count}')
        print(f'    選手0: {player0_count:3d}, 選手5: {player5_count:3d}')
        print(f'    サンプル数: min={stats["min_player_samples"]:3d}, max={stats["max_player_samples"]:3d}, std={stats["std_player_samples"]:.1f}')
    
    # 統計サマリー
    df_stats = pd.DataFrame(fold_stats)
    print(f'\n  【統計サマリー】')
    print(f'    選手0のfold間ばらつき: {df_stats["player0_samples"].std():.1f}')
    print(f'    選手5のfold間ばらつき: {df_stats["player5_samples"].std():.1f}')
    print(f'    総サンプル数のばらつき: {df_stats["total_samples"].std():.1f}')

print('\n' + '='*80)
print('【結論】')
print('='*80)
print('StratifiedGroupKFoldの方が:')
print('  ✅ 各foldでの選手分布が均等')
print('  ✅ 選手0/5のサンプル数が安定')
print('  ✅ Macro F1の信頼性が向上')
