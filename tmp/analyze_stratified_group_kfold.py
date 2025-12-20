"""
StratifiedGroupKFoldの詳細分析
このコンペでの挙動を確認
"""
import sys
sys.path.append('/workspace/atma_22_ca/')

import pandas as pd
import numpy as np
from src.util import Validation

# データ読み込み
df_train = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')
df_train['group'] = df_train['quarter']

print("="*80)
print("StratifiedGroupKFold の詳細分析")
print("="*80)

# ========================================
# 1. データの基本統計
# ========================================
print("\n【1. データ構造】")
print(f"総サンプル数: {len(df_train):,}")
print(f"選手数: {df_train['label_id'].nunique()}")
print(f"Quarter数: {df_train['quarter'].nunique()}")

# 選手ごとのサンプル数
player_counts = df_train['label_id'].value_counts().sort_index()
print(f"\n選手ごとのサンプル数:")
for player_id, count in player_counts.items():
    percentage = count / len(df_train) * 100
    print(f"  選手{player_id}: {count:,}サンプル ({percentage:.2f}%)")

# Quarterごとの選手分布
print(f"\n【重要】選手0と5の分布:")
player0_quarters = df_train[df_train['label_id'] == 0]['quarter'].unique()
player5_quarters = df_train[df_train['label_id'] == 5]['quarter'].unique()
print(f"  選手0が登場するquarter: {sorted(player0_quarters)}")
print(f"  選手5が登場するquarter: {sorted(player5_quarters)}")

all_quarters = sorted(df_train['quarter'].unique())
q1_quarters = [q for q in all_quarters if str(q).startswith('Q1')]
q2_quarters = [q for q in all_quarters if str(q).startswith('Q2')]
print(f"\nQuarter分類:")
print(f"  Q1 quarters ({len(q1_quarters)}個): {q1_quarters}")
print(f"  Q2 quarters ({len(q2_quarters)}個): {q2_quarters}")

# ========================================
# 2. StratifiedGroupKFoldの動作
# ========================================
print("\n" + "="*80)
print("【2. StratifiedGroupKFold (3-fold) の分割結果】")
print("="*80)

validator = Validation.create_validator(
    method='stratified_group',
    n_splits=3,
    shuffle=True,
    random_state=42
)

y = df_train['label_id'].values
groups = df_train['quarter'].values

fold_summaries = []

for fold_idx, (tr_idx, va_idx) in enumerate(validator.split(df_train, y, groups)):
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx}")
    print(f"{'='*80}")
    
    # 訓練・検証データ
    tr_df = df_train.iloc[tr_idx]
    va_df = df_train.iloc[va_idx]
    
    # サンプル数
    print(f"\nサンプル数:")
    print(f"  訓練: {len(tr_df):,} ({len(tr_df)/len(df_train)*100:.1f}%)")
    print(f"  検証: {len(va_df):,} ({len(va_df)/len(df_train)*100:.1f}%)")
    
    # Quarter分布
    tr_quarters = sorted(tr_df['quarter'].unique())
    va_quarters = sorted(va_df['quarter'].unique())
    
    print(f"\nQuarter分布:")
    print(f"  訓練: {len(tr_quarters)}個 quarters")
    print(f"    Q1: {[q for q in tr_quarters if str(q).startswith('Q1')]}")
    print(f"    Q2: {[q for q in tr_quarters if str(q).startswith('Q2')]}")
    print(f"  検証: {len(va_quarters)}個 quarters")
    print(f"    Q1: {[q for q in va_quarters if str(q).startswith('Q1')]}")
    print(f"    Q2: {[q for q in va_quarters if str(q).startswith('Q2')]}")
    
    # リークチェック
    overlap = set(tr_quarters) & set(va_quarters)
    if overlap:
        print(f"  ⚠️ リーク検出: {overlap}")
    else:
        print(f"  ✅ リークなし")
    
    # 選手分布
    tr_player_counts = tr_df['label_id'].value_counts().sort_index()
    va_player_counts = va_df['label_id'].value_counts().sort_index()
    
    print(f"\n選手分布 (訓練):")
    for player_id in range(11):
        tr_count = tr_player_counts.get(player_id, 0)
        tr_pct = tr_count / len(tr_df) * 100 if len(tr_df) > 0 else 0
        print(f"  選手{player_id}: {tr_count:,}サンプル ({tr_pct:.2f}%)")
    
    print(f"\n選手分布 (検証):")
    for player_id in range(11):
        va_count = va_player_counts.get(player_id, 0)
        va_pct = va_count / len(va_df) * 100 if len(va_df) > 0 else 0
        print(f"  選手{player_id}: {va_count:,}サンプル ({va_pct:.2f}%)")
    
    # ⚠️ 重要: 選手0/5の有無
    has_player0_train = 0 in tr_player_counts.index
    has_player5_train = 5 in tr_player_counts.index
    has_player0_valid = 0 in va_player_counts.index
    has_player5_valid = 5 in va_player_counts.index
    
    print(f"\n【重要】選手0/5の存在:")
    print(f"  訓練: 選手0={'○' if has_player0_train else '✕'}, 選手5={'○' if has_player5_train else '✕'}")
    print(f"  検証: 選手0={'○' if has_player0_valid else '✕'}, 選手5={'○' if has_player5_valid else '✕'}")
    
    # サマリー保存
    fold_summaries.append({
        'fold': fold_idx,
        'train_samples': len(tr_df),
        'valid_samples': len(va_df),
        'train_quarters': len(tr_quarters),
        'valid_quarters': len(va_quarters),
        'train_has_player0': has_player0_train,
        'train_has_player5': has_player5_train,
        'valid_has_player0': has_player0_valid,
        'valid_has_player5': has_player5_valid,
        'train_player0_samples': tr_player_counts.get(0, 0),
        'train_player5_samples': tr_player_counts.get(5, 0),
        'valid_player0_samples': va_player_counts.get(0, 0),
        'valid_player5_samples': va_player_counts.get(5, 0),
    })

# ========================================
# 3. サマリーテーブル
# ========================================
print("\n" + "="*80)
print("【3. 全Foldサマリー】")
print("="*80)

df_summary = pd.DataFrame(fold_summaries)
print("\n各Foldの統計:")
print(df_summary[['fold', 'train_samples', 'valid_samples', 'train_quarters', 'valid_quarters']])

print("\n選手0/5の分布:")
print(df_summary[['fold', 'train_has_player0', 'train_has_player5', 'valid_has_player0', 'valid_has_player5']])

# ========================================
# 4. 問題点の指摘
# ========================================
print("\n" + "="*80)
print("【4. 問題点と考察】")
print("="*80)

# 問題1: 選手0/5が全foldで訓練に含まれるか？
all_folds_have_player0 = all(df_summary['train_has_player0'])
all_folds_have_player5 = all(df_summary['train_has_player5'])

print("\n問題1: 全foldで選手0/5を学習できるか？")
if all_folds_have_player0:
    print("  ✅ 選手0: 全foldの訓練データに含まれる")
else:
    missing_folds = df_summary[~df_summary['train_has_player0']]['fold'].tolist()
    print(f"  ❌ 選手0: Fold {missing_folds}の訓練データに含まれない")
    print(f"     → これらのfoldでは選手0を予測できない！")

if all_folds_have_player5:
    print("  ✅ 選手5: 全foldの訓練データに含まれる")
else:
    missing_folds = df_summary[~df_summary['train_has_player5']]['fold'].tolist()
    print(f"  ❌ 選手5: Fold {missing_folds}の訓練データに含まれない")
    print(f"     → これらのfoldでは選手5を予測できない！")

# 問題2: 選手0/5のバランス
print("\n問題2: 選手0/5のサンプル数バランス")
for fold_idx in range(3):
    row = df_summary.iloc[fold_idx]
    total_train = row['train_samples']
    p0_train = row['train_player0_samples']
    p5_train = row['train_player5_samples']
    p0_pct = p0_train / total_train * 100 if total_train > 0 else 0
    p5_pct = p5_train / total_train * 100 if total_train > 0 else 0
    
    print(f"  Fold {fold_idx}:")
    print(f"    選手0: {p0_train:,}サンプル ({p0_pct:.2f}%)")
    print(f"    選手5: {p5_train:,}サンプル ({p5_pct:.2f}%)")

# 問題3: 層化の効果
print("\n問題3: 層化の効果（選手分布の均等性）")
player_distribution_std = []
for player_id in range(11):
    fold_percentages = []
    for fold_idx in range(3):
        tr_df = df_train.iloc[list(validator.split(df_train, y, groups))[fold_idx][0]]
        count = (tr_df['label_id'] == player_id).sum()
        percentage = count / len(tr_df) * 100 if len(tr_df) > 0 else 0
        fold_percentages.append(percentage)
    
    std = np.std(fold_percentages)
    player_distribution_std.append((player_id, std))
    print(f"  選手{player_id}: 標準偏差={std:.3f}% (Fold間のばらつき)")

avg_std = np.mean([std for _, std in player_distribution_std])
print(f"\n  平均標準偏差: {avg_std:.3f}%")
print(f"  → 小さいほど層化が効いている")

# ========================================
# 5. 推奨事項
# ========================================
print("\n" + "="*80)
print("【5. 推奨事項】")
print("="*80)

if not all_folds_have_player0 or not all_folds_have_player5:
    print("\n⚠️ 【重大な問題】")
    print("  選手0または5が一部のfoldで学習されません。")
    print("  これは、quarterをグループにした場合の必然的な結果です。")
    print("\n  対策:")
    print("  1. より細かい粒度のグループ列を使用（session等）")
    print("  2. Q1/Q2を意図的に混ぜる（ただしリークのリスク）")
    print("  3. 現状を受け入れる（一部のfoldで選手0/5の精度が低い）")
else:
    print("\n✅ 全foldで選手0-10を学習できます")
    print("  この設定で問題ありません。")

print(f"\n層化の効果: 平均標準偏差={avg_std:.3f}%")
if avg_std < 1.0:
    print("  ✅ 層化が効いています（選手分布が均等）")
elif avg_std < 2.0:
    print("  ○ 層化はある程度効いています")
else:
    print("  ⚠️ 層化の効果が弱いです（選手分布に偏り）")

print("\n" + "="*80)
print("分析完了")
print("="*80)
