"""
CV戦略分析スクリプト
データリークを防ぐための最適なグループ化戦略を検討
"""
import pandas as pd
import numpy as np

# データ読み込み
train_df = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/train_meta.csv')
test_df = pd.read_csv('data/raw/input/atmaCup22_2nd_meta/test_meta.csv')

print("="*80)
print("📊 データ構造分析")
print("="*80)

print(f"\n【訓練データ】")
print(f"  総サンプル数: {len(train_df):,}")
print(f"  選手数: {train_df['label_id'].nunique()} players")
print(f"  クオーター数: {train_df['quarter'].nunique()}")
print(f"  アングル: {train_df['angle'].unique()}")

print(f"\n【テストデータ】")
print(f"  総サンプル数: {len(test_df):,}")
print(f"  クオーター数: {test_df['quarter'].nunique()}")
print(f"  アングル: {test_df['angle'].unique()}")

print("\n" + "="*80)
print("🔍 グルーピング候補の分析")
print("="*80)

# 候補1: quarter
print("\n【候補1: quarter】")
print(f"  ユニークグループ数: {train_df['quarter'].nunique()}")
print(f"  各グループのサンプル数:")
quarter_counts = train_df['quarter'].value_counts().sort_index()
for q, count in quarter_counts.items():
    print(f"    {q}: {count:,}")

# 候補2: quarter + session
print("\n【候補2: quarter_session（推奨）】")
train_df['quarter_session'] = train_df['quarter'] + '_' + train_df['session'].astype(str)
print(f"  ユニークグループ数: {train_df['quarter_session'].nunique()}")
print(f"  各グループのサンプル数（top 10）:")
qs_counts = train_df['quarter_session'].value_counts().head(10)
for qs, count in qs_counts.items():
    print(f"    {qs}: {count:,}")
print(f"  最小グループサイズ: {train_df['quarter_session'].value_counts().min():,}")
print(f"  最大グループサイズ: {train_df['quarter_session'].value_counts().max():,}")
print(f"  平均グループサイズ: {train_df['quarter_session'].value_counts().mean():.1f}")

# 候補3: quarter + angle + session
print("\n【候補3: quarter_angle_session】")
train_df['qas'] = train_df['quarter'] + '_' + train_df['angle'] + '_' + train_df['session'].astype(str)
print(f"  ユニークグループ数: {train_df['qas'].nunique()}")
print(f"  最小グループサイズ: {train_df['qas'].value_counts().min():,}")
print(f"  最大グループサイズ: {train_df['qas'].value_counts().max():,}")

print("\n" + "="*80)
print("⚠️ リスク分析")
print("="*80)

# 同一sessionでのフレーム数と選手分布
print("\n【同一session内のフレーム数】")
frames_per_session = train_df.groupby('quarter_session')['frame'].nunique()
print(f"  平均フレーム数: {frames_per_session.mean():.1f}")
print(f"  最大フレーム数: {frames_per_session.max()}")
print(f"  → 同一session内のフレームは時間的に近く、選手の見た目が類似 → session単位でグループ化が必須")

# 同一session内の選手分布
print("\n【同一session内の選手分布】")
players_per_session = train_df.groupby('quarter_session')['label_id'].nunique()
print(f"  平均選手数/session: {players_per_session.mean():.1f}")
print(f"  最小選手数/session: {players_per_session.min()}")
print(f"  最大選手数/session: {players_per_session.max()}")

# 選手の出現分布（どの選手がどれだけのsessionに出現するか）
print("\n【選手の出現session数】")
player_sessions = train_df.groupby('label_id')['quarter_session'].nunique()
print(f"  選手の出現session数（平均）: {player_sessions.mean():.1f}")
for player_id in sorted(train_df['label_id'].unique()):
    sessions = train_df[train_df['label_id'] == player_id]['quarter_session'].nunique()
    samples = len(train_df[train_df['label_id'] == player_id])
    print(f"    Player {player_id}: {sessions} sessions, {samples:,} samples")

print("\n" + "="*80)
print("✅ 推奨CV戦略")
print("="*80)

print("""
【最善のCV戦略】

1. グループ列: quarter_session（quarter + '_' + session）
   - リーク防止: 同一session内のフレームは必ず同じfoldに
   - 十分なグループ数: {groups}グループ → 5-fold CVに十分
   - 適切な粒度: sessionは時間的な独立性を持つ

2. CV手法: GroupKFold
   - StratifiedGroupKFoldは選手分布の層化を試みるが、グループ制約で困難
   - GroupKFoldの方が確実にリークを防げる
   
3. 評価指標: Macro F1（unknown含む）

4. 注意点:
   - テストデータは異なるクオーター・sessionから来る
   - 選手の入れ替わりあり → 時系列の依存性も考慮
   - top/side両方の画角があるが、testはside多め

【実装例】
```python
from sklearn.model_selection import GroupKFold

# グループ列作成
df_train['group'] = df_train['quarter'] + '_' + df_train['session'].astype(str)

cv_setting = {{
    "group_col": "group",
    "n_splits": 5,
    "shuffle": True,  # GroupKFoldではFalse推奨（時系列考慮）
    "random_state": 42
}}
```

【より厳密な戦略（オプション）】
- 時系列を考慮: shuffle=False でクオーター順を維持
- Leave-One-Quarter-Out: 1つのクオーターを完全にholdout
  → テストデータが新しいクオーターから来る可能性を考慮
""".format(groups=train_df['quarter_session'].nunique()))

print("\n" + "="*80)
print("📝 次のステップ")
print("="*80)
print("""
1. runner.pyを修正してGroupKFoldに変更
2. ノートブックで新しいgroup列を作成
3. CV実行して各foldのスコアを確認
4. 可能であれば、時系列splitも試す（Leave-Future-Out）
""")
