"""
個数制約付き最適化による後処理モジュール

atmaCup#22 Discussion: takaito氏の手法を実装
"""

import numpy as np
import pandas as pd
from typing import Optional
import pulp
from tqdm import tqdm

def apply_post_processing(df: pd.DataFrame) -> np.ndarray:
    """
    DataFrameから後処理を適用して最適なラベル割当を返す
    
    【シーケンスの単位】
    quarter + session ごとにグループ化し、各グループを独立したシーケンスとして最適化。
    これにより「最大10選手」制約が各セッション内で適用される。
    
    Args:
        df: 以下の列を含むDataFrame
            - quarter: クオーター
            - angle: 画角（top/side）
            - session: セッション番号
            - frame: フレーム番号
            - probs: 予測確率の配列（各行にリスト形式で格納）
    
    Returns:
        predictions: 最適化されたラベル割当（元のDataFrameの順序）
    """
    df = df.copy()
    
    # グループIDを作成（quarter + session）
    df['group_id'] = (
        df['quarter'].astype(str) + '_' +
        df['session'].astype(str)
    )
    
    # フレームIDを作成（quarter + angle + session + frame）
    df['frame_id'] = (
        df['quarter'].astype(str) + '_' +
        df['angle'].astype(str) + '_' +
        df['session'].astype(str) + '_' +
        df['frame'].astype(str)
    )
    
    # 全体の予測結果を格納
    predictions = np.full(len(df), -1, dtype=int)
    
    # グループごとに独立して最適化
    for group_id, group_df in tqdm(df.groupby('group_id', sort=False), desc="後処理中"):
        # 元の順序を保持してフレーム順にソート
        group_df = group_df.sort_index()
        
        # このグループ内のユニークなフレーム（出現順を保持）
        unique_frames = group_df['frame_id'].unique()
        seq_len = len(unique_frames)
        
        # 各フレームのサンプル数を取得
        frame_sample_counts = group_df.groupby('frame_id', sort=False).size().values
        max_samples = frame_sample_counts.max()
        
        # probsカラムから配列を取得（リストから変換）
        probs_list = []
        index_mapping = []  # (frame_index, sample_index) -> original_df_index
        
        for t, frame_id in enumerate(unique_frames):
            frame_df = group_df[group_df['frame_id'] == frame_id]
            frame_probs = np.array([row for row in frame_df['probs'].values])  # (num_samples, num_labels)
            probs_list.append(frame_probs)
            
            # 元のインデックスを記録
            for i, idx in enumerate(frame_df.index):
                index_mapping.append((t, i, idx))
        
        # 各フレームのサンプル数が異なる場合、パディング
        num_labels = probs_list[0].shape[1]
        probs_padded = np.zeros((seq_len, max_samples, num_labels))
        
        for t, frame_probs in enumerate(probs_list):
            num_samples_t = frame_probs.shape[0]
            probs_padded[t, :num_samples_t, :] = frame_probs
        
        # このグループの後処理を実行
        assignment = post_processing(probs_padded)  # (seq_len, max_samples)
        
        # 結果を元のDataFrameの順序に戻す
        for t, i, idx in index_mapping:
            label = assignment[t, i]
            # label=11はunknownなので-1に変換
            predictions[idx] = -1 if label == 11 else label
    
    return predictions 

def post_processing(probs):
    problem = pulp.LpProblem("Sequence_Label_Assignment", pulp.LpMaximize)
    seq_len, num_samples, num_labels = probs.shape

    ### 変数定義
    x = pulp.LpVariable.dicts(
        "x",
        ((t, i, l)
         for t in range(seq_len)
         for i in range(num_samples)
         for l in range(num_labels)),
        cat="Binary"
    )

    y = pulp.LpVariable.dicts(
        "y",
        (l for l in range(num_labels)),
        cat="Binary"
    )

    ### 目的関数
    problem += pulp.lpSum(
        probs[t, i, l] * x[(t, i, l)]
        for t in range(seq_len)
        for i in range(num_samples)
        for l in range(num_labels)
    )

    ### 制約①：各時点・各サンプルに 1 ラベル
    for t in range(seq_len):
        for i in range(num_samples):
            problem += pulp.lpSum(
                x[(t, i, l)] for l in range(num_labels)
            ) == 1

    ### 制約②：各時点のラベル使用回数制限
    for t in range(seq_len):
        for l in range(num_labels):
            if l == 11:
                problem += pulp.lpSum(
                    x[(t, i, l)] for i in range(num_samples)
                ) <= 2
            else:
                problem += pulp.lpSum(
                    x[(t, i, l)] for i in range(num_samples)
                ) <= 1

    ### 制約③：全体で使えるラベル数は 10 以下
    # for t in range(seq_len):
    #     for i in range(num_samples):
    #         for l in range(num_labels):
    #             problem += x[(t, i, l)] <= y[l]

    # 使用ラベル数制限
    problem += pulp.lpSum(y[l] for l in range(num_labels)) <= 10
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    assignment = np.full((seq_len, num_samples), -1)

    for t in range(seq_len):
        for i in range(num_samples):
            for l in range(num_labels):
                if pulp.value(x[(t, i, l)]) == 1:
                    assignment[t, i] = l
    return assignment