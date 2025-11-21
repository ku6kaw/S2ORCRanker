# src/utils/metrics.py

import numpy as np

def calculate_recall_at_k(ranks, k_values):
    """
    Recall@K を計算する。
    ranks: 各クエリに対する「最初の正解」の順位リスト (見つからなかった場合は0または無限大)
    """
    recall_scores = {k: 0.0 for k in k_values}
    total_queries = len(ranks)
    
    if total_queries == 0:
        return recall_scores

    for k in k_values:
        # 順位が 1 以上 k 以下のものの割合
        hits = sum(1 for r in ranks if 1 <= r <= k)
        recall_scores[k] = hits / total_queries
        
    return recall_scores

def calculate_mrr(ranks):
    """
    MRR (Mean Reciprocal Rank) を計算する。
    """
    if not ranks:
        return 0.0
        
    reciprocal_ranks = [1.0 / r for r in ranks if r > 0]
    if not reciprocal_ranks:
        return 0.0
        
    return np.mean(reciprocal_ranks)

def get_rank_of_first_hit(sorted_indices, ground_truth_indices):
    """
    検索結果（ソート済みインデックス）の中で、最初の正解が現れる順位を返す。
    見つからない場合は 0 を返す。
    """
    # ground_truth_indices を set にして高速化
    gt_set = set(ground_truth_indices)
    
    for i, idx in enumerate(sorted_indices):
        if idx in gt_set:
            return i + 1 # 1-based rank
            
    return 0