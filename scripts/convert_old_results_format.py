# scripts/convert_old_results_format.py

import json
import os
import numpy as np

# 変換したいファイルのリスト
# (入力パス, 出力パス)
CONVERSION_TARGETS = [
    ("data/processed/legacy_embeddings/ranknet_evaluation_results.json", "data/processed/legacy_embeddings/ranknet_evaluation_results_converted.json"),
    ("data/processed/legacy_embeddings/contrastive_evaluation_results.json", "data/processed/legacy_embeddings/contrastive_evaluation_results_converted.json"),
]

# 評価に使用するKの値（現在の設定に合わせる）
K_VALUES = [1, 5, 10, 50, 100, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000]


def calculate_metrics(ranks_list):
    """First Hit RankのリストからMRRとRecallを計算"""
    # MRR
    reciprocal_ranks = [1.0 / r for r in ranks_list if r > 0]
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # Recall@K
    recall_scores = {}
    total = len(ranks_list)
    for k in K_VALUES:
        hits = sum(1 for r in ranks_list if 0 < r <= k)
        recall_scores[str(k)] = hits / total if total > 0 else 0.0
        
    return mrr, recall_scores

def convert_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Skipping {input_path}: File not found.")
        return

    print(f"Converting {input_path} ...")
    
    with open(input_path, 'r') as f:
        old_data = json.load(f)
        
    # もし既に新形式（辞書型）ならスキップ
    if isinstance(old_data, dict) and "mrr" in old_data:
        print("  -> Already in new format. Skipping.")
        return

    # 変換処理
    new_details = []
    first_hit_ranks = []
    
    for item in old_data:
        # キーのマッピングと構造変更
        # 旧: top_k_results (list of dict) -> 新: retrieved_candidates (list of doi)
        retrieved_candidates = []
        if "top_k_results" in item:
            # rank順にソートされている前提
            retrieved_candidates = [res["doi"] for res in item["top_k_results"]]
            
        new_item = {
            "query_doi": item.get("query_doi"),
            "query_text": "", # 旧データには無いので空文字
            "ground_truth_dois": item.get("ground_truth_dois", []),
            "retrieved_candidates": retrieved_candidates,
            "first_hit_rank_retriever": item.get("first_hit_rank", 0),
            "all_gt_ranks": item.get("ranks_of_all_hits", [])
        }
        new_details.append(new_item)
        first_hit_ranks.append(item.get("first_hit_rank", 0))

    # メトリクスの再計算
    mrr, recall = calculate_metrics(first_hit_ranks)
    
    # 新しい構造を作成
    new_data = {
        "mrr": mrr,
        "recall": recall,
        # "config": {}, # Config情報は復元できないので空
        "details": new_details # 詳細はキーを"details"にするか、あるいは単にリストにするか
        # visualize_results.py はトップレベルの "recall" を見ているので、
        # details の構造はそこまで重要ではないが、一応合わせる。
    }
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
        
    print(f"  -> Saved to {output_path}")
    print(f"  -> MRR: {mrr:.4f}")

if __name__ == "__main__":
    for inp, out in CONVERSION_TARGETS:
        convert_file(inp, out)