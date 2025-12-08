import pandas as pd
import json
import os
import glob
import numpy as np

# --- 設定 ---
MODEL_DIRS = {
    "SPECTER2 ": "data/processed/embeddings/SPECTER2_MNRL",
    "SPECTER2 Adapter Pretrained": "data/processed/embeddings/Pretrained_SPECTER2",
    "SPECTER2 Adapter Random Neg": "data/processed/embeddings/SPECTER2_Adapter",
    "SPECTER2 Adapter Hard Neg": "data/processed/embeddings/SPECTER2_HardNeg_round2",
}

OUTPUT_CSV = "recall_at_k_summary.csv"

def get_target_k_values():
    """
    指定されたkのリストを生成する
    - 10, 100, 500
    - 1000 ~ 10000 (1000刻み)
    - 15000 ~ 100000 (5000刻み)
    """
    k_list = [10, 100, 500]
    k_list.extend(range(1000, 11000, 1000))       # 1000, 2000, ..., 10000
    k_list.extend(range(15000, 105000, 5000))     # 15000, 20000, ..., 100000
    return sorted(list(set(k_list)))

def try_load_json(path):
    if not os.path.exists(path): return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and "details" in data: return data["details"]
        if isinstance(data, list) and len(data) > 0: return data
    except:
        pass
    return None

def find_details_data(directory):
    # FIXED版を最優先で探す
    search_order = [
        "evaluation_results_100k_FIXED.json",
        "candidates_for_reranking_100k_FIXED.json",
        "candidates_for_reranking_50q_100k.json",
        "evaluation_results_50q_100k.json",
        "evaluation_results.json",
        "candidates_for_reranking.json"
    ]
    for fname in search_order:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            details = try_load_json(path)
            if details: return details, path
    return None, None

def calculate_metrics():
    k_values = get_target_k_values()
    results = {}

    print(f"Calculating Recall for k = {k_values[:5]} ... {k_values[-5:]}")

    for model_name, directory in MODEL_DIRS.items():
        details, loaded_path = find_details_data(directory)
        
        if not details:
            print(f"  [Skip] No data found for {model_name}")
            continue
            
        print(f"  Processing {model_name} (from {os.path.basename(loaded_path)})")
        
        # ランクの再計算 (candidatesリストから)
        ranks = []
        for item in details:
            candidates = item.get("retrieved_candidates", [])
            gt_dois = set(item.get("ground_truth_dois", []))
            
            found_rank = float('inf')
            
            # リスト内探索
            for i, doi in enumerate(candidates):
                if doi in gt_dois:
                    found_rank = i + 1
                    break
            
            # リストになくても保存されたランクがあれば使う (念のため)
            if found_rank == float('inf'):
                saved_rank = item.get("first_hit_rank_retriever", item.get("first_hit_rank", 0))
                if saved_rank > 0:
                    found_rank = saved_rank
            
            ranks.append(found_rank)
        
        # Recall@k の計算
        total_queries = len(ranks)
        if total_queries == 0: continue
        
        model_scores = {}
        for k in k_values:
            # ランクが k 以下のものの割合
            hit_count = sum(1 for r in ranks if r <= k)
            recall = hit_count / total_queries
            model_scores[f"R@{k}"] = recall
            
        results[model_name] = model_scores

    # DataFrame化
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # 列（k）を正しい順序に並べ替え
    sorted_cols = [f"R@{k}" for k in k_values]
    # 存在する列だけを選択
    existing_cols = [c for c in sorted_cols if c in df.columns]
    df = df[existing_cols]

    # CSV保存
    df.to_csv(OUTPUT_CSV)
    print(f"\n✅ Table saved to: {OUTPUT_CSV}")
    
    # 画面表示 (見やすく転置して表示)
    print("\n=== Recall@k Table (Transposed) ===")
    print(df.T.to_markdown() if hasattr(df.T, 'to_markdown') else df.T)

if __name__ == "__main__":
    calculate_metrics()