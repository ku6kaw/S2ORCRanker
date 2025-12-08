import json
import os
import sys

# 設定
INPUT_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_50q_100k.json"
OUTPUT_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_30k_HITABLE.json"
TOP_K = 30000

def filter_hitable():
    print(f"Loading candidates from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("❌ Input file not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    original_count = len(data)
    filtered_data = []

    print(f"Filtering queries where Ground Truth is within Top-{TOP_K}...")

    for item in data:
        # 正解DOIのセット
        gt_dois = set(item.get("ground_truth_dois", []))
        
        # 候補DOI (Top-Kのみ)
        candidates = item.get("retrieved_candidates", [])[:TOP_K]
        candidate_set = set(candidates)
        
        # 共通部分があるか (正解が含まれているか)
        # isdisjoint は共通部分がなければ True を返す
        if not gt_dois.isdisjoint(candidate_set):
            filtered_data.append(item)
        else:
            # 含まれていない場合 (失敗ケース)
            pass

    print(f"\nOriginal Queries: {original_count}")
    print(f"Hitable Queries:  {len(filtered_data)} (Recall@30k found)")
    print(f"Removed Queries:  {original_count - len(filtered_data)}")
    
    # 保存
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n✅ Filtered candidates saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_hitable()