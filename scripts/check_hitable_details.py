import json
import numpy as np
import os

# 設定
INPUT_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_30k_HITABLE.json"

def check_details():
    print(f"Reading: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    ranks = []
    print(f"\n{'Rank':<6} | {'Query DOI':<25} | {'Query Snippet (Check this!)'}")
    print("-" * 80)

    for item in data:
        gt_dois = set(item.get("ground_truth_dois", []))
        candidates = item.get("retrieved_candidates", [])
        
        # 候補の中から正解を探す
        found_rank = -1
        for i, cand_doi in enumerate(candidates):
            if cand_doi in gt_dois:
                found_rank = i + 1
                break
        
        if found_rank != -1:
            ranks.append(found_rank)
            
            # クエリの冒頭を表示して、ちゃんと修正されているか確認
            query_text = item.get("query_text", "")
            snippet = query_text[:50].replace("\n", " ")
            
            print(f"{found_rank:<6} | {item['query_doi']:<25} | {snippet}...")

    # 統計情報
    if ranks:
        print("-" * 80)
        print(f"Total Hitable: {len(ranks)}")
        print(f"Best Rank:     {min(ranks)}")
        print(f"Worst Rank:    {max(ranks)}")
        print(f"Median Rank:   {np.median(ranks):.0f}")
        print(f"Mean Rank:     {np.mean(ranks):.0f}")
        
        # 順位分布
        print("\nRank Distribution:")
        print(f"  Top-10:    {len([r for r in ranks if r <= 10])}")
        print(f"  Top-100:   {len([r for r in ranks if r <= 100])}")
        print(f"  Top-1000:  {len([r for r in ranks if r <= 1000])}")
        print(f"  Top-10000: {len([r for r in ranks if r <= 10000])}")
        print(f"  Top-30000: {len(ranks)}")

if __name__ == "__main__":
    check_details()