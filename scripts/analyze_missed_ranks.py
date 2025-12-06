import json
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def analyze_ranks(file_path, threshold=50000):
    print(f"Loading results from: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    if "details" not in data:
        print("Error: 'details' key not found. Please make sure save_candidates=true was used.")
        return

    found_ranks = []
    missed_ranks = []
    zeros_count = 0 # calc_full_rank=Falseの場合、圏外は0になる

    for item in data["details"]:
        # all_gt_ranks リストから、最も順位が良いもの（最小値）を取得
        # calc_full_rank=true なら、ここには 1160万までの実際の順位が入っているはず
        ranks = item.get("all_gt_ranks", [])
        
        if not ranks:
            continue
            
        best_rank = min(ranks)
        
        if best_rank == 0:
            zeros_count += 1
            continue

        if best_rank <= threshold:
            found_ranks.append(best_rank)
        else:
            missed_ranks.append(best_rank)

    print(f"\n=== Analysis Threshold: Top-{threshold} ===")
    print(f"Total Queries: {len(data['details'])}")
    print(f"Found (<= {threshold}): {len(found_ranks)} queries")
    print(f"Missed (> {threshold}): {len(missed_ranks)} queries")
    
    if zeros_count > 0:
        print(f"⚠️ Warning: {zeros_count} queries have rank 0.")
        print("   This implies 'calc_full_rank=true' might not have been enabled for these entries.")
        print("   (Or ground truth is missing from corpus)")

    if missed_ranks:
        print("\n=== Statistics for 'Missed' Queries ===")
        print(f"Min Rank: {min(missed_ranks):,}")
        print(f"Max Rank: {max(missed_ranks):,}")
        print(f"Mean Rank: {np.mean(missed_ranks):,.0f}")
        print(f"Median Rank: {np.median(missed_ranks):,.0f}")
        
        # 分布の可視化（ヒストグラム）
        plt.figure(figsize=(10, 6))
        plt.hist(missed_ranks, bins=50, color='salmon', edgecolor='black')
        plt.title(f"Distribution of Ranks for Missed Queries (> {threshold})")
        plt.xlabel("Rank in Corpus (1 - 11.6M)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # 保存
        output_img = file_path.replace(".json", "_missed_dist.png")
        plt.savefig(output_img)
        print(f"\nHistogram saved to: {output_img}")
        
        # 具体的な例を表示
        print("\n--- Examples of Missed Ranks ---")
        missed_ranks.sort()
        print(f"Best of the missed (Closest to threshold): {missed_ranks[:5]} ...")
        print(f"Worst of the missed (Completely lost): ... {missed_ranks[-5:]}")

    else:
        print("No queries were missed! (Or data format issue)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # デフォルトファイル（適宜書き換えてください）
        target_file = "data/processed/embeddings/SPECTER2_HardNeg_round2/evaluation_results_50q_100k.json"
    
    # 閾値を変更したい場合は第2引数で (デフォルト50000)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    
    analyze_ranks(target_file, limit)