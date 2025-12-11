import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# --- 設定 ---
INPUT_FILE = "data/processed/evaluation_summary_50_ranked.json"
OUTPUT_FILE = "reports/figures/rank_distribution_50_groups_v2.png"

def visualize_ranks():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("Loading data...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # クエリをソート (正解数が多い順)
    data.sort(key=lambda x: x["candidates_count"], reverse=True)

    # プロット用のデータフレームを作成
    plot_data = []
    
    # クエリごとの統計情報を保存するリスト（Y軸ラベル装飾用）
    query_stats = []

    for q_idx, item in enumerate(data):
        group_label = f"Q{q_idx+1}" # 基本ラベル
        
        candidates = item["candidates"]
        total_count = len(candidates)
        missed_count = 0
        
        for cand in candidates:
            # Retriever Rank
            ret_rank = cand.get("retriever_rank")
            # 著者被り
            is_overlap = cand.get("is_author_overlap", False)

            status = "Missed"
            plot_rank = None 
            
            if ret_rank is not None:
                status = "Retrieved"
                plot_rank = ret_rank
            else:
                missed_count += 1
            
            # Missed以外のみプロットデータに追加
            if status == "Retrieved":
                plot_data.append({
                    "Query Index": q_idx,
                    "Rank": plot_rank,
                    "Status": status,
                    "Overlap": is_overlap,
                    "DOI": cand["doi"]
                })
        
        # クエリごとの統計を保存
        query_stats.append({
            "label_base": f"{group_label} (n={total_count})",
            "missed": missed_count,
            "total": total_count
        })

    df = pd.DataFrame(plot_data)

    # --- 描画 ---
    plt.figure(figsize=(15, 20)) # 縦長にする
    
    # 1. 検索済み (Retrieved) のプロット
    retrieved = df[df["Status"] == "Retrieved"]
    plt.scatter(retrieved["Rank"], retrieved["Query Index"], 
                color="skyblue", marker="o", alpha=0.6, s=60, edgecolors="blue", label="Retrieved (Top-30k)")

    # 2. 著者被り (Overlap) の強調
    overlap = df[df["Overlap"] == True]
    if not overlap.empty:
        plt.scatter(overlap["Rank"], overlap["Query Index"], 
                    s=80, facecolors='none', edgecolors='orange', linewidth=2, label="Author Overlap")

    # 軸の設定
    plt.xlim(0.8, 100000) # 1位〜
    
    # Y軸の設定 (反転)
    plt.ylim(-1, len(data))
    plt.gca().invert_yaxis() # 上から順に
    
    # --- Y軸ラベルの装飾 ---
    ax = plt.gca()
    
    final_labels = []
    label_colors = [] # (color, weight)
    
    for stats in query_stats:
        text = stats["label_base"]
        color = "black"
        weight = "normal"
        
        if stats["missed"] == stats["total"]:
            # 全滅 (Zero Recall) -> 赤色 & 太字
            color = "red"
            weight = "bold"
        
        final_labels.append(text)
        label_colors.append((color, weight))
        
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(final_labels)
    
    # 色の適用
    for tick, (color, weight) in zip(ax.get_yticklabels(), label_colors):
        tick.set_color(color)
        tick.set_fontweight(weight)

    # 装飾
    plt.axvline(30000, color="gray", linestyle="-", alpha=0.3, label="Retriever Limit")
    
    plt.xlabel("Rank")
    plt.ylabel("Query Group (Sorted by Candidate Count)")
    plt.title("Distribution of Candidates Ranks (Missed candidates are highlighted in labels)")
    plt.legend(loc="upper right")
    plt.grid(axis="x", which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    print(f"Saving rank map to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    visualize_ranks()