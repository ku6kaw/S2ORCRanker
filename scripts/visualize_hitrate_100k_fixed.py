# scripts/visualize_recall_full.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
import numpy as np
import glob

# --- 設定 ---
MODEL_DIRS = {
    "SPECTER2 ": "data/processed/embeddings/SPECTER2_MNRL",
    "SPECTER2 Adapter Pretrained": "data/processed/embeddings/Pretrained_SPECTER2",
    "SPECTER2 Adapter Random Neg": "data/processed/embeddings/SPECTER2_Adapter",
    "SPECTER2 Adapter Hard Neg": "data/processed/embeddings/SPECTER2_HardNeg_round2",
}

def try_load_json(path):
    """JSONを読み込み、詳細データがあれば返す"""
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # パターンA: {"details": [...]}
        if isinstance(data, dict) and "details" in data:
            return data["details"]
        
        # パターンB: [...] (リストそのもの = candidatesファイル)
        if isinstance(data, list) and len(data) > 0:
            # 必要なキーが含まれているか確認
            if "first_hit_rank_retriever" in data[0] or "first_hit_rank" in data[0]:
                return data
                
    except Exception:
        pass
    return None

def find_details_data(directory):
    """
    ディレクトリ内から詳細データを含むJSONを探し出す
    """
    # 優先順位リスト
    search_order = [
        "candidates_for_reranking_50q_100k.json", # 50qの候補ファイル
        # "evaluation_results_50q_100k.json",
        # "evaluation_results.json",
        # "candidates_for_reranking.json",     # 通常の候補ファイル
        # "*_evaluation_results.json"
    ]
    
    for fname in search_order:
        path = os.path.join(directory, fname)
        
        # ワイルドカード展開
        paths = glob.glob(path) if "*" in path else [path]
        
        for p in paths:
            details = try_load_json(p)
            if details:
                return details, p
                
    return None, None

def load_recall_curve_data():
    all_data = []
    print("Loading data for visualization...")
    
    for model_name, directory in MODEL_DIRS.items():
        if not os.path.exists(directory):
            print(f"  [Skip] Directory not found: {directory}")
            continue
            
        details, loaded_path = find_details_data(directory)
        
        if details:
            filename = os.path.basename(loaded_path)
            print(f"  Loaded {model_name} ({filename})")
            
            ranks = []
            for item in details:
                # ========================================================
                # 修正箇所: 保存されたランクを使わず、リストから再計算する
                # ========================================================
                candidates = item.get("retrieved_candidates", [])
                gt_dois = set(item.get("ground_truth_dois", []))
                
                # リスト内で最も上位にある正解を探す
                found_rank = -1
                for i, doi in enumerate(candidates):
                    if doi in gt_dois:
                        found_rank = i + 1
                        break
                
                # リストで見つかればその順位を、なければ保存された順位をフォールバックとして使う
                if found_rank > 0:
                    ranks.append(found_rank)
                else:
                    # リストにない場合、念のため保存された値も見てみる
                    saved_rank = item.get("first_hit_rank_retriever", item.get("first_hit_rank", 0))
                    if saved_rank > 0:
                        ranks.append(saved_rank)
            
            ranks.sort()
            total_queries = len(details)
            if total_queries == 0: continue

            # --- データポイントの生成 (変更なし) ---
            all_data.append({
                "Model": model_name,
                "Rank": 1,
                "Recall": 0.0
            })

            for i, rank in enumerate(ranks):
                recall = (i + 1) / total_queries
                all_data.append({
                    "Model": model_name,
                    "Rank": rank,
                    "Recall": recall
                })
            
            final_recall = len(ranks) / total_queries
            all_data.append({
                "Model": model_name,
                "Rank": 100000, 
                "Recall": final_recall
            })
        else:
            print(f"  Warning: No valid rank data found in {directory}")

    return pd.DataFrame(all_data)

def plot_full_recall_curves(df):
    if df.empty:
        print("No data to plot.")
        return

    # プロット設定
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # カラーパレット
    palette = sns.color_palette("deep", n_colors=len(df["Model"].unique()))

    # 階段状プロット
    sns.lineplot(
        data=df, 
        x="Rank", 
        y="Recall", 
        hue="Model", 
        style="Model",
        palette=palette,
        linewidth=2.5,
        markers=False, 
        dashes=False,
        drawstyle='steps-post' 
    )

    plt.xscale("log")
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("HitRate", fontsize=14)
    plt.title("HitRate@k", fontsize=16)
    
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls=":", alpha=0.4)
    plt.ylim(-0.02, 1.05)
    plt.xlim(0.9, 100000)
    
    plt.legend(title="Models", fontsize=12, title_fontsize=12, loc="upper left")
    
    output_file = "HitRate@k_fixed_v2_log.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    df = load_recall_curve_data()
    plot_full_recall_curves(df)