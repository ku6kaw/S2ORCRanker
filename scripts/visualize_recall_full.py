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
    "SciBERT (Baseline)": "data/processed/legacy_embeddings/pretrained",
    "RankNet v1 (Low LR)": "data/processed/embeddings/Bi_RankNet_HardNeg_v1",
    "RankNet v2 (Fixed LR)": "data/processed/embeddings/Bi_RankNet_HardNeg_v2_FixLR",
    "RankNet (Random Neg)": "data/processed/embeddings/Bi_RankNet_Random_neg_SameLR",
    "Contrastive (No Head)": "data/processed/embeddings/Bi_Contrastive_noHead",
    "Pretrained SPECTER2": "data/processed/embeddings/Pretrained_SPECTER2",
    "SPECTER2 Fine-tuned": "data/processed/embeddings/SPECTER2_MNRL",
    "SPECTER2 Adapter": "data/processed/embeddings/SPECTER2_Adapter",
    "SPECTER2 Hard Neg": "data/processed/embeddings/SPECTER2_HardNeg"
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
        "evaluation_results_50q.json",
        "candidates_for_reranking_50q.json", # 50qの候補ファイル
        "evaluation_results.json",
        "candidates_for_reranking.json",     # 通常の候補ファイル
        "*_evaluation_results.json"
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
                # キー名の揺れに対応
                r = item.get("first_hit_rank_retriever", item.get("first_hit_rank", 0))
                if r > 0:
                    ranks.append(r)
            
            ranks.sort()
            total_queries = len(details)
            if total_queries == 0: continue

            # --- データポイントの生成 ---
            # 1. 開始点
            all_data.append({
                "Model": model_name,
                "Rank": 1,
                "Recall": 0.0
            })

            # 2. イベント発生点
            for i, rank in enumerate(ranks):
                recall = (i + 1) / total_queries
                all_data.append({
                    "Model": model_name,
                    "Rank": rank,
                    "Recall": recall
                })
            
            # 3. 終了点
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

    # plt.xscale("log")
    plt.xlabel("Rank (Log Scale)", fontsize=14)
    plt.ylabel("Cumulative Recall", fontsize=14)
    plt.title("Full Recall Curve (50 Queries)", fontsize=16)
    
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls=":", alpha=0.4)
    plt.ylim(-0.02, 1.05)
    plt.xlim(0.9, 50000)
    
    plt.legend(title="Models", fontsize=12, title_fontsize=12, loc="upper left")
    
    output_file = "recall_curve_full_stepped.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    df = load_recall_curve_data()
    plot_full_recall_curves(df)