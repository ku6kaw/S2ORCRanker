# scripts/visualize_results.py

import wandb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os

# --- 1. WandB 設定 ---
ENTITY = "ku6kaw-kyushu-university" # ユーザー名
PROJECT = "s2orc-ranker"            # プロジェクト名

# WandBから取得したい実験の Run Name
TARGET_RUNS = [
    "eval_Baseline_PretrainedSciBERT",
    "eval_Siamese_Contrastive",
    "eval_Bi_RankNet_HardNeg_v2_FixLR", # v1は省略(v2と比較するため)
    "eval_Bi_Contrastive_noHead"        # 最新のContrastive
]

# グラフの凡例（表示名）
LEGEND_NAMES = {
    "eval_Baseline_PretrainedSciBERT": "SciBERT (Baseline)",
    "eval_Siamese_Contrastive": "Siamese (Contrastive Old)",
    "eval_Bi_RankNet_HardNeg_v2_FixLR": "RankNet v2 (Fixed LR)",
    "eval_Bi_Contrastive_noHead": "Contrastive (No Head)"
}

# --- 2. ローカルファイル設定 (WandBにない過去のデータ) ---
# ディレクトリ整理後のパスを指定してください
LOCAL_FILES = {
    # 例: ノートブック時代のRankNet結果 (Recall 6%のもの)
    # 整理したパスに合わせて書き換えてください
    "RankNet (Notebook v1)": "data/processed/legacy_embeddings/ranknet_v1/ranknet_evaluation_results.json",
}

def get_wandb_data():
    """WandBからデータを取得"""
    try:
        api = wandb.Api()
        runs = api.runs(f"{ENTITY}/{PROJECT}")
    except Exception as e:
        print(f"WandB Access Error: {e}")
        return pd.DataFrame()
    
    data = []
    print(f"Fetching runs from WandB: {ENTITY}/{PROJECT}...")
    
    for run in runs:
        if run.name in TARGET_RUNS:
            print(f"  Found run: {run.name}")
            history = run.summary
            
            # "recall/recall@K" 形式のキーを探す
            for key, value in history.items():
                if key.startswith("recall/recall@"):
                    try:
                        k = int(key.split("@")[1])
                        # 必要なKだけフィルタリングしても良い
                        if k in [1, 10, 100, 1000, 10000]:
                            data.append({
                                "Model": LEGEND_NAMES.get(run.name, run.name),
                                "K": k,
                                "Recall": value,
                                "Source": "WandB"
                            })
                    except:
                        continue
    return pd.DataFrame(data)

def get_local_data():
    """ローカルのJSONファイルからデータを取得"""
    data = []
    print("Fetching local files...")
    
    for label, filepath in LOCAL_FILES.items():
        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue
            
        print(f"  Loading: {label}")
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # データの形式を自動判別
            recall_dict = {}
            
            # パターンA: 新しい形式 {"recall": {"1": 0.1...}}
            if isinstance(results, dict) and "recall" in results:
                recall_dict = results["recall"]
            
            # パターンB: 古いリスト形式 (変換が必要な場合)
            elif isinstance(results, list):
                # 簡易的にRecallを計算 (全件に対するランクが含まれている場合)
                ranks = [item.get("first_hit_rank", 0) for item in results if isinstance(item, dict)]
                total = len(ranks)
                if total > 0:
                    for k in [1, 10, 100, 1000, 10000]:
                        hits = sum(1 for r in ranks if 0 < r <= k)
                        recall_dict[str(k)] = hits / total

            # データをリストに追加
            for k_str, score in recall_dict.items():
                try:
                    k = int(k_str)
                    if k in [1, 10, 100, 1000, 10000]: # 主要なKのみ
                        data.append({
                            "Model": label,
                            "K": k,
                            "Recall": float(score),
                            "Source": "Local"
                        })
                except:
                    pass
                
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            
    return pd.DataFrame(data)

def plot_recall_curves(df):
    if df.empty:
        print("No data found to plot.")
        return

    # Kでソート
    df = df.sort_values(["Model", "K"])
    
    # プロット設定
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # 折れ線グラフを描画
    sns.lineplot(
        data=df, 
        x="K", 
        y="Recall", 
        hue="Model", 
        style="Model", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=9
    )
    
    plt.title("Recall@K Comparison (All Models)", fontsize=16)
    plt.xlabel("K (Number of Retrieved Documents)", fontsize=14)
    plt.ylabel("Recall", fontsize=14)
    plt.xscale("log") # 対数軸
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    plt.legend(title="Models", fontsize=11, title_fontsize=12, loc="upper left")
    
    # 保存
    output_file = "recall_comparison_all.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to {output_file}")
    # plt.show()

if __name__ == "__main__":
    df_wandb = get_wandb_data()
    df_local = get_local_data()
    
    # 結合
    df_final = pd.concat([df_wandb, df_local], ignore_index=True)
    
    if not df_final.empty:
        print(f"\nTotal data points: {len(df_final)}")
        # 重複削除（念のため）
        df_final = df_final.drop_duplicates(subset=["Model", "K"])
        plot_recall_curves(df_final)
    else:
        print("No data available.")