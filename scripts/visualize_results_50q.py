# scripts/visualize_results.py

import wandb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

# 設定
ENTITY = "ku6kaw-kyushu-university" # あなたのWandBユーザー名/チーム名
PROJECT = "s2orc-ranker"            # プロジェクト名

# 比較したい実験の Run Name リスト
# ※ 実際のWandB上のNameと一致させてください
TARGET_RUNS = [
    "eval_Baseline_PretrainedSciBERT_50q",
    "eval_Bi_RankNet_Random_neg_SameLR_50q",
    "eval_Bi_RankNet_HardNeg_v1_50q",
    "eval_Bi_RankNet_HardNeg_v2_FixLR_50q",
    "eval_Bi_Contrastive_noHead_50q"
]

# グラフの表示名（凡例）のマッピング
LEGEND_NAMES = {
    "eval_Baseline_PretrainedSciBERT_50q": "SciBERT (Baseline)",
    "eval_Bi_RankNet_Random_neg_SameLR_50q": "SciBERT (RankNet Loss, Random Neg, Same LR)",
    "eval_Bi_RankNet_HardNeg_v1_50q": "SciBERT (RankNet Loss, Miss LR)",
    "eval_Bi_RankNet_HardNeg_v2_FixLR_50q": "SciBERT (RankNet Loss, Fixed LR)",
    "eval_Bi_Contrastive_noHead_50q": "SciBERT (Contrastive Loss, No Head)"
}

def get_recall_data():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    data = []
    
    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    
    for run in runs:
        if run.name in TARGET_RUNS:
            print(f"Found run: {run.name}")
            history = run.summary
            
            # "recall/recall@K" 形式のキーを探す
            for key, value in history.items():
                if key.startswith("recall/recall@"):
                    # Kの値を取り出す
                    try:
                        k = int(key.split("@")[1])
                        data.append({
                            "Run": run.name,
                            "Model": LEGEND_NAMES.get(run.name, run.name),
                            "K": k,
                            "Recall": value
                        })
                    except:
                        continue
    
    return pd.DataFrame(data)

def plot_recall_curves(df):
    if df.empty:
        print("No data found. Check ENTITY, PROJECT, and TARGET_RUNS.")
        return

    # Kでソート
    df = df.sort_values("K")
    
    # プロット設定
    plt.figure(figsize=(12, 8))
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
        markersize=8
    )
    
    plt.title("Recall@K Comparison", fontsize=16)
    plt.xlabel("K (Number of Retrieved Documents)", fontsize=14)
    plt.ylabel("Recall", fontsize=14)
    # plt.xscale("log") # Kは桁が変わるので対数軸が見やすい（線形がいい場合はコメントアウト）
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(title="Models", fontsize=12, title_fontsize=12)
    
    # 保存
    output_file = "recall_comparison_50q.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    df = get_recall_data()
    plot_recall_curves(df)