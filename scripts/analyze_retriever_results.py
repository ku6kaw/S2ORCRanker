import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 設定 ---
# 1. Retrieverの検索結果 (30k件の候補リストが入っているファイル)
RETRIEVER_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_30k_HITABLE_WITH_TEXT.json"

# 2. メタデータまとめファイル (正解の詳細情報)
SUMMARY_FILE = "data/processed/evaluation_summary_50_final.json"

# 3. 出力先
OUTPUT_DIR = "reports/retriever_analysis"

def analyze_retriever():
    if not os.path.exists(RETRIEVER_FILE) or not os.path.exists(SUMMARY_FILE):
        print("Required files not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    # 検索結果 (Retrieved List)
    with open(RETRIEVER_FILE, 'r') as f:
        retriever_data = json.load(f)
        # クエリDOIをキーにして検索結果セットを辞書化
        retrieved_map = {
            item["query_doi"]: set(item["retrieved_candidates"]) 
            for item in retriever_data
        }

    # 正解メタデータ (Ground Truth Details)
    with open(SUMMARY_FILE, 'r') as f:
        summary_data = json.load(f)

    # --- 分析処理 ---
    query_stats = []
    gt_comparison = []

    print("Analyzing Found vs Missed candidates...")

    for item in summary_data:
        query = item["query_paper"]
        q_doi = query["doi"]
        
        # このクエリに対するRetrieverの結果 (30,000件)
        retrieved_set = retrieved_map.get(q_doi, set())
        
        # クエリのテキスト (類似度計算用)
        q_text = (query.get("abstract") or "").lower()
        q_tokens = set(q_text.split())

        # 正解候補ごとの判定
        candidates = item["candidates"]
        found_count = 0
        
        for cand in candidates:
            c_doi = cand["doi"]
            is_found = c_doi in retrieved_set
            
            if is_found:
                found_count += 1
            
            # テキスト類似度 (Jaccard)
            c_text = (cand.get("abstract") or "").lower()
            c_tokens = set(c_text.split())
            intersection = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            similarity = intersection / union if union > 0 else 0.0

            gt_comparison.append({
                "query_doi": q_doi,
                "dataset_doi": item["dataset_paper_doi"],
                "candidate_doi": c_doi,
                "status": "Found" if is_found else "Missed",
                "is_author_overlap": "Yes" if cand["is_author_overlap"] else "No",
                "citation_count": cand.get("citation_count", 0) or 0,
                "text_similarity": similarity,
                "year": cand.get("year")
            })

        # クエリ単位の統計
        total_gt = len(candidates)
        recall = found_count / total_gt if total_gt > 0 else 0.0
        
        query_stats.append({
            "query_doi": q_doi,
            "total_gt": total_gt,
            "found_gt": found_count,
            "missed_gt": total_gt - found_count,
            "recall_at_30k": recall
        })

    # DataFrame化
    df_q = pd.DataFrame(query_stats)
    df_c = pd.DataFrame(gt_comparison)

    # --- 1. クエリごとのRecall状況 ---
    print("\n" + "="*50)
    print(" 1. Query-level Recall@30k Statistics")
    print("="*50)
    print(df_q[["total_gt", "found_gt", "recall_at_30k"]].describe())
    
    # 全く拾えていないクエリ (Zero Recall)
    zero_recall = df_q[df_q["found_gt"] == 0]
    print(f"\nQueries with ZERO Found Candidates: {len(zero_recall)} / {len(df_q)}")
    if not zero_recall.empty:
        print(zero_recall[["query_doi", "total_gt"]].head())

    # ヒストグラム (見つかった数の分布)
    plt.figure(figsize=(10, 6))
    # 積み上げ棒グラフ用データ作成
    df_q_sorted = df_q.sort_values("total_gt", ascending=False)
    
    plt.bar(range(len(df_q_sorted)), df_q_sorted["found_gt"], label="Found", color="skyblue")
    plt.bar(range(len(df_q_sorted)), df_q_sorted["missed_gt"], bottom=df_q_sorted["found_gt"], label="Missed", color="lightgray")
    
    plt.title("Found vs Missed GTs per Query (Retriever Top-30k)")
    plt.xlabel("Query Index (Sorted by GT count)")
    plt.ylabel("Number of GTs")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/found_vs_missed_counts.png")
    plt.close()

    # --- 2. Found vs Missed の特性比較 ---
    print("\n" + "="*50)
    print(" 2. Characteristics: Found vs Missed")
    print("="*50)
    
    # 著者被りの影響
    overlap_stats = df_c.groupby(["status", "is_author_overlap"]).size().unstack(fill_value=0)
    print("\n[Author Overlap Count]")
    print(overlap_stats)
    
    # 割合計算
    overlap_rate_found = df_c[df_c["status"]=="Found"]["is_author_overlap"].apply(lambda x: 1 if x=="Yes" else 0).mean()
    overlap_rate_missed = df_c[df_c["status"]=="Missed"]["is_author_overlap"].apply(lambda x: 1 if x=="Yes" else 0).mean()
    print(f"\nOverlap Rate in FOUND:  {overlap_rate_found:.1%}")
    print(f"Overlap Rate in MISSED: {overlap_rate_missed:.1%}")

    # テキスト類似度と引用数
    print("\n[Text Similarity & Citations]")
    print(df_c.groupby("status")[["text_similarity", "citation_count"]].mean())

    # 箱ひげ図で比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 類似度
    sns.boxplot(data=df_c, x="status", y="text_similarity", ax=axes[0], palette="Set2")
    axes[0].set_title("Text Similarity (Jaccard)")
    
    # 引用数 (対数軸)
    sns.boxplot(data=df_c, x="status", y="citation_count", ax=axes[1], palette="Set2")
    axes[1].set_yscale("log")
    axes[1].set_title("Citation Count (Log Scale)")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/found_missed_characteristics.png")
    plt.close()
    
    print(f"\nAnalysis complete. Reports saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_retriever()