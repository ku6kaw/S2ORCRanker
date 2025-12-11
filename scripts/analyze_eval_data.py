import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# --- 設定 ---
INPUT_FILE = "data/processed/evaluation_summary_50_final.json"
OUTPUT_DIR = "reports/analysis_v1"

def analyze_data():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # データをフラットなリスト（候補単位）と、クエリ単位のDataFrameに変換
    query_stats = []
    candidate_stats = []
    
    for q_idx, item in enumerate(data):
        q = item["query_paper"]
        q_text = q.get("abstract", "") or ""
        q_tokens = set(q_text.lower().split())
        
        candidates = item["candidates"]
        
        # クエリ単位の統計
        query_stats.append({
            "query_doi": q["doi"],
            "dataset_doi": item["dataset_paper_doi"],
            "candidate_count": len(candidates),
            "q_year": q.get("year"),
            "q_has_abstract": bool(q_text),
            "q_has_venue": bool(q.get("venue")),
            "dataset_id": f"D{q_idx+1}"
        })
        
        # 候補単位の統計
        for c in candidates:
            c_text = c.get("abstract", "") or ""
            c_tokens = set(c_text.lower().split())
            
            # Jaccard類似度 (単語の被り具合)
            intersection = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            
            candidate_stats.append({
                "query_doi": q["doi"],
                "candidate_doi": c["doi"],
                "is_author_overlap": c["is_author_overlap"],
                "citation_count": c.get("citation_count", 0) or 0, # None対策
                "c_year": c.get("year"),
                "c_has_abstract": bool(c_text),
                "c_has_venue": bool(c.get("venue")),
                "text_similarity": jaccard,
                "year_diff": (q.get("year") or 0) - (c.get("year") or 0)
            })

    df_q = pd.DataFrame(query_stats)
    df_c = pd.DataFrame(candidate_stats)

    print("\n" + "="*50)
    print(" 1. 基本的な健全性 (Missing Values)")
    print("="*50)
    print(f"Total Queries: {len(df_q)}")
    print(f"Total Candidates: {len(df_c)}")
    print("\n[Missing Rates]")
    print(f"Query Abstract Missing: {1 - df_q['q_has_abstract'].mean():.1%}")
    print(f"Query Venue Missing:    {1 - df_q['q_has_venue'].mean():.1%}")
    print(f"Cand. Abstract Missing: {1 - df_c['c_has_abstract'].mean():.1%}")
    print(f"Cand. Venue Missing:    {1 - df_c['c_has_venue'].mean():.1%}")

    print("\n" + "="*50)
    print(" 2. 候補数の分布 (Difficulty)")
    print("="*50)
    print(df_q['candidate_count'].describe())
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df_q['candidate_count'], bins=20)
    plt.title("Distribution of Candidates per Query")
    plt.xlabel("Number of Candidates")
    plt.savefig(f"{OUTPUT_DIR}/candidates_count_dist.png")
    plt.close()

    print("\n" + "="*50)
    print(" 3. テキスト類似度とリーケージ (Jaccard Similarity)")
    print("="*50)
    # 類似度が0.8 (80%) を超えるものは「ほぼ同じ文章」なのでリーケージの疑いあり
    high_sim = df_c[df_c['text_similarity'] > 0.8]
    print(f"⚠️ High Similarity Candidates (>0.8): {len(high_sim)} / {len(df_c)} ({len(high_sim)/len(df_c):.2%})")
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df_c['text_similarity'], bins=50)
    plt.axvline(0.8, color='r', linestyle='--', label='Leakage Threshold')
    plt.title("Text Similarity between Query and Candidates")
    plt.xlabel("Jaccard Similarity")
    plt.savefig(f"{OUTPUT_DIR}/text_similarity_dist.png")
    plt.close()

    print("\n" + "="*50)
    print(" 4. 著者被り (Bias)")
    print("="*50)
    overlap_counts = df_c['is_author_overlap'].value_counts()
    print(overlap_counts)
    print(f"Author Overlap Rate: {df_c['is_author_overlap'].mean():.1%}")

    print("\n" + "="*50)
    print(" 5. 引用数バイアス (Popularity)")
    print("="*50)
    print(f"Mean Citation Count of Candidates: {df_c['citation_count'].mean():.1f}")
    print(f"Median Citation Count: {df_c['citation_count'].median():.1f}")
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df_c['citation_count'])
    plt.xscale('log')
    plt.title("Distribution of Candidate Citation Counts (Log Scale)")
    plt.savefig(f"{OUTPUT_DIR}/citation_count_dist.png")
    plt.close()

    print("\n" + "="*50)
    print(" 6. 出版年の関係 (Temporal)")
    print("="*50)
    # Query Year - Candidate Year (正の値ならQueryの方が新しい＝正常)
    # Noneを除外
    valid_years = df_c.dropna(subset=['year_diff'])
    print(f"Mean Year Difference (Query - Cand): {valid_years['year_diff'].mean():.1f} years")
    print(f"Negative Diff (Query is older): {(valid_years['year_diff'] < 0).mean():.1%}")

    print(f"\nAnalysis complete. Reports saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_data()