import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 入力ファイル (先ほど作成したフルメタデータ版)
INPUT_FILE = "data/processed/evaluation_dataset_metadata_full.jsonl"
OUTPUT_DIR = "reports/figures/data_analysis"

def load_data():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return []
    with open(INPUT_FILE, 'r') as f:
        return [json.loads(line) for line in f]

def analyze_and_plot(data):
    if not data: return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # DataFrame作成用リスト
    rows = []
    for item in data:
        q_year = item.get("query_year")
        q_cites = item.get("query_citation_count", 0)
        if q_cites is None: q_cites = 0
        
        # 著者被り判定
        q_authors = set(item.get("query_authors", []))
        has_overlap = False
        gt_years = []
        gt_cites = []
        
        for gt in item.get("ground_truth_details", []):
            gt_authors = set(gt.get("authors", []))
            if not q_authors.isdisjoint(gt_authors):
                has_overlap = True
            
            y = gt.get("year")
            c = gt.get("citationCount")
            if y: gt_years.append(y)
            if c is not None: gt_cites.append(c)

        rows.append({
            "query_doi": item["query_doi"],
            "year": q_year,
            "citations": q_cites,
            "has_author_overlap": "Overlap" if has_overlap else "No Overlap",
            "gt_avg_year": sum(gt_years)/len(gt_years) if gt_years else None,
            "gt_avg_citations": sum(gt_cites)/len(gt_cites) if gt_cites else 0
        })

    df = pd.DataFrame(rows)

    # --- 1. 著者被りの割合 (円グラフ) ---
    plt.figure(figsize=(6, 6))
    overlap_counts = df['has_author_overlap'].value_counts()
    plt.pie(overlap_counts, labels=overlap_counts.index, autopct='%1.1f%%', 
            colors=['skyblue', 'lightcoral'], startangle=140)
    plt.title("Author Overlap in Evaluation Data")
    plt.savefig(f"{OUTPUT_DIR}/author_overlap_pie.png")
    print(f"Saved: {OUTPUT_DIR}/author_overlap_pie.png")

    # --- 2. 出版年の分布 (ヒストグラム) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="year", bins=20, kde=True, hue="has_author_overlap", multiple="stack")
    plt.title("Distribution of Query Publication Years")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.savefig(f"{OUTPUT_DIR}/year_distribution.png")
    print(f"Saved: {OUTPUT_DIR}/year_distribution.png")

    # --- 3. 被引用数の分布 (箱ひげ図) ---
    # 外れ値（超有名論文）が多いので対数軸で見るのが一般的
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="has_author_overlap", y="citations")
    plt.yscale("log")
    plt.title("Citation Counts: Overlap vs Non-Overlap")
    plt.ylabel("Citation Count (Log Scale)")
    plt.savefig(f"{OUTPUT_DIR}/citation_boxplot.png")
    print(f"Saved: {OUTPUT_DIR}/citation_boxplot.png")

    # --- 統計サマリー ---
    print("\n=== Data Statistics ===")
    print(df.groupby("has_author_overlap")[["year", "citations"]].describe().T)

if __name__ == "__main__":
    data = load_data()
    analyze_and_plot(data)