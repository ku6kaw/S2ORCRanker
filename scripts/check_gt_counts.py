import json
import os
import pandas as pd

# 可視化に使用したファイル
INPUT_FILE = "data/processed/evaluation_dataset_metadata_full.jsonl"

def check_counts():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE}...")
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # 集計用リスト
    stats = []
    for item in data:
        q_doi = item.get("query_doi", "N/A")
        q_title = item.get("query_title", "N/A")
        
        # 正解IDのリスト (重複がないかsetで確認)
        gt_dois = item.get("ground_truth_dois", [])
        unique_gt_count = len(set(gt_dois))
        raw_gt_count = len(gt_dois)
        
        # メタデータ詳細の数
        details_count = len(item.get("ground_truth_details", []))

        stats.append({
            "query_doi": q_doi,
            "query_title": q_title[:50] + "..." if len(q_title) > 50 else q_title,
            "gt_count_unique": unique_gt_count,
            "gt_count_raw": raw_gt_count,
            "details_count": details_count
        })

    # 正解数が多い順にソート
    stats.sort(key=lambda x: x["gt_count_unique"], reverse=True)

    # 上位20件を表示
    print("\n=== Top 20 Queries by Ground Truth Count ===")
    print(f"{'Rank':<5} | {'Count':<6} | {'DOI':<25} | {'Title'}")
    print("-" * 80)
    
    for i, row in enumerate(stats[:20]):
        print(f"{i+1:<5} | {row['gt_count_unique']:<6} | {row['query_doi']:<25} | {row['query_title']}")

    # 統計
    print("\n=== Statistics ===")
    df = pd.DataFrame(stats)
    print(df['gt_count_unique'].describe())

    # 異常値のチェック
    print("\n=== Integrity Check ===")
    if df['gt_count_unique'].equals(df['gt_count_raw']):
        print("✅ No duplicate DOIs found in ground_truth_dois list.")
    else:
        print("⚠️  Duplicates found! 'ground_truth_dois' contains duplicate values.")
        
    if df['gt_count_unique'].equals(df['details_count']):
        print("✅ Metadata details count matches DOI count.")
    else:
        print("⚠️  Mismatch! 'ground_truth_details' count differs from DOI count.")

if __name__ == "__main__":
    check_counts()