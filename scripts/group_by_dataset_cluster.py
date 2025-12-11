import json
import os
import pandas as pd
from collections import defaultdict

# 入力ファイル (メタデータ付き推奨)
INPUT_FILE = "data/processed/evaluation_dataset_metadata_full.jsonl"
OUTPUT_FILE = "data/processed/evaluation_dataset_grouped.json"

def group_queries():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # --- グルーピング処理 ---
    # Key: 論文セットのハッシュ (frozenset), Value: そのグループに属するクエリ情報のリスト
    clusters = defaultdict(list)

    for item in data:
        query_doi = item.get("query_doi")
        gt_dois = item.get("ground_truth_dois", [])
        
        # グループを定義するユニークな署名を作成
        # 「クエリ」+「正解」=「そのデータセットを使っている全論文」
        # これが一致すれば、同じデータセットに関するデータであるとみなす
        group_members = set(gt_dois)
        group_members.add(query_doi)
        
        # setはハッシュ化できないのでfrozensetにして辞書のキーにする
        cluster_signature = frozenset(group_members)
        
        clusters[cluster_signature].append(item)

    # --- 分析と保存 ---
    print(f"\nTotal Queries Processed: {len(data)}")
    print(f"Unique Groups (Datasets) Found: {len(clusters)}")

    grouped_results = []
    
    # グループIDを付与して整理
    print("\n=== Group Details ===")
    print(f"{'Group ID':<10} | {'Total Papers':<15} | {'Queries in File':<15}")
    print("-" * 50)

    for group_id, (signature, queries) in enumerate(clusters.items()):
        # このグループ（データセット）に含まれる論文の総数
        total_papers_in_group = len(signature)
        # このファイル内にクエリとして登場した回数
        queries_count = len(queries)
        
        # グループ情報を保存用リストに追加
        group_info = {
            "group_id": group_id,
            "total_papers": total_papers_in_group, # データセットを使っている論文の総数
            "queries": queries # このグループに属するクエリのリスト
        }
        grouped_results.append(group_info)
        
        # 上位または一部を表示
        if group_id < 20: 
            print(f"{group_id:<10} | {total_papers_in_group:<15} | {queries_count:<15}")

    if len(clusters) > 20:
        print("... (remaining groups omitted)")

    # --- 保存 ---
    print(f"\nSaving grouped data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(grouped_results, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    group_queries()