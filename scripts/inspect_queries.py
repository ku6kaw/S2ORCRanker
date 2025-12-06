import json
import random
import pandas as pd
import os
import sys

# 設定
INPUT_FILE = "data/processed/evaluation_dataset_rich.jsonl"
OUTPUT_CSV = "checked_queries.csv"
SEED = 42

def load_and_sample_queries(file_path, seed=42):
    print(f"Loading from {file_path}...")
    if not os.path.exists(file_path):
        print("File not found.")
        return []

    all_data = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not data.get("query_abstract"):
                continue
            all_data.append(data)

    # 評価時と同じサンプリングロジック (1 dataset -> 1 query)
    random.seed(seed)
    grouped = {}
    for item in all_data:
        # メタデータからデータセットの識別子を取得
        # ※データによってキーが異なる場合があるため、安全に取得
        meta = item.get("metadata", {})
        src = meta.get("source_datapaper", "unknown_source")
        
        if src not in grouped:
            grouped[src] = []
        grouped[src].append(item)
    
    sampled_data = []
    for src, group in grouped.items():
        # 各グループからランダムに1つ選ぶ
        selected = random.choice(group)
        sampled_data.append(selected)
    
    print(f"Total entries: {len(all_data)}")
    print(f"Sampled entries (1 per dataset): {len(sampled_data)}")
    return sampled_data

def inspect_data(sampled_data):
    rows = []
    
    for item in sampled_data:
        query = item.get("query_abstract", "")
        meta = item.get("metadata", {})
        
        # データセット名（またはデータ論文のタイトル）の取得を試みる
        # evaluation_dataset_rich.jsonl の構造に依存しますが、
        # 多くの場合 'source_datapaper_title' や 'title' に正解の名前が入っています
        target_name = meta.get("source_datapaper_title", "")
        if not target_name:
            target_name = meta.get("title", "Unknown Target")

        # 簡易リークチェック: タイトルがクエリに含まれているか？
        # (大文字小文字を無視してチェック)
        is_leak = False
        if target_name and target_name != "Unknown Target":
            if target_name.lower() in query.lower():
                is_leak = True
        
        rows.append({
            "Target Name": target_name,
            "Leak?": "YES" if is_leak else "",
            "Query Abstract": query
        })

    df = pd.DataFrame(rows)
    return df

def main():
    data = load_and_sample_queries(INPUT_FILE, SEED)
    if not data:
        return

    df = inspect_data(data)

    # CSVに保存 (Excelなどで確認しやすいように)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nSaved full list to {OUTPUT_CSV}")

    # 画面表示 (見やすくするため文字数を制限)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.max_rows', 50)
    
    print("\n=== Query Inspection (First 10 rows) ===")
    print(df[['Target Name', 'Leak?', 'Query Abstract']].head(10))

    # リーク疑いがあるものがあれば警告
    le