import json
import os
import random
import pandas as pd

# --- 設定 ---
INPUT_FILE = "data/processed/evaluation_dataset_metadata_full.jsonl" # メタデータ付きがあればそちら推奨
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = "data/processed/evaluation_dataset_rich.jsonl"

OUTPUT_FILE = "data/processed/evaluation_dataset_official_50.jsonl"

# 以前の設定と同じにする (各データセットから1つ選ぶ)
QUERIES_PER_DATASET = 1
SEED = 42

def load_queries_from_jsonl(jsonl_path, queries_per_dataset=None, seed=42):
    print(f"Loading queries from {jsonl_path}...")
    
    # データを読み込む
    all_data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # 必須フィールドの確認
            if not item.get("query_abstract") and not item.get("query_text"):
                continue
            
            # グルーピングキー (source_datapaper) を取得
            # メタデータ内にある場合と、フラットにある場合の考慮
            source_doi = "unknown"
            if "metadata" in item and "source_datapaper" in item["metadata"]:
                source_doi = item["metadata"]["source_datapaper"]
            elif "data_paper_doi" in item:
                source_doi = item["data_paper_doi"]
            
            # データ整形（元のロジックに合わせつつ、全情報を保持）
            item["group_key_doi"] = source_doi
            all_data.append(item)

    print(f"Total queries in pool: {len(all_data)}")
    
    # グルーピングとサンプリング
    if queries_per_dataset and queries_per_dataset > 0:
        print(f"Sampling {queries_per_dataset} query(s) per dataset (Seed={seed})...")
        random.seed(seed) 
        
        grouped = {}
        for q in all_data:
            src = q["group_key_doi"]
            if src not in grouped:
                grouped[src] = []
            grouped[src].append(q)
        
        sampled_data = []
        for src, group in grouped.items():
            if len(group) > queries_per_dataset:
                # ユーザー様のコードと同じロジック
                selected = random.sample(group, queries_per_dataset)
            else:
                selected = group
            sampled_data.extend(selected)
            
        print(f"Reduced queries from {len(all_data)} to {len(sampled_data)}.")
        print(f"Total Dataset Groups: {len(grouped)}")
        return sampled_data

    return all_data

def main():
    # 1. クエリの選択 (復元)
    selected_queries = load_queries_from_jsonl(
        INPUT_FILE, 
        queries_per_dataset=QUERIES_PER_DATASET, 
        seed=SEED
    )

    # 2. 保存
    print(f"Saving selected queries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in selected_queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Done.")

    # 3. 確認用統計
    df = pd.DataFrame(selected_queries)
    print("\n[Selected Dataset Stats]")
    print(f"Count: {len(df)}")
    if 'ground_truth_dois' in df.columns:
        print(f"Avg GT Count: {df['ground_truth_dois'].apply(len).mean():.1f}")

if __name__ == "__main__":
    main()