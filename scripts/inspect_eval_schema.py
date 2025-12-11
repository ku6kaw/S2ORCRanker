import json
import os
import pandas as pd
import numpy as np

# 分析対象のファイル
# もしファイル名が異なる場合はここを変更してください
TARGET_FILE = "data/processed/evaluation_dataset_rich.jsonl"

def inspect_schema():
    print(f"Checking file: {TARGET_FILE}")
    if not os.path.exists(TARGET_FILE):
        print("❌ File not found.")
        return

    # データを読み込む
    data = []
    with open(TARGET_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if not data:
        print("File is empty.")
        return

    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print(" 1. Available Keys (Columns)")
    print("="*50)
    # キーの一覧と、欠損していないデータの割合を表示
    info = []
    for col in df.columns:
        non_null_count = df[col].count()
        sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
        # リストや長い文字列は省略して表示
        if isinstance(sample_val, str) and len(sample_val) > 50:
            sample_val = sample_val[:50] + "..."
        elif isinstance(sample_val, list):
            sample_val = str(sample_val)[:50] + "..."
            
        info.append({
            "Key Name": col,
            "Type": type(df[col].iloc[0]).__name__,
            "Fill Rate": f"{(non_null_count / len(df)):.1%}",
            "Sample Value": sample_val
        })
    
    print(pd.DataFrame(info).to_markdown(index=False))

    print("\n" + "="*50)
    print(" 2. Sample Record (Full Content)")
    print("="*50)
    # ランダムな1件を詳細表示
    sample_record = df.sample(1).iloc[0].to_dict()
    print(json.dumps(sample_record, indent=2, ensure_ascii=False))

    print("\n" + "="*50)
    print(" 3. Basic Statistics")
    print("="*50)
    print(f"Total Queries: {len(df)}")
    
    if 'ground_truth_dois' in df.columns:
        gt_counts = df['ground_truth_dois'].apply(len)
        print(f"Ground Truths per Query:")
        print(f"  - Mean:   {gt_counts.mean():.2f}")
        print(f"  - Median: {gt_counts.median():.2f}")
        print(f"  - Max:    {gt_counts.max()}")
        print(f"  - Min:    {gt_counts.min()}")

    if 'query_text' in df.columns or 'query_abstract' in df.columns:
        col = 'query_text' if 'query_text' in df.columns else 'query_abstract'
        text_lens = df[col].fillna("").apply(len)
        print(f"Query Length (chars):")
        print(f"  - Mean:   {text_lens.mean():.0f}")

if __name__ == "__main__":
    inspect_schema()