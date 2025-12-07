import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os
import sys

# 設定
DATA_FILE = "data/processed/evaluation_dataset_rich.jsonl"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
OUTPUT_IMG = "query_token_length_dist.png"

def check_lengths():
    print(f"Loading tokenizer: {MODEL_NAME} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Reading data from: {DATA_FILE} ...")
    if not os.path.exists(DATA_FILE):
        print("File not found.")
        return

    lengths = []
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            query = data.get("query_abstract", "")
            if not query:
                continue
            
            # トークン化して長さを取得
            # (特殊トークンを含まない純粋な長さ)
            tokens = tokenizer.encode(query, add_special_tokens=False)
            lengths.append(len(tokens))

    if not lengths:
        print("No queries found.")
        return

    # 統計計算
    lengths = np.array(lengths)
    print("\n=== Query Token Length Statistics (BGE-M3 Tokenizer) ===")
    print(f"Count:  {len(lengths)}")
    print(f"Min:    {np.min(lengths)}")
    print(f"Max:    {np.max(lengths)}")
    print(f"Mean:   {np.mean(lengths):.2f}")
    print(f"Median: {np.median(lengths):.2f}")
    print(f"95%ile: {np.percentile(lengths, 95):.2f}")

    # 推定される総入力長 (Query + Document)
    # DocumentもAbstractだと仮定すると、入力長は Queryの約2倍 + 特殊トークン(3~4個)
    est_total = np.median(lengths) * 2 + 4
    print(f"\nEstimated Total Input Length (Query + Doc): ~{est_total:.0f} tokens (Median)")

    # ヒストグラムの作成
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Query Token Lengths\n({MODEL_NAME})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    
    # 統計線を引く
    plt.axvline(np.mean(lengths), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(lengths):.0f}')
    plt.axvline(np.percentile(lengths, 95), color='orange', linestyle='dashed', linewidth=1, label=f'95%: {np.percentile(lengths, 95):.0f}')
    plt.legend()

    plt.savefig(OUTPUT_IMG)
    print(f"\nHistogram saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    check_lengths()