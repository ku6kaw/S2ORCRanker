import json
import numpy as np
from transformers import AutoTokenizer
import os

# 設定
DATA_FILE = "data/processed/evaluation_dataset_rich.jsonl"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

def inspect_short_queries():
    print(f"Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Reading data from: {DATA_FILE} ...")
    queries = []
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            q_text = data.get("query_abstract", "")
            q_doi = data.get("query_doi", "unknown")
            
            if not q_text:
                continue
            
            # トークン数を計測
            tokens = tokenizer.encode(q_text, add_special_tokens=False)
            queries.append({
                "doi": q_doi,
                "text": q_text,
                "length": len(tokens)
            })

    # 長さ順（昇順）にソート
    queries.sort(key=lambda x: x["length"])

    print(f"\n=== Top 20 Shortest Queries ===")
    print(f"{'Length':<8} | {'DOI':<20} | {'Text'}")
    print("-" * 100)
    
    for q in queries[:20]:
        # 改行を削除して表示
        clean_text = q["text"].replace("\n", " ")
        # 長すぎる場合はカット
        display_text = (clean_text[:70] + '...') if len(clean_text) > 70 else clean_text
        print(f"{q['length']:<8} | {q['doi']:<20} | {display_text}")

if __name__ == "__main__":
    inspect_short_queries()