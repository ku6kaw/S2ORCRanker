import json
import csv
import sys
import os

# 設定
INPUT_FILE = "data/processed/evaluation_dataset_rich.jsonl"
OUTPUT_FILE = "evaluation_dataset_rich_editable.csv"

def to_csv():
    print(f"Reading from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))

    if not data:
        print("No data found.")
        return

    # CSVのヘッダー定義 (編集しやすい順序に並べる)
    # query_abstract を手前に持ってきています
    fieldnames = ["query_doi", "query_abstract", "ground_truth_dois", "metadata"]

    print(f"Writing to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in data:
            row = {
                "query_doi": item.get("query_doi", ""),
                "query_abstract": item.get("query_abstract", ""),
                # リストや辞書はJSON文字列化してセルに入れる（壊れないように）
                "ground_truth_dois": json.dumps(item.get("ground_truth_dois", []), ensure_ascii=False),
                "metadata": json.dumps(item.get("metadata", {}), ensure_ascii=False)
            }
            writer.writerow(row)

    print("✅ Conversion complete. You can now edit the CSV in Excel.")

if __name__ == "__main__":
    to_csv()