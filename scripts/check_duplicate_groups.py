import json
import os

# 入力ファイル
INPUT_FILE = "data/processed/evaluation_summary_50_ranked.json"
# 上書き保存します
OUTPUT_FILE = "data/processed/evaluation_summary_50_ranked.json"

def fix_duplicate():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    print(f"Current total: {len(data)}")

    seen_dois = set()
    unique_data = []
    removed_count = 0

    for item in data:
        # データセットDOIを取得
        ds_doi = item.get("dataset_paper_doi")
        
        if ds_doi in seen_dois:
            print(f"❌ Removing duplicate group for dataset: {ds_doi}")
            print(f"   (Query DOI: {item['query_paper']['doi']})")
            removed_count += 1
            continue
        
        seen_dois.add(ds_doi)
        unique_data.append(item)

    print("-" * 30)
    print(f"Removed {removed_count} duplicates.")
    print(f"New total: {len(unique_data)}")

    if len(unique_data) == 50:
        print("✅ Correct count (50) achieved.")
        print(f"Saving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(unique_data, f, indent=2, ensure_ascii=False)
        print("Done.")
    else:
        print(f"⚠️ Warning: Count is {len(unique_data)}, not 50. Please check.")

if __name__ == "__main__":
    fix_duplicate()