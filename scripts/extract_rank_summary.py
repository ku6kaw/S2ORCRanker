import json
import csv
import sys
import os

def extract_summary(json_path, csv_path):
    print(f"Loading JSON from: {json_path}")
    if not os.path.exists(json_path):
        print("❌ File not found.")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("❌ Failed to decode JSON. The file might be corrupted or too large.")
        return

    if "details" not in data:
        print("❌ 'details' key not found in JSON.")
        return

    details = data["details"]
    print(f"Found {len(details)} queries. Extracting ranks...")

    # CSV出力
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # ヘッダー
        writer.writerow(["Query DOI", "Rank", "Found?", "Num GT", "Query Snippet"])

        for item in details:
            query_doi = item.get("query_doi", "unknown")
            query_text = item.get("query_text", "")
            
            # 順位情報の取得
            # evaluate.py のバージョンによってキーが異なる可能性があるため安全に取得
            rank = item.get("first_hit_rank_retriever", 0)
            
            # calc_full_rank=true の場合、all_gt_ranks がある
            if rank == 0 and "all_gt_ranks" in item:
                gt_ranks = item["all_gt_ranks"]
                if gt_ranks:
                    # 0を除いた最小値を探す（0は圏外）
                    valid_ranks = [r for r in gt_ranks if r > 0]
                    if valid_ranks:
                        rank = min(valid_ranks)
                    else:
                        rank = 0

            # 判定 (Rank > 0 なら発見)
            found_status = "FOUND" if rank > 0 else "MISSED"
            
            # クエリ本文は長すぎるので先頭100文字だけ保存（照合用）
            snippet = query_text[:100].replace("\n", " ")

            writer.writerow([
                query_doi,
                rank,
                found_status,
                len(item.get("ground_truth_dois", [])),
                snippet
            ])

    print(f"✅ Summary saved to: {csv_path}")
    print("\n--- Preview (First 5 rows) ---")
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 6: print(line.strip())

if __name__ == "__main__":
    # 引数処理
    if len(sys.argv) > 1:
        input_json = sys.argv[1]
    else:
        # デフォルトパス（必要に応じて書き換えてください）
        input_json = "data/processed/embeddings/SPECTER2_HardNeg_round2/evaluation_results_50q_100k.json"
    
    # 出力ファイル名は入力ファイル名の拡張子を変えたもの
    output_csv = input_json.replace(".json", "_summary.csv")
    
    extract_summary(input_json, output_csv)