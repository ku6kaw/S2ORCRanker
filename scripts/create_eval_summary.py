import json
import os
import sqlite3
from tqdm import tqdm

# --- 設定 ---
# 1. 以前選んだ50件のクエリリスト
SELECTED_FILE = "data/processed/evaluation_dataset_official_50.jsonl"
# 2. APIでメタデータを付与したファイル
METADATA_FILE = "data/processed/evaluation_dataset_metadata_full.jsonl"
# 3. アブストラクト取得用のDB
DB_PATH = "data/processed/s2orc_filtered.db"
# 4. 出力ファイル
OUTPUT_FILE = "data/processed/evaluation_summary_50_full.json"

def get_abstract_from_db(doi, cursor):
    """DBからアブストラクトを取得する"""
    if not doi: return None
    try:
        cursor.execute("SELECT abstract FROM papers WHERE doi=?", (doi,))
        res = cursor.fetchone()
        if res and res[0]:
            return res[0]
        # papersになければmetadataテーブルも確認
        cursor.execute("SELECT abstract FROM metadata WHERE doi=?", (doi,))
        res = cursor.fetchone()
        if res and res[0]:
            return res[0]
    except Exception:
        pass
    return None

def create_summary_v2():
    # ファイル存在チェック
    if not os.path.exists(SELECTED_FILE) or not os.path.exists(METADATA_FILE):
        print("Required JSONL files not found.")
        return
    if not os.path.exists(DB_PATH):
        print("Database file not found.")
        return

    print("Loading selected 50 queries...")
    target_query_dois = set()
    dataset_doi_map = {} 

    with open(SELECTED_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            q_doi = item.get("query_doi")
            if q_doi:
                target_query_dois.add(q_doi)
                ds_doi = item.get("metadata", {}).get("source_datapaper") or item.get("data_paper_doi", "unknown")
                dataset_doi_map[q_doi] = ds_doi

    print(f"Targeting {len(target_query_dois)} queries.")

    # DB接続
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Constructing full summary (joining Metadata & DB)...")
    summary_data = []
    
    with open(METADATA_FILE, 'r') as f:
        # 全行読み込む（件数が少ないのでリスト化してもOKだが、メモリ節約のためイテレータで）
        for line in tqdm(f, desc="Processing"):
            item = json.loads(line)
            q_doi = item.get("query_doi")
            
            if q_doi not in target_query_dois:
                continue

            # --- 1. データセット論文のDOI ---
            dataset_doi = dataset_doi_map.get(q_doi, "unknown")
            
            # --- 2. クエリ論文の情報 ---
            query_info = {
                "doi": q_doi,
                "title": item.get("query_title", "N/A"),
                "year": item.get("query_year"),
                "venue": item.get("query_venue"), # クエリ側は取得済み
                "authors": item.get("query_authors", []),
                "abstract": item.get("query_abstract") or item.get("query_text", "")
            }
            
            # --- 3. 候補（正解）論文のリスト ---
            candidates_list = []
            gt_details = item.get("ground_truth_details", [])
            q_authors_set = set(item.get("query_authors", []))
            
            for gt in gt_details:
                gt_doi = gt.get("doi")
                
                # 著者被り判定
                gt_authors = gt.get("authors", [])
                gt_authors_set = set(gt_authors)
                is_overlap = not q_authors_set.isdisjoint(gt_authors_set)
                
                # ★追加: DBからアブストラクトを取得
                abstract_text = get_abstract_from_db(gt_doi, cursor)
                
                # ★追加: Venue (API結果にあれば)
                # 注: 前回のスクリプトでgt_detailsにvenueを保存していない場合、ここはNoneになります
                venue_text = gt.get("venue") 

                candidates_list.append({
                    "doi": gt_doi,
                    "title": gt.get("title", "N/A"),
                    "year": gt.get("year"),
                    "venue": venue_text,  # 新規追加
                    "authors": gt_authors,
                    "citation_count": gt.get("citationCount"),
                    "abstract": abstract_text, # 新規追加
                    "is_author_overlap": is_overlap
                })
            
            entry = {
                "dataset_paper_doi": dataset_doi,
                "query_paper": query_info,
                "candidates": candidates_list,
                "candidates_count": len(candidates_list)
            }
            summary_data.append(entry)

    conn.close()

    # --- 保存 ---
    print(f"Saving summary to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print("Done.")

    # 確認
    if summary_data:
        first_cand = summary_data[0]["candidates"][0]
        print("\n=== Check First Candidate Data ===")
        print(f"DOI: {first_cand.get('doi')}")
        print(f"Venue: {first_cand.get('venue')}")     # Noneの可能性あり
        print(f"Abstract Length: {len(first_cand.get('abstract') or '')}") # 文字数が入っていれば成功

if __name__ == "__main__":
    create_summary_v2()