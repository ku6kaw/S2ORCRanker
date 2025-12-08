import json
import sqlite3
import os
from tqdm import tqdm

# 設定
INPUT_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_30k_HITABLE.json"
OUTPUT_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_30k_HITABLE_WITH_TEXT.json"
DB_PATH = "data/processed/s2orc_filtered.db"

def hydrate_candidates():
    print(f"Reading candidates from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # 全ての候補DOIを収集（重複排除）
    all_candidate_dois = set()
    for item in data:
        candidates = item.get("retrieved_candidates", [])
        all_candidate_dois.update(candidates)
    
    print(f"Total unique candidates to fetch: {len(all_candidate_dois)}")

    # DBからテキストを取得
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return

    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # DOI -> Abstract のマッピングを作成
    doi_to_text = {}
    
    # SQLiteの制限（一度に指定できるパラメータ数）を回避するため、チャンクに分けてクエリ実行
    chunk_size = 900
    doi_list = list(all_candidate_dois)
    
    print("Fetching abstracts...")
    for i in tqdm(range(0, len(doi_list), chunk_size)):
        chunk = doi_list[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        
        # テーブル名は 'papers' か 'metadata' か確認が必要ですが、
        # 一般的に 'papers' と仮定。もしエラーならテーブル名を確認してください。
        try:
            query = f"SELECT doi, abstract FROM papers WHERE doi IN ({placeholders})"
            cursor.execute(query, chunk)
        except sqlite3.OperationalError:
            # テーブル名が違う場合のフォールバック（例: metadata）
            query = f"SELECT doi, abstract FROM metadata WHERE doi IN ({placeholders})"
            cursor.execute(query, chunk)
            
        rows = cursor.fetchall()
        for doi, abstract in rows:
            if abstract:
                doi_to_text[doi] = abstract

    conn.close()
    print(f"Fetched {len(doi_to_text)} abstracts.")

    # 元データにテキスト情報を付与
    # データサイズ削減のため、各クエリ項目内ではなく、
    # ファイル全体の末尾または別キーとしてマッピングを持たせるのが効率的ですが、
    # Rerankerスクリプトの修正を最小限にするため、各item内に 'candidate_texts' 辞書を追加します。
    
    valid_data_count = 0
    
    for item in data:
        candidates = item.get("retrieved_candidates", [])
        # このクエリに必要なDOIのテキストだけを部分辞書として持たせる
        # (ファイルサイズは大きくなりますが、扱いやすい形式です)
        texts_map = {}
        for doi in candidates:
            if doi in doi_to_text:
                texts_map[doi] = doi_to_text[doi]
            else:
                # DBになかった場合（ごく稀）
                texts_map[doi] = "" 
        
        item["candidate_texts"] = texts_map
        valid_data_count += 1

    # 保存
    print(f"Saving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Hydration complete.")

if __name__ == "__main__":
    hydrate_candidates()