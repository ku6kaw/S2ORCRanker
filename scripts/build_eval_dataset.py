# scripts/build_eval_dataset.py

import sys
import os
import sqlite3
import pandas as pd
import json
from tqdm.auto import tqdm

# srcへのパス
sys.path.append(os.getcwd())
from src.utils.cleaning import clean_text

# 設定
DB_PATH = "data/processed/s2orc_filtered.db"
INPUT_CSV = "data/datapapers/sampled/evaluation_data_papers_50_v2.csv" # 状況に合わせて _v2.csv 等に変更してください
OUTPUT_FILE = "data/processed/evaluation_dataset_rich.jsonl"

def get_paper_metadata(conn, doi):
    """DBから論文のメタデータを取得する"""
    try:
        # 将来的には title, year, field 等もここで取得可能
        row = conn.execute("SELECT abstract FROM papers WHERE doi = ?", (doi,)).fetchone()
        if row:
            return {"abstract": row[0]}
    except Exception:
        pass
    # 見つからない場合は空の情報を返す
    return {"abstract": ""}

def main():
    print(f"Loading input CSV: {INPUT_CSV}")
    df_eval = pd.read_csv(INPUT_CSV)
    
    # 評価用データ（データ論文DOI）のリスト
    target_datapaper_dois = df_eval['cited_datapaper_doi'].unique()
    
    valid_queries = []
    stats = {"total": 0, "empty_abstract": 0, "short_abstract": 0, "no_gt": 0}
    
    with sqlite3.connect(DB_PATH) as conn:
        for dp_doi in tqdm(target_datapaper_dois, desc="Building Dataset"):
            # 1. 正解ペアの候補を取得 (Human Annotation = 1 のもの)
            try:
                query_gt = """
                    SELECT citing_doi 
                    FROM positive_candidates 
                    WHERE cited_datapaper_doi = ? AND human_annotation_status = 1
                """
                gt_rows = conn.execute(query_gt, (dp_doi,)).fetchall()
            except sqlite3.OperationalError:
                print("Warning: DB schema might be missing 'positive_candidates' table.")
                continue
            
            # 重複除去
            all_related_dois = list({row[0] for row in gt_rows})
            
            # 正解が2つ以上ないと「クエリ」と「正解」を作れない（自分以外を正解にするため）ので、
            # ここだけはスキップします（物理的に評価不可能なため）
            if len(all_related_dois) < 2:
                stats["no_gt"] += 1
                continue
            
            # 2. クエリと正解のセットを作成
            for i, query_doi in enumerate(all_related_dois):
                stats["total"] += 1
                
                # メタデータ取得
                meta = get_paper_metadata(conn, query_doi)
                raw_text = meta['abstract'] if meta and meta['abstract'] else ""
                
                # クリーニング (空文字でもエラーにならない)
                cleaned_text = clean_text(raw_text)
                
                # 統計情報の更新 (除外はしません)
                if not cleaned_text:
                    stats["empty_abstract"] += 1
                elif len(cleaned_text) < 50:
                    stats["short_abstract"] += 1
                
                # 正解リスト (自分以外)
                ground_truths = [d for d in all_related_dois if d != query_doi]
                
                # データ構築
                # ここに必要な情報を詰め込んでおきます
                record = {
                    "query_doi": query_doi,
                    "query_abstract": cleaned_text, # 空でもそのまま記録
                    "ground_truth_dois": ground_truths,
                    "metadata": {
                        "source_datapaper": dp_doi,
                        "char_length": len(cleaned_text),
                        "original_abstract_found": bool(raw_text) # 元データがあったかどうかのフラグ
                    }
                }
                valid_queries.append(record)

    # 保存
    print(f"Saving {len(valid_queries)} queries to {OUTPUT_FILE}...")
    
    # 出力ディレクトリの作成
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_FILE, 'w') as f:
        for record in valid_queries:
            f.write(json.dumps(record) + "\n")
            
    print("\n=== Summary ===")
    print(f"Total Queries Created: {len(valid_queries)}")
    print(f"  - Empty Abstracts: {stats['empty_abstract']} (Included)")
    print(f"  - Short Abstracts (<50 chars): {stats['short_abstract']} (Included)")
    print(f"Skipped (Not enough GT to form pair): {stats['no_gt']}")

if __name__ == "__main__":
    main()