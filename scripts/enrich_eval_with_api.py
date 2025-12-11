import json
import requests
import time
import os
import pandas as pd
from tqdm import tqdm

# --- 設定 ---
# 直前に作成したタイトル付きファイル、なければ元の rich.jsonl を指定
INPUT_FILE = "data/processed/evaluation_dataset_enriched.jsonl" 
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = "data/processed/evaluation_dataset_rich.jsonl"

OUTPUT_FILE = "data/processed/evaluation_dataset_metadata_full.jsonl"

# Semantic Scholar API Endpoint
API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,year,authors,citationCount,venue"

def fetch_metadata_batch(dois):
    """Semantic Scholar APIからバッチでメタデータを取得"""
    # DOIの形式をAPI用に調整 ("DOI:..." は不要、純粋なDOIのみ)
    clean_dois = [d for d in dois if d and "/" in d]
    if not clean_dois:
        return {}
    
    # バッチサイズ制限 (APIの推奨は最大100-500程度)
    batch_size = 50
    results = {}

    for i in range(0, len(clean_dois), batch_size):
        batch = clean_dois[i : i + batch_size]
        payload = {"ids": batch}
        
        try:
            # APIリクエスト
            response = requests.post(
                API_URL, 
                params={"fields": FIELDS}, 
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                for paper in data:
                    if paper: # paperがNoneの場合がある
                        paper_id = paper.get("paperId") # APIのID
                        # 入力DOIとのマッピングが難しいので、レスポンスのDOIを探す
                        # しかしBatch APIは入力順序を保証しないため、
                        # 今回は簡易的に "DOIでリクエストして返ってきたもの" を辞書化する工夫が必要
                        # S2 APIは戻り値に入力IDを含まないことがあるため、少し面倒ですが
                        # ここでは paper['externalIds']['DOI'] を確認する手もあるが
                        # batchリクエストの順番は維持される仕様なのでそれを利用する
                        pass
                
                # S2 API Batchは入力順に対応するリストを返す仕様
                for input_doi, paper_data in zip(batch, data):
                    if paper_data:
                        results[input_doi] = paper_data
            else:
                print(f"API Error {response.status_code}: {response.text}")
                
            # レート制限回避のためのウェイト
            time.sleep(1.0)
            
        except Exception as e:
            print(f"Request Error: {e}")
            time.sleep(1.0)

    return results

def enrich_with_api():
    print(f"Loading dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f]

    # 1. 必要な全DOIをリストアップ (重複排除)
    all_dois = set()
    for item in data:
        if item.get("query_doi"):
            all_dois.add(item["query_doi"])
        for gt in item.get("ground_truth_dois", []):
            all_dois.add(gt)
            
    print(f"Fetching metadata for {len(all_dois)} papers via API...")
    print("This may take a minute...")
    
    # メタデータ取得
    metadata_map = fetch_metadata_batch(list(all_dois))
    
    print(f"Successfully fetched metadata for {len(metadata_map)} DOIs.")

    # 2. データセットに結合
    enriched_data = []
    
    for item in tqdm(data):
        q_doi = item.get("query_doi")
        
        # クエリのメタデータ
        q_meta = metadata_map.get(q_doi, {})
        item["query_authors"] = [a["name"] for a in q_meta.get("authors", [])]
        item["query_year"] = q_meta.get("year")
        item["query_citation_count"] = q_meta.get("citationCount")
        item["query_venue"] = q_meta.get("venue")
        
        # 正解データのメタデータ (リスト形式で保持)
        gt_details = []
        gt_dois = item.get("ground_truth_dois", [])
        
        for gt_doi in gt_dois:
            gt_meta = metadata_map.get(gt_doi, {})
            gt_details.append({
                "doi": gt_doi,
                "title": gt_meta.get("title"),
                "authors": [a["name"] for a in gt_meta.get("authors", [])],
                "year": gt_meta.get("year"),
                "citationCount": gt_meta.get("citationCount")
            })
            
        item["ground_truth_details"] = gt_details
        enriched_data.append(item)

    # 保存
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in enriched_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done.")
    
    # 著者重複チェックの簡易統計
    overlap_count = 0
    for item in enriched_data:
        q_authors = set(item.get("query_authors", []))
        has_overlap = False
        for gt in item.get("ground_truth_details", []):
            gt_authors = set(gt.get("authors", []))
            if not q_authors.isdisjoint(gt_authors):
                has_overlap = True
                break
        if has_overlap:
            overlap_count += 1
            
    print(f"\n[Preliminary Analysis]")
    print(f"Queries with Author Overlap (Self-Citation/Same Lab): {overlap_count} / {len(enriched_data)} ({overlap_count/len(enriched_data):.1%})")

if __name__ == "__main__":
    enrich_with_api()