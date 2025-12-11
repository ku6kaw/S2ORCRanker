import json
import requests
import time
import os
from tqdm import tqdm

# --- 設定 ---
# 直前に作成したまとめファイル
INPUT_FILE = "data/processed/evaluation_summary_50_full.json"
OUTPUT_FILE = "data/processed/evaluation_summary_50_final.json" # 最終版

# Semantic Scholar API
API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# venueとpublicationVenueの両方を取得して、良い方を使う
FIELDS = "venue,publicationVenue"

def fetch_venues_batch(dois):
    """DOIリストからVenue情報を一括取得"""
    clean_dois = [d for d in dois if d and "/" in d]
    if not clean_dois:
        return {}
    
    batch_size = 50
    results = {}

    for i in range(0, len(clean_dois), batch_size):
        batch = clean_dois[i : i + batch_size]
        payload = {"ids": batch}
        
        try:
            response = requests.post(
                API_URL, 
                params={"fields": FIELDS}, 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                # 入力順序と結果の対応付け
                for input_doi, paper_data in zip(batch, data):
                    if paper_data:
                        # venue (文字列) または publicationVenue.name を優先利用
                        venue_str = paper_data.get("venue")
                        if not venue_str and paper_data.get("publicationVenue"):
                             venue_str = paper_data["publicationVenue"].get("name")
                        
                        results[input_doi] = venue_str
            else:
                print(f"API Error {response.status_code}")
                
            time.sleep(1.0) # レート制限回避
            
        except Exception as e:
            print(f"Request Error: {e}")
            time.sleep(1.0)

    return results

def fill_venues():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # 1. Venueが欠けている候補のDOIを収集
    target_dois = set()
    
    print("Scanning for missing venues...")
    for item in data:
        # クエリ自体のVenueも無ければ取得対象に
        if not item["query_paper"].get("venue"):
             target_dois.add(item["query_paper"]["doi"])

        # 候補論文
        for cand in item["candidates"]:
            if not cand.get("venue"):
                target_dois.add(cand["doi"])
    
    print(f"Found {len(target_dois)} papers missing venue information.")
    
    if len(target_dois) == 0:
        print("All venues are already present. No API calls needed.")
        # そのまま保存
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    # 2. APIで取得
    print("Fetching venue data via API...")
    venue_map = fetch_venues_batch(list(target_dois))
    print(f"Fetched {len(venue_map)} venues.")

    # 3. データを更新
    update_count = 0
    for item in data:
        # クエリの更新
        q_doi = item["query_paper"]["doi"]
        if q_doi in venue_map and venue_map[q_doi]:
            item["query_paper"]["venue"] = venue_map[q_doi]

        # 候補の更新
        for cand in item["candidates"]:
            c_doi = cand["doi"]
            if c_doi in venue_map and venue_map[c_doi]:
                cand["venue"] = venue_map[c_doi]
                update_count += 1

    # 4. 保存
    print(f"Updated {update_count} candidates.")
    print(f"Saving final dataset to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print("Done.")

if __name__ == "__main__":
    fill_venues()