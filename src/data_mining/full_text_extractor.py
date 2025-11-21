import gzip
import json
import sqlite3
import os
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

def _extract_full_text_from_chunk(args):
    """
    単一のS2ORCファイルチャンクをスキャンし、
    ターゲットDOIのリストに一致する論文の全文テキストを返す。
    """
    filepath, target_doi_set = args
    found_texts = {} # {doi: full_text}
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if not isinstance(record, dict): continue
                
                doi = record.get('externalids', {}).get('doi')
                if doi and doi.upper() in target_doi_set:
                    full_text = record.get('content', {}).get('text', '')
                    if full_text:
                        found_texts[doi.upper()] = full_text
            except:
                continue
    return found_texts

class FullTextExtractor:
    """
    S2ORCの生データから、指定されたDOIリストに一致する論文の
    全文テキストを抽出し、データベースに格納するクラス。
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS full_texts (
                    doi TEXT PRIMARY KEY,
                    full_text TEXT
                )
            ''')
            conn.commit()

    def build_table(self, s2orc_dir: str):
        """
        positive_candidatesテーブルからDOIを読み込み、
        S2ORC全体をスキャンしてfull_textsテーブルを構築する。
        """
        with sqlite3.connect(self.db_path) as conn:
            # 1. ターゲットとなるDOIをDBから取得
            print("Fetching target DOIs from `positive_candidates` table...")
            df_candidates = pd.read_sql_query("SELECT DISTINCT citing_doi FROM positive_candidates", conn)
            target_dois = set(df_candidates['citing_doi'].str.upper())
            print(f"Found {len(target_dois):,} unique candidate DOIs to extract.")

            # 2. S2ORCファイルを並列処理
            filepaths = [os.path.join(s2orc_dir, f) for f in os.listdir(s2orc_dir) if f.endswith('.gz')]
            tasks = [(fp, target_dois) for fp in filepaths]
            
            with Pool(processes=cpu_count()) as pool:
                pbar = tqdm(total=len(filepaths), desc="Extracting Full Texts")
                for result_dict in pool.imap_unordered(_extract_full_text_from_chunk, tasks):
                    if result_dict:
                        # 抽出したデータをDBに書き込み
                        to_insert = list(result_dict.items())
                        conn.executemany("INSERT OR IGNORE INTO full_texts (doi, full_text) VALUES (?,?)", to_insert)
                        conn.commit()
                    pbar.update(1)
                pbar.close()
        
        print("✅ Full text table construction complete.")