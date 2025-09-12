import gzip
import json
import sqlite3
import os
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

def _extract_pdf_url_from_chunk(args):
    """
    単一のS2ORCファイルチャンクをスキャンし、
    ターゲットDOIのリストに一致する論文のPDF URLを返す。
    """
    filepath, target_doi_set = args
    found_urls = {} # {doi: pdf_url}
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if not isinstance(record, dict): continue
                
                doi = record.get('externalids', {}).get('doi')
                if doi and doi.upper() in target_doi_set:
                    # オープンアクセスURLを優先的に取得
                    pdf_url = record.get('content', {}).get('source', {}).get('oainfo', {}).get('openaccessurl')
                    # なければ、pdfurlsリストの最初のURLを取得
                    if not pdf_url:
                        pdf_urls = record.get('content', {}).get('source', {}).get('pdfurls', [])
                        if pdf_urls:
                            pdf_url = pdf_urls[0]
                    
                    if pdf_url:
                        found_urls[doi.upper()] = pdf_url
            except:
                continue
    return found_urls

class PdfUrlExtractor:
    """
    S2ORCの生データから、指定されたDOIリストに一致する論文の
    PDF URLを抽出し、データベースに格納するクラス。
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS used_paper_pdf_links (
                    doi TEXT PRIMARY KEY,
                    pdf_url TEXT
                )
            ''')
            conn.commit()

    def build_table(self, s2orc_dir: str):
        """
        positive_candidatesテーブルから'Used'と判定されたDOIを読み込み、
        S2ORC全体をスキャンしてused_paper_pdf_linksテーブルを構築する。
        """
        with sqlite3.connect(self.db_path) as conn:
            # 1. ターゲットとなるDOI ('Used'と判定された論文) をDBから取得
            print("Fetching target DOIs from `positive_candidates` table...")
            query = "SELECT citing_doi FROM positive_candidates WHERE llm_annotation_status = 1"
            df_candidates = pd.read_sql_query(query, conn)
            target_dois = set(df_candidates['citing_doi'].str.upper())
            print(f"Found {len(target_dois):,} 'Used' candidate DOIs to extract.")

            # 2. S2ORCファイルを並列処理
            filepaths = [os.path.join(s2orc_dir, f) for f in os.listdir(s2orc_dir) if f.endswith('.gz')]
            tasks = [(fp, target_dois) for fp in filepaths]
            
            with Pool(processes=cpu_count()) as pool:
                pbar = tqdm(total=len(filepaths), desc="Extracting PDF URLs")
                for result_dict in pool.imap_unordered(_extract_pdf_url_from_chunk, tasks):
                    if result_dict:
                        to_insert = list(result_dict.items())
                        conn.executemany("INSERT OR IGNORE INTO used_paper_pdf_links (doi, pdf_url) VALUES (?,?)", to_insert)
                        conn.commit()
                    pbar.update(1)
                pbar.close()
        
        print("✅ PDF URL table construction complete.")