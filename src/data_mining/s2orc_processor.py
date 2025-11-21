import gzip
import json
import sqlite3
import os
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# --- フィルタリング条件を定数として定義 ---
MIN_ABSTRACT_LENGTH = 50
MIN_BODY_TEXT_LENGTH = 1000

def _extract_annotated_text(full_text, annotation_str):
    """annotationsの情報を使って、全文テキストから特定の部分を抽出するヘルパー関数"""
    if not full_text or not annotation_str: return ""
    try:
        spans = json.loads(annotation_str)
        if spans and isinstance(spans[0], dict) and 'start' in spans[0] and 'end' in spans[0]:
            start, end = int(spans[0]['start']), int(spans[0]['end'])
            return full_text[start:end]
    except: return ""
    return ""

# ▼▼▼ 修正点: この関数をクラスの外に定義 ▼▼▼
def process_chunk(filepath: str) -> tuple[list, list]:
    """
    単一の圧縮ファイルを処理し、書き込むべきデータのリストを返す。
    この関数は並列処理されるワーカーの役割を担う。
    """
    papers_to_insert = []
    citations_to_insert = []
    
    # この関数はDBに直接書き込まないため、ダミーのパスでインスタンス化
    processor = S2ORCProcessor(db_path=":memory:") 

    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            parsed_data = processor.parse_and_filter_line(line)
            if parsed_data:
                papers_to_insert.append((
                    parsed_data['corpus_id'], parsed_data['doi'],
                    parsed_data['title'], parsed_data['abstract']
                ))
                for cited_doi in parsed_data['citations']:
                    citations_to_insert.append((parsed_data['doi'], cited_doi))
    return papers_to_insert, citations_to_insert

class S2ORCProcessor:
    """S2ORCデータセットの前処理とデータベース構築を管理するクラス"""

    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """データベースに接続し、必要なテーブルを作成する"""
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_tables_exist(conn)

    @staticmethod
    def _ensure_tables_exist(conn):
        """DB接続を受け取り、テーブルの存在を保証する"""
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS papers (corpus_id INTEGER PRIMARY KEY, doi TEXT UNIQUE, title TEXT, abstract TEXT)')
        cursor.execute('CREATE TABLE IF NOT EXISTS citations (citing_doi TEXT, cited_doi TEXT)')
        conn.commit()

    def parse_and_filter_line(self, line: str) -> dict | None:
        """S2ORCの1行を解析し、品質フィルタを適用する"""
        try:
            record = json.loads(line)
            if not isinstance(record, dict): return None
            
            doi = record.get('externalids', {}).get('doi')
            if not doi: return None

            content = record.get('content') or {}
            full_text = content.get('text', '')
            annotations = content.get('annotations') or {}
            abstract = _extract_annotated_text(full_text, annotations.get('abstract'))
            
            if not abstract or len(abstract) < MIN_ABSTRACT_LENGTH: return None
            if not full_text or len(full_text) < MIN_BODY_TEXT_LENGTH: return None
            
            citations = []
            bib_entries_str = annotations.get('bibentry')
            if bib_entries_str:
                bib_entries = json.loads(bib_entries_str)
                for bib in bib_entries:
                    if isinstance(bib, dict):
                        bib_doi = bib.get('attributes', {}).get('doi')
                        if bib_doi and isinstance(bib_doi, str):
                            citations.append(bib_doi)

            return {
                "corpus_id": record.get('corpusid'), "doi": doi.upper(),
                "title": _extract_annotated_text(full_text, annotations.get('title')),
                "abstract": abstract, "citations": [c.upper() for c in citations if c]
            }
        except: return None

    def build_database_parallel(self, s2orc_dir: str):
        """ワーカープロセスにファイル処理をさせ、メインプロセスがDBへの書き込みを行う"""
        log_path = self.db_path + ".log"
        processed_files = set()
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                processed_files = set(line.strip() for line in f)
        
        all_filepaths = {os.path.join(s2orc_dir, f) for f in os.listdir(s2orc_dir) if f.endswith('.gz')}
        filepaths_to_process = sorted(list(all_filepaths - processed_files))

        if not filepaths_to_process:
            print("✅ All files have already been processed."); return

        print(f"Total files: {len(all_filepaths)}. Processed: {len(processed_files)}. Remaining: {len(filepaths_to_process)}.")
        
        # メインプロセスがDB接続とログファイルを管理
        with Pool(processes=cpu_count()) as pool, sqlite3.connect(self.db_path) as conn, open(log_path, 'a', encoding='utf-8') as log_file:
            cursor = conn.cursor()
            pbar = tqdm(total=len(filepaths_to_process), desc="Building Filtered Database")
            
            # imap_unorderedで各ファイルの処理結果（データリスト）を順次受け取る
            for i, (papers, citations) in enumerate(pool.imap_unordered(process_chunk, filepaths_to_process)):
                # メインプロセスがDBに書き込む
                if papers:
                    cursor.executemany('INSERT OR IGNORE INTO papers (corpus_id, doi, title, abstract) VALUES (?,?,?,?)', papers)
                if citations:
                    cursor.executemany('INSERT OR IGNORE INTO citations (citing_doi, cited_doi) VALUES (?,?)', citations)
                
                # 定期的にコミットして進捗をDBに反映
                if (i + 1) % 50 == 0:
                    conn.commit()

                # 完了したファイル名をログに記録
                processed_filepath = filepaths_to_process[i]
                log_file.write(processed_filepath + '\n')
                log_file.flush()
                pbar.update(1)
            
            conn.commit() # 最後に残りをコミット
            pbar.close()
        
        print("✅ Database construction complete.")