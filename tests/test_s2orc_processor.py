import unittest
import os
import sys
import gzip
import json
import pandas as pd
import sqlite3
from unittest.mock import patch

# `src`ディレクトリへのパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# テスト対象のモジュールと関数をインポート
from s2orc_processor import S2ORCProcessor, process_chunk

class TestS2ORCProcessor(unittest.TestCase):
    """S2ORCProcessorクラスのテストケース"""

    def setUp(self):
        """各テストの前に、メモリ上にDBを作成し、テストデータを読み込む"""
        self.db_path = ":memory:"
        self.conn = sqlite3.connect(self.db_path)
        S2ORCProcessor._ensure_tables_exist(self.conn)

        # 外部ファイルから実際のS2ORCレコードをテストデータとして読み込む
        test_data_path = os.path.join(os.path.dirname(__file__), 'mock_data', 's2orc_record_pass.json')
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.real_passing_record = json.load(f)
            
        # フィルタリングで除外されるべき失敗ケース用のデータも定義
        self.failing_record = {"corpusid": 2, "externalids": {"doi": None}}
    
    def tearDown(self):
        """各テストの後に、DB接続を閉じる"""
        self.conn.close()

    def test_parse_and_filter_line_with_real_data(self):
        """フィルタリングロジックが実際のデータで正しく機能するかのテスト"""
        processor = S2ORCProcessor(self.db_path)
        
        # 成功ケースのテスト
        result_pass = processor.parse_and_filter_line(json.dumps(self.real_passing_record))
        self.assertIsNotNone(result_pass)
        self.assertEqual(result_pass['doi'], '10.21608/AASJ.2020.155061')
        self.assertTrue(len(result_pass['citations']) > 0)
        self.assertIn('10.2139/SSRN.3585147', result_pass['citations'])

        # 失敗ケースのテスト
        result_fail = processor.parse_and_filter_line(json.dumps(self.failing_record))
        self.assertIsNone(result_fail)

    @patch('s2orc_processor.gzip.open')
    def test_process_chunk_and_verify_db_content(self, mock_gzip_open):
        """
        実際のデータでファイル処理を行い、DBへの書き込み内容を検証するテスト
        """
        # gzip.openが実際のテストデータを返すように設定
        mock_file_content = "\n".join([json.dumps(self.real_passing_record), json.dumps(self.failing_record)])
        mock_gzip_open.return_value.__enter__.return_value = mock_file_content.splitlines()

        # ワーカー関数を実行して、書き込むべきデータを抽出
        papers_to_insert, citations_to_insert = process_chunk("dummy/path.gz")
        
        # 抽出したデータをテスト用のDBに書き込む
        if papers_to_insert:
            self.conn.executemany('INSERT OR IGNORE INTO papers (corpus_id, doi, title, abstract) VALUES (?,?,?,?)', papers_to_insert)
        if citations_to_insert:
            self.conn.executemany('INSERT OR IGNORE INTO citations (citing_doi, cited_doi) VALUES (?,?)', citations_to_insert)
        self.conn.commit()

        # --- DB内容の検証と表示 ---
        print("\n--- Verifying DB content in test_process_chunk_and_verify_db_content ---")
        
        # papersテーブルの検証
        df_papers = pd.read_sql_query("SELECT * FROM papers", self.conn)
        print("\n[papers table]")
        print(df_papers)
        self.assertEqual(len(df_papers), 1)
        self.assertEqual(df_papers.iloc[0]['doi'], '10.21608/AASJ.2020.155061')
        
        # citationsテーブルの検証
        df_citations = pd.read_sql_query("SELECT * FROM citations", self.conn)
        print("\n[citations table]")
        print(df_citations)
        self.assertTrue(len(df_citations) > 0)
        self.assertIn('10.2139/SSRN.3585147', df_citations['cited_doi'].values)
        print("-" * 60)


if __name__ == '__main__':
    unittest.main(verbosity=2)