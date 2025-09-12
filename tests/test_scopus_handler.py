import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# `src`ディレクトリへのパスを追加
# このテストスクリプトが'tests'ディレクトリにあることを想定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

from src.scopus_handler import ScopusSearcher

# --- テスト用の偽のAPIレスポンスデータ ---
# get_hit_count用のレスポンス
mock_hit_count_response = {
    "search-results": {"opensearch:totalResults": "1234"}
}
# fetch_all_results用のレスポンス（1ページ目）
mock_fetch_page1_response = {
    "search-results": {
        "opensearch:totalResults": "30",
        "entry": [{"eid": f"2-s2.0-{i}"} for i in range(25)], # 25件のダミーデータ
        "cursor": {"@next": "DUMMY_CURSOR_VALUE"}
    }
}
# fetch_all_results用のレスポンス（2ページ目、最終）
mock_fetch_page2_response = {
    "search-results": {
        "opensearch:totalResults": "30",
        "entry": [{"eid": f"2-s2.0-{i}"} for i in range(25, 30)], # 残り5件のダミーデータ
        "cursor": {} # 次のカーソルなし
    }
}


class TestScopusSearcher(unittest.TestCase):
    """ScopusSearcherクラスのテストケース"""

    def setUp(self):
        """各テストの前に実行されるセットアップ処理"""
        self.searcher = ScopusSearcher(api_key="DUMMY_API_KEY")

    @patch('requests.Session.get')
    def test_get_hit_count_success(self, mock_get):
        """get_hit_countが正常に件数を返すかのテスト"""
        # mock_getが返す偽のレスポンスを設定
        mock_response = MagicMock()
        mock_response.json.return_value = mock_hit_count_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # テスト対象のメソッドを実行
        count = self.searcher.get_hit_count(query="test query")

        # 結果を検証
        self.assertEqual(count, 1234)

    @patch('requests.Session.get')
    def test_fetch_all_results_pagination(self, mock_get):
        """fetch_all_resultsがページネーションを正しく処理するかのテスト"""
        # 複数回のAPIコールに対して、異なるレスポンスを順番に返すように設定
        mock_response1 = MagicMock(); mock_response1.json.return_value = mock_fetch_page1_response; mock_response1.raise_for_status.return_value = None
        mock_response2 = MagicMock(); mock_response2.json.return_value = mock_fetch_page2_response; mock_response2.raise_for_status.return_value = None
        mock_get.side_effect = [mock_response1, mock_response2]

        # テスト対象のメソッドを実行
        results = self.searcher.fetch_all_results(query="test query", max_results=50)

        # 結果を検証
        self.assertEqual(len(results), 30) # 25件 + 5件 = 30件
        self.assertEqual(mock_get.call_count, 2) # APIが2回呼ばれたか
        
    @patch('requests.Session.get')
    def test_fetch_all_results_max_results_limit(self, mock_get):
        """fetch_all_resultsがmax_resultsの上限を守るかのテスト"""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fetch_page1_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # max_resultsを10に設定してメソッドを実行
        results = self.searcher.fetch_all_results(query="test query", max_results=10)

        # 結果を検証
        self.assertEqual(len(results), 10) # 25件取得したが、10件に切り詰められているか


if __name__ == '__main__':
    unittest.main(verbosity=2)