import os
import time
import requests
from tqdm.auto import tqdm

class ScopusSearcher:
    """
    Scopus APIとの通信を管理し、論文検索を実行するクラス。
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Scopus API key cannot be empty.")
        self.api_key = api_key
        self.base_url = "https://api.elsevier.com/content/search/scopus"
        self.session = requests.Session()

    def get_hit_count(self, query: str) -> int | None:
        """
        指定されたクエリの検索結果の総件数のみを取得する。
        """
        params = {
            "apiKey": self.api_key,
            "query": query,
            "count": 0
        }
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return int(data.get('search-results', {}).get('opensearch:totalResults', 0))
        except Exception as e:
            print(f"🚨 Error getting hit count: {e}")
            return None

    def fetch_all_results(self, query: str, max_results: float = float('inf')) -> list:
        """
        指定されたクエリで論文を検索し、指定された最大件数まで全件取得する。
        max_resultsに無限大(inf)を指定すると、APIが返す限り全て取得する。
        """
        retrieved_papers = []
        params = {
            "apiKey": self.api_key,
            "query": query,
            "count": 25,
            "view": "COMPLETE", # 全てのメタデータを取得
            "cursor": "*"
        }
        
        with tqdm(desc=f"Searching Scopus", unit=" papers") as pbar:
            while len(retrieved_papers) < max_results:
                try:
                    res = self.session.get(self.base_url, params=params)
                    res.raise_for_status()
                    data = res.json()
                    
                    search_results = data.get('search-results', {})
                    
                    # プログレスバーの総数を初回レスポンスから設定
                    if pbar.total == 0 or pbar.total is None:
                        total_available = int(search_results.get('opensearch:totalResults', 0))
                        pbar.total = min(total_available, max_results) if max_results != float('inf') else total_available

                    entries = search_results.get('entry', [])
                    if not entries:
                        break
                    
                    retrieved_papers.extend(entries)
                    pbar.update(len(entries))

                    # 指定した最大件数に達したら終了
                    if len(retrieved_papers) >= max_results:
                        retrieved_papers = retrieved_papers[:int(max_results)]
                        break

                    # 次のページへのカーソルを取得し、パラメータを更新
                    next_cursor = search_results.get('cursor', {}).get('@next')
                    if not next_cursor:
                        break
                    
                    params['cursor'] = next_cursor
                    time.sleep(0.1)

                except Exception as e:
                    print(f"💥 Scopus search failed: {e}")
                    break
        
        return retrieved_papers