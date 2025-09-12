import os
import sqlite3
import time
from tqdm.auto import tqdm
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.gemini_handler import GeminiHandler

# --- Pydanticスキーマ定義 ---
class AnnotationDecision(BaseModel):
    decision: str

class PaperAnnotator:
    """
    LLMを使い、正例候補論文のデータセット使用状況を機械的にアノテーションするクラス。
    """
    def __init__(self, db_path: str, gemini_handler: GeminiHandler, prompt_path: str):
        self.db_path = db_path
        self.gemini = gemini_handler
        
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def _annotate_single_paper(self, candidate_data: tuple) -> tuple | None:
        """単一のペアをアノテーションし、更新用データタプルを返す"""
        citing_doi, cited_doi, cited_paper_title, citing_paper_title, citing_paper_cleaned_text = candidate_data
        prompt = self.prompt_template.format(
            cited_data_paper_title=cited_paper_title,
            citing_paper_title=citing_paper_title,
            full_text=citing_paper_cleaned_text
        )
        result = self.gemini.generate_structured_output(prompt, AnnotationDecision)
        if result and result.decision in ["Used", "Not Used"]:
            status = 1 if result.decision == "Used" else -1
            return (status, 'machine_parallel', citing_doi, cited_doi)
        return None

    def run_parallel(self, limit: int = None, max_workers: int = 10):
        """
        【ペア作成可能な】未アノテーションの正例候補をスレッド並列処理する。
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # ▼▼▼ 修正点: WHERE句に is_pair_creatable = 1 を追加 ▼▼▼
            unannotated_query = """
                SELECT
                    pc.citing_doi,
                    pc.cited_datapaper_doi,
                    pc.cited_datapaper_title,
                    p_citing.title AS citing_paper_title,
                    ft_citing.cleaned_text AS citing_paper_full_text
                FROM
                    positive_candidates AS pc
                JOIN
                    papers AS p_citing ON pc.citing_doi = p_citing.doi
                JOIN
                    full_texts AS ft_citing ON pc.citing_doi = ft_citing.doi
                WHERE
                    pc.annotation_status = 0 AND pc.is_pair_creatable = 1
            """
            if limit:
                unannotated_query += f" LIMIT {limit}"
            
            candidates = cursor.execute(unannotated_query).fetchall()
            
            if not candidates:
                print("✅ No pair-creatable unannotated papers were found.")
                return

            print(f"Found {len(candidates)} pair-creatable candidates to annotate. Starting parallel annotation...")
            
            update_data = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_candidate = {executor.submit(self._annotate_single_paper, c): c for c in candidates}
                for future in tqdm(as_completed(future_to_candidate), total=len(candidates), desc="Annotating pair-creatable papers"):
                    result = future.result()
                    if result:
                        update_data.append(result)
            
            if update_data:
                print(f"\nUpdating database with {len(update_data)} new annotations...")
                update_query = "UPDATE positive_candidates SET annotation_status = ?, annotation_source = ? WHERE citing_doi = ? AND cited_datapaper_doi = ?"
                cursor.executemany(update_query, update_data)
                conn.commit()

        print("✨ Annotation process for pair-creatable papers is complete.")