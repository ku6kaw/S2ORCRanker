import os
import time
import random
from google import genai
from pydantic import BaseModel
from google.api_core import exceptions

class GeminiHandler:
    """
    Gemini APIとの通信を管理し、構造化された出力を生成するクラス。
    """
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        try:
            self.client = genai.Client()
        except Exception as e:
            raise ValueError(f"GEMINI_API_KEYの設定を確認してください。エラー: {e}")
        
        # ご指示に従い、'models/'プレフィックスを付けずにモデル名を保持
        self.model_name = model_name
        print(f"✅ GeminiHandler initialized for model: {self.model_name}")

    def generate_structured_output(self, prompt: str, pydantic_schema: type[BaseModel]) -> BaseModel | None:
        """
        指定されたプロンプトとPydanticスキーマを使い、構造化されたデータを生成する。
        レート制限エラーに対する自動リトライ機能付き。
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # ご指示に従い、'config'パラメータを使用
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": pydantic_schema,
                    }
                )
                return response.parsed
            
            except exceptions.ResourceExhausted as e:
                # 指数関数的バックオフで待機時間を計算
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            
            except Exception as e:
                print(f"💥 An unexpected error occurred during Gemini API call: {e}")
                return None
        
        print(f"❌ Failed to get response after {max_retries} retries.")
        return None