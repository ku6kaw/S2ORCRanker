import re

def preprocess_text_for_llm(text: str) -> str:
    """LLMへの入力用に、論文の全文テキストを前処理・クレンジングする"""
    if not isinstance(text, str):
        return ""

    # 1. 参考文献セクション以降を削除
    # "References", "REFERENCES", "Bibliography" などを目印にする
    # re.IGNORECASEで大文字・小文字を区別しない
    text_before_refs = re.split(r'\n\s*(?:references|bibliography)\s*\n', text, maxsplit=1, flags=re.IGNORECASE)[0]
    
    # 2. URLとメールアドレスを削除
    text_no_urls = re.sub(r'https?://\S+|www\.\S+', '', text_before_refs)
    text_no_emails = re.sub(r'\S+@\S+', '', text_no_urls)
    
    # 3. 改行コードをスペースに置換
    text_no_newlines = text_no_emails.replace('\n', ' ').replace('\r', ' ')
    
    # 4. 連続する空白を単一のスペースに統一
    cleaned_text = re.sub(r'\s+', ' ', text_no_newlines).strip()
    
    return cleaned_text