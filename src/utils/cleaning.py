# src/utils/cleaning.py

import re

# 定数定義
# ★修正点: 'introduction' を削除しました。
# アブストラクトの冒頭が "Introduction" で始まるケースで全文削除されるのを防ぐためです。
STOP_WORDS = [
    'keywords', 'key words', 'references', 'acknowledgments',
    'acknowledgements', 'bibliography', 'pubmed abstract', 'publisher full text', 'full text'
]

STOP_PATTERN = re.compile(r'\b(' + '|'.join(STOP_WORDS) + r')\b', re.IGNORECASE)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S*@\S*\s?')
NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]+')
# 特殊文字除去（英数字と一部の記号以外を削除）
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s\.\,\!\?\-\'\(\)\[\]\{\}\<\>\/\=\+\*\%]')

def clean_text(text, max_length=3000):
    """
    論文アブストラクト用のクリーニング関数。
    不要なヘッダー/フッター除去、URL削除、文字数制限などを行う。
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 特定のキーワード以降（参考文献など）を切り捨て
    match = STOP_PATTERN.search(text)
    if match:
        # マッチした位置が極端に先頭に近い（例えば先頭50文字以内）場合は、
        # それは「末尾のゴミ」ではなく「本文の一部」である可能性が高いため、切り捨てないようにする安全策を追加
        if match.start() > 50:
            text = text[:match.start()]
        
    # 2. ノイズパターンの削除
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    text = NON_ASCII_PATTERN.sub('', text)
    text = SPECIAL_CHARS_PATTERN.sub('', text)
    
    # 3. 空白の正規化
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. 文字数制限
    return text[:max_length]