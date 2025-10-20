from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import pandas as pd

# --- Flaskアプリケーションの初期化 ---
app = Flask(__name__)

# --- パスの設定 ---
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 's2orc_filtered.db'))
EVAL_DATAPAPERS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'datapapers', 'sampled', 'evaluation_data_papers_50.csv'))
PROMPT_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'prompts', 'annotation_scoring_prompt_jp.txt'))



# --- 評価用DOIリストの読み込み ---
try:
    df_eval = pd.read_csv(EVAL_DATAPAPERS_FILE)
    EVAL_DOI_LIST = df_eval['cited_datapaper_doi'].str.upper().tolist()
    print(f"✅ Loaded {len(EVAL_DOI_LIST)} evaluation data paper DOIs.")
except FileNotFoundError:
    print(f"❌ Error: Evaluation data paper file not found at {EVAL_DATAPAPERS_FILE}")
    EVAL_DOI_LIST = []

# --- データベース関連の関数 ---
def get_db_connection():
    """データベースへの接続を取得する"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    return conn

def get_next_paper_to_annotate():
    """【評価用】かつ【LLMがUsedと判定】した、人間が未確認の論文を1件取得する"""
    with get_db_connection() as conn:
        placeholders = ','.join('?' for _ in EVAL_DOI_LIST)
        
        # ▼▼▼ 修正点: used_paper_pdf_links テーブルを LEFT JOIN し、pdf_url を取得 ▼▼▼
        query = f"""
            SELECT
                pc.citing_doi, 
                pc.cited_datapaper_doi, 
                pc.cited_datapaper_title,
                p.title AS citing_paper_title, 
                ft.cleaned_text AS citing_paper_text,
                pc.llm_annotation_status,
                pdf.pdf_url
            FROM positive_candidates AS pc
            JOIN papers AS p ON pc.citing_doi = p.doi
            JOIN full_texts AS ft ON pc.citing_doi = ft.doi
            LEFT JOIN used_paper_pdf_links AS pdf ON pc.citing_doi = pdf.doi
            WHERE pc.human_annotation_status = 0 
              AND pc.llm_annotation_status = 1
              AND pc.cited_datapaper_doi IN ({placeholders})
            ORDER BY pc.cited_datapaper_doi, pc.citing_doi
            LIMIT 1;
        """
        paper = conn.execute(query, EVAL_DOI_LIST).fetchone()
        return dict(paper) if paper else None

def get_progress():
    """【評価用】かつ【LLMがUsedと判定】した論文に対する進捗を取得する"""
    with get_db_connection() as conn:
        placeholders = ','.join('?' for _ in EVAL_DOI_LIST)
        
        total_query = f"""
            SELECT COUNT(*) FROM positive_candidates 
            WHERE cited_datapaper_doi IN ({placeholders}) AND llm_annotation_status = 1
        """
        annotated_query = f"""
            SELECT COUNT(*) FROM positive_candidates 
            WHERE cited_datapaper_doi IN ({placeholders}) AND llm_annotation_status = 1 AND human_annotation_status != 0
        """
        
        total = conn.execute(total_query, EVAL_DOI_LIST).fetchone()[0]
        annotated = conn.execute(annotated_query, EVAL_DOI_LIST).fetchone()[0]
        return {"annotated": annotated, "total": total}

# --- Webページの表示とAPIの定義 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_task')
def get_task():
    paper_data = get_next_paper_to_annotate()
    progress_data = get_progress()
    return jsonify({"paper": paper_data, "progress": progress_data})

@app.route('/annotate', methods=['POST'])
def annotate():
    data = request.json
    status = 1 if data.get('decision') == 'used' else -1
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE positive_candidates SET human_annotation_status = ? WHERE citing_doi = ? AND cited_datapaper_doi = ?",
            (status, data.get('citing_doi'), data.get('cited_datapaper_doi'))
        )
        conn.commit()
    return jsonify({"status": "success"})

@app.route('/get_llm_prompt', methods=['POST'])
def get_llm_prompt():
    """
    リクエストで受け取った論文情報を基に、
    ファイルから読み込んだプロンプトをフォーマットして返すAPI
    """
    paper_data = request.json
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # --- 修正点: .format() の代わりに .replace() を使用 ---
        # これにより、本文中の波括弧に影響されなくなる
        prompt = prompt_template.replace("{cited_data_paper_title}", paper_data.get('cited_title', ''))
        prompt = prompt.replace("{citing_paper_title}", paper_data.get('citing_title', ''))
        # 長すぎる場合に備えて20000文字に制限
        prompt = prompt.replace("{full_text}", paper_data.get('citing_text', '')[:20000])

        return jsonify({"prompt": prompt})
        
    except FileNotFoundError:
        return jsonify({"error": "Prompt file not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- サーバー起動 ---
if __name__ == '__main__':
    app.run(debug=True)