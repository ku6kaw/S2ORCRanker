from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import pandas as pd
import random # 429エラーのリトライ用

# --- ▼▼▼ ユーザ設定: ここでモードを切り替える ▼▼▼ ---
# "evaluation", "training", "training_advanced"
ANNOTATION_MODE = "training_advanced" 
# --- ▲▲▲ -------------------------------------- ---

app = Flask(__name__)
print(f"--- Application starting in [ {ANNOTATION_MODE.upper()} ] mode ---")

# --- パスの設定 ---
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 's2orc_filtered.db'))
EVAL_DATAPAPERS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'datapapers', 'sampled', 'evaluation_data_papers_50.csv'))
TRAINING_DATAPAPERS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'datapapers', 'sampled', 'training_data_papers_50.csv'))
PROMPT_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'prompts', 'annotation_scoring_prompt_jp.txt'))

# --- 各DOIリストを読み込む ---
try:
    df_eval = pd.read_csv(EVAL_DATAPAPERS_FILE)
    EVAL_DOI_LIST = tuple(df_eval['cited_datapaper_doi'].str.upper().tolist())
except FileNotFoundError:
    print(f"⚠️ Warning: Evaluation data paper file not found.")
    EVAL_DOI_LIST = ()

try:
    df_train = pd.read_csv(TRAINING_DATAPAPERS_FILE)
    TRAIN_DOI_LIST = tuple(df_train['cited_datapaper_doi'].str.upper().tolist())
    df_top_20 = df_train.nlargest(20, 'used_paper_count')
    TRAIN_TOP20_DOI_LIST = tuple(df_top_20['cited_datapaper_doi'].str.upper().tolist())
except FileNotFoundError:
    print(f"⚠️ Warning: Training data paper file not found.")
    TRAIN_DOI_LIST = ()
    TRAIN_TOP20_DOI_LIST = ()

# --- データベース関連の関数 ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    return conn

def get_next_paper_to_annotate():
    """【モード別】アノテーションすべき次の論文を1件取得する"""
    paper = None
    with get_db_connection() as conn:
        
        # SQLクエリの共通部分（SELECT句とJOIN句）
        query_common_part = """
            SELECT
                pc.citing_doi, pc.cited_datapaper_doi, pc.cited_datapaper_title,
                p.title AS citing_paper_title, ft.cleaned_text AS citing_paper_text,
                pc.llm_annotation_status, pdf.pdf_url,
                COALESCE(cc.total_llm_used_count, 0) AS data_paper_total_candidates
            FROM positive_candidates AS pc
            JOIN papers AS p ON pc.citing_doi = p.doi
            JOIN full_texts AS ft ON pc.citing_doi = ft.doi
            LEFT JOIN used_paper_pdf_links AS pdf ON pc.citing_doi = pdf.doi
            LEFT JOIN (
                SELECT 
                    cited_datapaper_doi, 
                    COUNT(citing_doi) as total_llm_used_count
                FROM positive_candidates
                WHERE llm_annotation_status = 1
                GROUP BY cited_datapaper_doi
            ) AS cc ON pc.cited_datapaper_doi = cc.cited_datapaper_doi
            WHERE pc.human_annotation_status = 0 
              AND pc.llm_annotation_status = 1
        """
        
        if ANNOTATION_MODE == "evaluation":
            # --- 評価用ロジック ---
            placeholders = ','.join('?' for _ in EVAL_DOI_LIST)
            query = query_common_part + f" AND pc.cited_datapaper_doi IN ({placeholders}) ORDER BY pc.cited_datapaper_doi, pc.citing_doi LIMIT 1;"
            paper = conn.execute(query, EVAL_DOI_LIST).fetchone()
            
        elif ANNOTATION_MODE == "training":
            # --- 訓練用(基本)ロジック ---
            placeholders = ','.join('?' for _ in TRAIN_DOI_LIST)
            query_needs_anchor = f"""
                SELECT cited_datapaper_doi
                FROM positive_candidates
                WHERE cited_datapaper_doi IN ({placeholders})
                GROUP BY cited_datapaper_doi
                HAVING SUM(CASE WHEN human_annotation_status = 1 THEN 1 ELSE 0 END) = 0
            """
            rows = conn.execute(query_needs_anchor, TRAIN_DOI_LIST).fetchall()
            needs_anchor_dois = tuple([row[0] for row in rows])

            if not needs_anchor_dois:
                paper = None
            else:
                placeholders_needs_anchor = ','.join('?' for _ in needs_anchor_dois)
                query = query_common_part + f" AND pc.cited_datapaper_doi IN ({placeholders_needs_anchor}) ORDER BY pc.cited_datapaper_doi LIMIT 1;"
                paper = conn.execute(query, needs_anchor_dois).fetchone()
        
        # ▼▼▼ 修正点: 'training_advanced'のロジックを正しく実装 ▼▼▼
        elif ANNOTATION_MODE == "training_advanced":
            # --- 訓練用(追加)ロジック ---
            for data_paper_doi in TRAIN_TOP20_DOI_LIST:
                # 1. そのデータ論文の、人間が確認した「Used」の数をカウント
                count_query = "SELECT COUNT(*) FROM positive_candidates WHERE cited_datapaper_doi = ? AND human_annotation_status = 1"
                human_used_count = conn.execute(count_query, (data_paper_doi,)).fetchone()[0]
                
                # 2. まだ10件に達していない場合
                if human_used_count < 10:
                    # 3. 未確認の候補(LLM=Used)を探す
                    query_candidate = query_common_part + " AND pc.cited_datapaper_doi = ? LIMIT 1;"
                    paper = conn.execute(query_candidate, (data_paper_doi,)).fetchone()
                    
                    if paper:
                        break # 処理すべき候補が見つかったのでループを抜ける
            
    return dict(paper) if paper else None

def get_progress():
    """【モード別】のアノテーションの進捗状況を取得する"""
    with get_db_connection() as conn:
        
        if ANNOTATION_MODE == "evaluation":
            placeholders = ','.join('?' for _ in EVAL_DOI_LIST)
            total_query = f"SELECT COUNT(*) FROM positive_candidates WHERE cited_datapaper_doi IN ({placeholders}) AND llm_annotation_status = 1"
            annotated_query = f"SELECT COUNT(*) FROM positive_candidates WHERE cited_datapaper_doi IN ({placeholders}) AND llm_annotation_status = 1 AND human_annotation_status != 0"
            total = conn.execute(total_query, EVAL_DOI_LIST).fetchone()[0]
            annotated = conn.execute(annotated_query, EVAL_DOI_LIST).fetchone()[0]
            progress_data = {"annotated": annotated, "total": total, "mode": "Evaluation (Target: LLM 'Used' papers)"}

        elif ANNOTATION_MODE == "training":
            placeholders = ','.join('?' for _ in TRAIN_DOI_LIST)
            total = len(TRAIN_DOI_LIST)
            query_annotated = f"""
                SELECT COUNT(DISTINCT cited_datapaper_doi)
                FROM positive_candidates
                WHERE cited_datapaper_doi IN ({placeholders})
                  AND human_annotation_status = 1
            """
            annotated = conn.execute(query_annotated, TRAIN_DOI_LIST).fetchone()[0]
            progress_data = {"annotated": annotated, "total": total, "mode": "Training (Target: 1 anchor per Data Paper)"}
            
        elif ANNOTATION_MODE == "training_advanced":
            placeholders = ','.join('?' for _ in TRAIN_TOP20_DOI_LIST)
            # Total = 上位20グループの「LLM=Used」の総数
            total_query = f"SELECT COUNT(*) FROM positive_candidates WHERE cited_datapaper_doi IN ({placeholders}) AND llm_annotation_status = 1"
            # Annotated = 上位20グループで「人間が何らかの判断をした」総数
            annotated_query = f"SELECT COUNT(*) FROM positive_candidates WHERE cited_datapaper_doi IN ({placeholders}) AND human_annotation_status != 0"
            total = conn.execute(total_query, TRAIN_TOP20_DOI_LIST).fetchone()[0]
            annotated = conn.execute(annotated_query, TRAIN_TOP20_DOI_LIST).fetchone()[0]
            progress_data = {"annotated": annotated, "total": total, "mode": "Training-Advanced (Target: Top 20 groups)"}
        
        return progress_data

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

@app.route('/skip_datapaper', methods=['POST'])
def skip_datapaper():
    data = request.json
    data_paper_doi = data.get('cited_datapaper_doi')
    if not data_paper_doi:
        return jsonify({"status": "error", "message": "No DOI provided"}), 400
    with get_db_connection() as conn:
        query = "UPDATE positive_candidates SET human_annotation_status = -2 WHERE cited_datapaper_doi = ? AND human_annotation_status = 0"
        conn.execute(query, (data_paper_doi,))
        conn.commit()
    return jsonify({"status": "success"})

@app.route('/get_llm_prompt', methods=['POST'])
def get_llm_prompt():
    paper_data = request.json
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        prompt = prompt_template.replace("{cited_data_paper_title}", paper_data.get('cited_title', ''))
        prompt = prompt.replace("{citing_paper_title}", paper_data.get('citing_title', ''))
        prompt = prompt.replace("{full_text}", paper_data.get('citing_text', '')[:20000])

        return jsonify({"prompt": prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- サーバー起動 ---
if __name__ == '__main__':
    app.run(debug=True)