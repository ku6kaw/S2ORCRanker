# scripts/evaluate.py

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import pandas as pd
import json
import sqlite3
import faiss
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig

# srcへのパスを通す
sys.path.append(os.getcwd())
from src.modeling.bi_encoder import SiameseBiEncoder
from src.utils.cleaning import clean_text
from src.utils.metrics import calculate_recall_at_k, calculate_mrr, get_rank_of_first_hit

def load_ground_truth(csv_path, db_path):
    """評価用CSVとDBから、クエリと正解ペア(DOI)のリストを作成する"""
    df_eval = pd.read_csv(csv_path)
    eval_data_paper_dois = df_eval['cited_datapaper_doi'].unique()
    
    queries = []
    
    with sqlite3.connect(db_path) as conn:
        for dp_doi in tqdm(eval_data_paper_dois, desc="Loading Ground Truth"):
            # 正解DOI (Human=1) を取得
            # テーブル名はプロジェクトに合わせて調整 (positive_candidates想定)
            try:
                query_gt = "SELECT citing_doi FROM positive_candidates WHERE cited_datapaper_doi = ? AND human_annotation_status = 1"
                gt_rows = conn.execute(query_gt, (dp_doi,)).fetchall()
            except sqlite3.OperationalError:
                print("Warning: 'positive_candidates' table not found or schema mismatch.")
                continue
                
            ground_truth_dois = {row[0] for row in gt_rows}
            
            # 正解が2件以上ある場合、1つをクエリ、残りを正解とする
            # (Query 1 : DB N)
            if len(ground_truth_dois) >= 2:
                # 1件をクエリとして使用
                query_doi = list(ground_truth_dois)[0]
                target_dois = list(ground_truth_dois)[1:]
                
                # クエリの本文取得
                row = conn.execute("SELECT abstract FROM papers WHERE doi = ?", (query_doi,)).fetchone()
                if row and row[0]:
                    queries.append({
                        "query_doi": query_doi,
                        "query_text": clean_text(row[0]),
                        "ground_truth_dois": target_dois,
                        "data_paper_doi": dp_doi
                    })
    
    return queries

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Evaluation ===")
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. コーパスベクトルとDOIマップのロード
    embeddings_path = os.path.join(cfg.data.output_dir, cfg.data.embeddings_file)
    doi_map_path = os.path.join(cfg.data.output_dir, cfg.data.doi_map_file)
    
    print(f"Loading DOI map from {doi_map_path}...")
    with open(doi_map_path, 'r') as f:
        doi_list = json.load(f)
    
    # DOI -> Index の逆引きマップ
    doi_to_index = {doi: i for i, doi in enumerate(doi_list)}
    num_vectors = len(doi_list)
    
    print(f"Loading embeddings (mmap) from {embeddings_path}...")
    # 次元数を取得（設定ファイルかデフォルト値）
    hidden_size = 768 
    
    # Memmapロード (Read-only)
    corpus_embeddings = np.memmap(
        embeddings_path, 
        dtype='float32', 
        mode='r', 
        shape=(num_vectors, hidden_size)
    )
    
    # 2. 検索インデックスの構築 (Faiss)
    print("Building Faiss Index...")
    # 内積(IP) = コサイン類似度 (正規化されていれば)
    # SciBERTの出力は正規化されていない場合が多いので、検索時に正規化するか、
    # ここでIndexFlatIPを使う（ユークリッド距離より内積の方が一般的）
    
    if cfg.evaluation.use_faiss:
        index = faiss.IndexFlatIP(hidden_size)
        if cfg.evaluation.gpu_search and torch.cuda.is_available():
            print("Moving Faiss index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # データをメモリに乗せて追加（メモリが足りない場合はバッチ追加が必要）
        # ここでは全件オンメモリ前提。足りない場合はCPUインデックスで対応。
        # 注意: memmapはディスク上にあるが、faiss.addするとメモリに載る
        print("Adding vectors to index (this may consume RAM)...")
        # バッチで追加して進捗表示
        batch_size = 50000
        for i in tqdm(range(0, num_vectors, batch_size), desc="Indexing"):
            batch_vecs = np.array(corpus_embeddings[i : i + batch_size])
            # 正規化 (L2 Norm) - コサイン類似度にするため
            faiss.normalize_L2(batch_vecs)
            index.add(batch_vecs)
    else:
        print("Faiss disabled. Using manual matrix multiplication (slow).")
        index = None

    # 3. クエリモデルの準備
    print(f"Loading query encoder: {cfg.model.path}")
    try:
        config = AutoConfig.from_pretrained(cfg.model.path)
        model = SiameseBiEncoder.from_pretrained(cfg.model.path, config=config)
    except:
        model = SiameseBiEncoder.from_pretrained(cfg.model.base_name)
        
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    # 4. 評価データの準備
    queries = load_ground_truth(cfg.data.eval_data_path, cfg.data.db_path)
    print(f"Loaded {len(queries)} queries for evaluation.")
    
    if len(queries) == 0:
        print("No queries found. Exiting.")
        return

    # 5. 検索実行
    ranks = []
    top_k_results = []
    
    max_k = max(cfg.evaluation.k_values)
    
    for q_data in tqdm(queries, desc="Evaluating"):
        # クエリベクトル化
        inputs = tokenizer(
            q_data["query_text"], 
            padding=True, 
            truncation=True, 
            max_length=cfg.model.max_length, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_vectors=True)
            q_vec = out.logits.cpu().numpy()
        
        # 正規化
        faiss.normalize_L2(q_vec)
        
        # 検索
        # 検索件数: 最大のKまで + 余裕（訓練データ除外のため）
        search_k = max_k + 50 
        D, I = index.search(q_vec, search_k)
        
        # 結果処理
        found_indices = I[0]
        # scores = D[0]
        
        # インデックス -> DOI変換
        found_dois = [doi_list[idx] for idx in found_indices if idx >= 0 and idx < len(doi_list)]
        
        # 正解インデックスの特定 (検索結果の中での位置)
        ground_truth_set = set(q_data["ground_truth_dois"])
        
        # 訓練データの除外（自分自身や、学習に使ったペアなど）が必要ならここで行う
        # 今回はシンプルに「正解が含まれているか」だけを見る
        
        first_hit_rank = 0
        for rank, doi in enumerate(found_dois, 1):
            if doi in ground_truth_set:
                first_hit_rank = rank
                break
        
        ranks.append(first_hit_rank)
        
        # 結果ログ用
        top_k_results.append({
            "query_doi": q_data["query_doi"],
            "first_hit_rank": first_hit_rank,
            "found_dois": found_dois[:10] # Top 10だけログに残す
        })

    # 6. メトリクス計算
    print("\n=== Evaluation Results ===")
    recall_scores = calculate_recall_at_k(ranks, cfg.evaluation.k_values)
    mrr = calculate_mrr(ranks)
    
    print(f"MRR: {mrr:.4f}")
    for k, score in recall_scores.items():
        print(f"Recall@{k}: {score:.4f}")
        
    # 7. 結果保存
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mrr": mrr,
        "recall": recall_scores,
        "details": top_k_results
    }
    
    out_file = os.path.join(cfg.data.output_dir, cfg.evaluation.result_file)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()