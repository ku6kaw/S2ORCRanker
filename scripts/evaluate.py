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

sys.path.append(os.getcwd())
from src.modeling.bi_encoder import SiameseBiEncoder
from src.utils.cleaning import clean_text
from src.utils.metrics import calculate_recall_at_k, calculate_mrr

def load_ground_truth(csv_path, db_path):
    """評価用CSVとDBから、クエリと正解ペア(DOI)のリストを作成する"""
    df_eval = pd.read_csv(csv_path)
    eval_data_paper_dois = df_eval['cited_datapaper_doi'].unique()
    
    queries = []
    
    with sqlite3.connect(db_path) as conn:
        for dp_doi in tqdm(eval_data_paper_dois, desc="Loading Ground Truth"):
            try:
                # テーブル名はプロジェクトに合わせて調整
                query_gt = "SELECT citing_doi FROM positive_candidates WHERE cited_datapaper_doi = ? AND human_annotation_status = 1"
                gt_rows = conn.execute(query_gt, (dp_doi,)).fetchall()
            except sqlite3.OperationalError:
                continue
                
            ground_truth_dois = {row[0] for row in gt_rows}
            
            if len(ground_truth_dois) >= 2:
                query_doi = list(ground_truth_dois)[0]
                target_dois = list(ground_truth_dois)[1:]
                
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
    
    # 1. ロード処理
    embeddings_path = os.path.join(cfg.data.output_dir, cfg.data.embeddings_file)
    doi_map_path = os.path.join(cfg.data.output_dir, cfg.data.doi_map_file)
    
    print(f"Loading DOI map from {doi_map_path}...")
    with open(doi_map_path, 'r') as f:
        doi_list = json.load(f)
    
    num_vectors = len(doi_list)
    hidden_size = 768 
    
    print(f"Loading embeddings (mmap)...")
    corpus_embeddings = np.memmap(
        embeddings_path, dtype='float32', mode='r', shape=(num_vectors, hidden_size)
    )
    
    # 2. Faissインデックス構築
    print("Building Faiss Index...")
    if cfg.evaluation.use_faiss:
        index = faiss.IndexFlatIP(hidden_size)
        if cfg.evaluation.gpu_search and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        batch_size = 50000
        for i in tqdm(range(0, num_vectors, batch_size), desc="Indexing"):
            batch_vecs = np.array(corpus_embeddings[i : i + batch_size])
            faiss.normalize_L2(batch_vecs)
            index.add(batch_vecs)
    else:
        print("Faiss is required for this script.")
        return

    # 3. モデルロード
    print(f"Loading query encoder: {cfg.model.path}")
    try:
        config = AutoConfig.from_pretrained(cfg.model.path)
        model = SiameseBiEncoder.from_pretrained(cfg.model.path, config=config)
    except:
        model = SiameseBiEncoder.from_pretrained(cfg.model.base_name)
        
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    # 4. 評価実行
    queries = load_ground_truth(cfg.data.eval_data_path, cfg.data.db_path)
    if not queries:
        print("No queries found.")
        return

    ranks = []
    candidates_list = [] # リランキング用候補リスト
    
    max_k = max(cfg.evaluation.k_values)
    save_k = cfg.evaluation.get("candidates_k", 1000)
    search_k = max(max_k, save_k) + 50 # 余裕を持って検索
    
    for q_data in tqdm(queries, desc="Evaluating"):
        inputs = tokenizer(
            q_data["query_text"], padding=True, truncation=True, 
            max_length=cfg.model.max_length, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_vectors=True)
            q_vec = out.logits.cpu().numpy()
        
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, search_k)
        
        found_indices = I[0]
        # インデックス -> DOI
        found_dois = [doi_list[idx] for idx in found_indices if idx >= 0 and idx < len(doi_list)]
        
        # ランク計算
        ground_truth_set = set(q_data["ground_truth_dois"])
        first_hit_rank = 0
        for rank, doi in enumerate(found_dois, 1):
            if doi in ground_truth_set:
                first_hit_rank = rank
                break
        ranks.append(first_hit_rank)
        
        # ▼▼▼ 候補リストの保存（リランキング用） ▼▼▼
        # クエリ情報、正解情報、検索された候補（Top-K）を保存
        candidates_list.append({
            "query_doi": q_data["query_doi"],
            "query_text": q_data["query_text"], # リランカーへの入力として重要
            "ground_truth_dois": q_data["ground_truth_dois"],
            "retrieved_candidates": found_dois[:save_k], # 指定件数まで保存
            "first_hit_rank_retriever": first_hit_rank
        })

    # 5. 結果保存
    print("\n=== Evaluation Results (Retriever) ===")
    recall_scores = calculate_recall_at_k(ranks, cfg.evaluation.k_values)
    mrr = calculate_mrr(ranks)
    print(f"MRR: {mrr:.4f}")
    
    # メトリクス保存
    out_file = os.path.join(cfg.data.output_dir, cfg.evaluation.result_file)
    with open(out_file, 'w') as f:
        json.dump({"mrr": mrr, "recall": recall_scores}, f, indent=2)
        
    # 候補リスト保存
    if cfg.evaluation.get("save_candidates", False):
        cand_file = os.path.join(cfg.data.output_dir, cfg.evaluation.candidates_file)
        print(f"Saving candidates for reranking to {cand_file}...")
        with open(cand_file, 'w') as f:
            json.dump(candidates_list, f, indent=2)

if __name__ == "__main__":
    main()