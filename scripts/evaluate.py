# scripts/evaluate.py

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import json
import faiss
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig
import wandb

sys.path.append(os.getcwd())
from src.modeling.bi_encoder import SiameseBiEncoder
from src.utils.cleaning import clean_text
from src.utils.metrics import calculate_recall_at_k, calculate_mrr

def load_queries_from_jsonl(jsonl_path):
    """作成済みの評価用JSONLファイルを読み込む"""
    print(f"Loading queries from {jsonl_path}...")
    queries = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {jsonl_path}")

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not data.get("query_abstract"):
                continue
            queries.append({
                "query_doi": data["query_doi"],
                "query_text": data["query_abstract"],
                "ground_truth_dois": data["ground_truth_dois"],
                "data_paper_doi": data.get("metadata", {}).get("source_datapaper", "")
            })
    return queries

def calculate_full_ranks(query_vec, gt_indices, corpus_embeddings, device, batch_size=100000):
    """
    全件スキャンを行い、正解(gt_indices)の正確な順位を計算する。
    ソートを行わず「自分よりスコアが高い件数」を数えるためメモリ効率が良い。
    """
    num_docs = corpus_embeddings.shape[0]
    
    # 正解のベクトルを取得してスコア計算
    # corpus_embeddingsはmemmapなので、必要な行だけ読み込む
    gt_vecs = torch.tensor(corpus_embeddings[gt_indices], device=device) # (Num_GT, Dim)
    q_vec_t = torch.tensor(query_vec, device=device).unsqueeze(1)        # (Dim, 1)
    
    # 正解のスコア: (Num_GT, 1)
    gt_scores = torch.matmul(gt_vecs, q_vec_t).squeeze(1) # (Num_GT,)
    
    # ランクの初期値（自分自身が含まれるため1位スタート）
    # ここでは「自分より高いスコアの数 + 1」をランクとする
    ranks = torch.ones(len(gt_indices), dtype=torch.long, device=device)
    
    # 全件スキャンループ
    for i in range(0, num_docs, batch_size):
        end = min(i + batch_size, num_docs)
        
        # バッチ読み込み & GPU転送
        batch_vecs = torch.tensor(corpus_embeddings[i:end], device=device)
        
        # スコア計算 (Batch, Dim) @ (Dim, 1) -> (Batch, 1)
        batch_scores = torch.matmul(batch_vecs, q_vec_t).squeeze(1) # (Batch,)
        
        # 比較: 各正解スコアより高いものがバッチ内にいくつあるか
        # gt_scores: (Num_GT, 1)
        # batch_scores: (1, Batch)
        # comparison: (Num_GT, Batch) -> True if batch_score > gt_score
        # sum(dim=1) -> (Num_GT,)
        
        # メモリ節約のため、正解ごとにループせず放送（Broadcasting）を使う
        # 正解数が多いとメモリを食うが、通常数十件なのでOK
        better_counts = (batch_scores.unsqueeze(0) > gt_scores.unsqueeze(1)).sum(dim=1)
        ranks += better_counts
        
    return ranks.cpu().numpy().tolist()

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Evaluation ===")
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. ロード処理
    embeddings_path = os.path.join(cfg.data.output_dir, cfg.data.embeddings_file)
    doi_map_path = os.path.join(cfg.data.output_dir, cfg.data.doi_map_file)
    
    print(f"Loading DOI map from {doi_map_path}...")
    with open(doi_map_path, 'r') as f:
        doi_list = json.load(f)
    
    # DOI -> Index の逆引きマップ
    doi_to_index = {doi: i for i, doi in enumerate(doi_list)}
    num_vectors = len(doi_list)
    
    print(f"Loading embeddings (mmap) from {embeddings_path}...")
    
    # SciBERTの次元数は768固定、または設定から取得
    hidden_size = 768 
    
    # Memmapロード (Read-only)
    corpus_embeddings = np.memmap(
        embeddings_path, 
        dtype='float32', 
        mode='r', 
        shape=(num_vectors, hidden_size)
    )
    
    corpus_embeddings = np.memmap(
        embeddings_path, dtype='float32', mode='r', shape=(num_vectors, hidden_size)
    )
    
    # 2. Faissインデックス構築 (候補抽出用)
    # 全件ランク計算を行う場合でも、Top-K候補の保存にはFaissが便利
    index = None
    if cfg.evaluation.use_faiss:
        print("Building Faiss Index for candidate retrieval...")
        index = faiss.IndexFlatIP(hidden_size)
        if cfg.evaluation.gpu_search and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                print(f"GPU Faiss failed: {e}. Using CPU.")
        
        # メモリに乗らない場合はここで工夫が必要だが、一旦全件追加
        # バッチで追加
        batch_size = 50000
        for i in tqdm(range(0, num_vectors, batch_size), desc="Indexing"):
            batch_vecs = np.array(corpus_embeddings[i : i + batch_size])
            faiss.normalize_L2(batch_vecs)
            index.add(batch_vecs)

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
    queries = load_queries_from_jsonl(cfg.data.eval_jsonl_file)
    print(f"Loaded {len(queries)} queries.")

    ranks = [] # First Hit Rankのリスト
    candidates_list = []
    wandb_table_data = []
    
    max_k = max(cfg.evaluation.k_values)
    save_k = cfg.evaluation.get("candidates_k", 1000)
    search_k = max(max_k, save_k) + 50
    
    # 全件ランク計算用のバッチサイズ（VRAMに合わせて調整）
    scan_batch_size = 500000 

    for q_data in tqdm(queries, desc="Evaluating"):
        # クエリベクトル化
        inputs = tokenizer(
            q_data["query_text"], padding=True, truncation=True, 
            max_length=cfg.model.max_length, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_vectors=True)
            q_vec = out.logits.cpu().numpy().flatten() # (Dim,)
        
        # --- A. Faissによる候補抽出 (Top-K) ---
        faiss_q_vec = q_vec.reshape(1, -1).copy()
        faiss.normalize_L2(faiss_q_vec)
        
        found_dois = []
        if index is not None:
            D, I = index.search(faiss_q_vec, search_k)
            found_indices = I[0]
            found_dois = [doi_list[idx] for idx in found_indices if idx >= 0 and idx < len(doi_list)]
        
        # --- B. 全件ランク計算 (Full Scan) ---
        all_gt_ranks = []
        first_hit_rank = 0
        
        ground_truth_dois = q_data["ground_truth_dois"]
        gt_indices = [doi_to_index[doi] for doi in ground_truth_dois if doi in doi_to_index]
        
        if not gt_indices:
            # 正解がDBに存在しない場合
            print(f"Warning: GT DOIs not found in DB for query {q_data['query_doi']}")
            ranks.append(0)
            continue

        if cfg.evaluation.get("calc_full_rank", False):
            # 全件スキャンで正確なランクを計算
            # ※ q_vec は正規化前を使う（内積 = スコア）
            all_gt_ranks = calculate_full_ranks(
                q_vec, gt_indices, corpus_embeddings, device, batch_size=scan_batch_size
            )
            # 昇順ソート（1位が最高）
            all_gt_ranks.sort()
            first_hit_rank = all_gt_ranks[0]
        else:
            # Faissの結果から推定（圏外なら0）
            ground_truth_set = set(ground_truth_dois)
            for rank, doi in enumerate(found_dois, 1):
                if doi in ground_truth_set:
                    first_hit_rank = rank
                    # Faissのみの場合は最初の1つしか正確なランクがわからない
                    all_gt_ranks.append(rank)
                    break
            if first_hit_rank == 0:
                # 圏外
                all_gt_ranks = [0] * len(ground_truth_dois)

        ranks.append(first_hit_rank)
        
        # 候補リスト保存
        candidates_list.append({
            "query_doi": q_data["query_doi"],
            "query_text": q_data["query_text"],
            "ground_truth_dois": q_data["ground_truth_dois"],
            "retrieved_candidates": found_dois[:save_k],
            "first_hit_rank_retriever": first_hit_rank,
            "all_gt_ranks": all_gt_ranks # 全正解のランクも保存
        })

        # WandB Table
        if cfg.logging.use_wandb:
            # ランク一覧を文字列化
            rank_str = ", ".join(map(str, all_gt_ranks[:10])) # 多すぎると見づらいのでTop10のみ
            if len(all_gt_ranks) > 10:
                rank_str += "..."
            
            wandb_table_data.append([
                q_data["query_doi"],
                q_data["query_text"][:100],
                len(ground_truth_dois),
                first_hit_rank,
                rank_str, # すべての順位
                found_dois[0] if found_dois else "None"
            ])

    # 5. 結果保存・ログ送信
    print("\n=== Evaluation Results (Retriever) ===")
    recall_scores = calculate_recall_at_k(ranks, cfg.evaluation.k_values)
    mrr = calculate_mrr(ranks)
    print(f"MRR: {mrr:.4f}")
    for k, score in recall_scores.items():
        print(f"Recall@{k}: {score:.4f}")
    
    out_file = os.path.join(cfg.data.output_dir, cfg.evaluation.result_file)
    with open(out_file, 'w') as f:
        json.dump({"mrr": mrr, "recall": recall_scores}, f, indent=2)
        
    if cfg.evaluation.get("save_candidates", False):
        cand_file = os.path.join(cfg.data.output_dir, cfg.evaluation.candidates_file)
        print(f"Saving candidates for reranking to {cand_file}...")
        with open(cand_file, 'w') as f:
            json.dump(candidates_list, f, indent=2)

    if cfg.logging.use_wandb:
        log_dict = {}
        log_dict["mrr"] = float(mrr)
        for k, score in recall_scores.items():
            log_dict[f"recall/recall@{k}"] = float(score)
        wandb.log(log_dict)
        
        columns = ["Query DOI", "Query Text", "Num GT", "First Hit Rank", "All GT Ranks", "Top-1 DOI"]
        safe_table_data = []
        for row in wandb_table_data:
            safe_table_data.append([str(x) for x in row])
            
        table = wandb.Table(data=safe_table_data, columns=columns)
        wandb.log({"retrieval_details": table})
        wandb.finish()

if __name__ == "__main__":
    main()