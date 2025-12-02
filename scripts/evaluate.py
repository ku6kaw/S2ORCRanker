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
import random
import adapters

sys.path.append(os.getcwd())
from src.modeling.bi_encoder import SiameseBiEncoder
from src.utils.cleaning import clean_text
from src.utils.metrics import calculate_recall_at_k, calculate_mrr

def load_queries_from_jsonl(jsonl_path, queries_per_dataset=None, seed=42):
    print(f"Loading queries from {jsonl_path}...")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {jsonl_path}")

    all_queries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not data.get("query_abstract"):
                continue
            all_queries.append({
                "query_doi": data["query_doi"],
                "query_text": data["query_abstract"],
                "ground_truth_dois": data["ground_truth_dois"],
                "data_paper_doi": data.get("metadata", {}).get("source_datapaper", "unknown")
            })
    
    if queries_per_dataset and queries_per_dataset > 0:
        print(f"Sampling {queries_per_dataset} query(s) per dataset...")
        random.seed(seed) 
        grouped = {}
        for q in all_queries:
            src = q["data_paper_doi"]
            if src not in grouped:
                grouped[src] = []
            grouped[src].append(q)
        
        sampled_queries = []
        for src, group in grouped.items():
            if len(group) > queries_per_dataset:
                selected = random.sample(group, queries_per_dataset)
            else:
                selected = group
            sampled_queries.extend(selected)
        print(f"Reduced queries from {len(all_queries)} to {len(sampled_queries)}.")
        return sampled_queries

    return all_queries

def calculate_full_ranks(query_vec, gt_indices, corpus_embeddings, device, batch_size=100000):
    num_docs = corpus_embeddings.shape[0]
    # float16のままだと精度が落ちる可能性があるため、計算時にfloat32へキャストしても良いが、
    # GPUメモリ節約のためfloat16で計算し、必要ならtensor作成時にdtypeを指定する
    # ここでは元のnumpy配列の型に従う
    gt_vecs = torch.tensor(corpus_embeddings[gt_indices], device=device) 
    q_vec_t = torch.tensor(query_vec, device=device).unsqueeze(1)
    
    # 型を合わせる（q_vecがfloat32なら合わせる）
    if gt_vecs.dtype != q_vec_t.dtype:
        gt_vecs = gt_vecs.to(q_vec_t.dtype)

    gt_scores = torch.matmul(gt_vecs, q_vec_t).squeeze(1)
    ranks = torch.ones(len(gt_indices), dtype=torch.long, device=device)
    
    for i in range(0, num_docs, batch_size):
        end = min(i + batch_size, num_docs)
        batch_vecs = torch.tensor(corpus_embeddings[i:end], device=device)
        if batch_vecs.dtype != q_vec_t.dtype:
            batch_vecs = batch_vecs.to(q_vec_t.dtype)
            
        batch_scores = torch.matmul(batch_vecs, q_vec_t).squeeze(1)
        better_counts = (batch_scores.unsqueeze(0) > gt_scores.unsqueeze(1)).sum(dim=1)
        ranks += better_counts
        
    return ranks.cpu().numpy().tolist()

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Evaluation (float16 Support) ===")
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    embeddings_path = os.path.join(cfg.data.output_dir, cfg.data.embeddings_file)
    doi_map_path = os.path.join(cfg.data.output_dir, cfg.data.doi_map_file)
    
    print(f"Loading DOI map from {doi_map_path}...")
    with open(doi_map_path, 'r') as f:
        doi_list = json.load(f)
    
    doi_to_index = {doi: i for i, doi in enumerate(doi_list)}
    num_vectors = len(doi_list)
    
    print(f"Loading embeddings (mmap) from {embeddings_path}...")
    hidden_size = 768 
    
    # ▼▼▼ 修正: float16 で読み込む ▼▼▼
    corpus_embeddings = np.memmap(
        embeddings_path, 
        dtype='float16',  # float32 -> float16
        mode='r', 
        shape=(num_vectors, hidden_size)
    )
    
    index = None
    if cfg.evaluation.use_faiss:
        print("Building Faiss Index for candidate retrieval...")
        # Faissのインデックスは通常float32
        index = faiss.IndexFlatIP(hidden_size)
        if cfg.evaluation.gpu_search and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                print(f"GPU Faiss failed: {e}. Using CPU.")
        
        batch_size = 50000
        for i in tqdm(range(0, num_vectors, batch_size), desc="Indexing"):
            # ▼▼▼ 修正: Faissに入れる前に float32 に変換 ▼▼▼
            batch_vecs = np.array(corpus_embeddings[i : i + batch_size]).astype('float32')
            faiss.normalize_L2(batch_vecs)
            index.add(batch_vecs)

    print(f"Loading query encoder: {cfg.model.path}")
    try:
        config = AutoConfig.from_pretrained(cfg.model.path)
        model = SiameseBiEncoder.from_pretrained(cfg.model.path, config=config)
    except:
        model = SiameseBiEncoder.from_pretrained(cfg.model.base_name)
    
    adapter_name = cfg.model.get("adapter_name", None)
    if adapter_name:
        print(f"🔄 Loading Adapter config: {adapter_name}")
        adapters.init(model.bert)
        try:
            print(f"   Attempting to load adapter from checkpoint: {cfg.model.path}")
            loaded_name = model.bert.load_adapter(cfg.model.path, set_active=True)
        except Exception as e:
            print(f"   Checkpoint load failed ({e}), falling back to Hub: {adapter_name}")
            loaded_name = model.bert.load_adapter(adapter_name, source="hf", set_active=True)
        model.bert.set_active_adapters(loaded_name)
        print(f"✅ Adapter '{loaded_name}' activated.")

    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    queries_per_dataset = cfg.evaluation.get("queries_per_dataset", None)
    queries = load_queries_from_jsonl(cfg.data.eval_jsonl_file, queries_per_dataset, seed=cfg.seed)
    print(f"Loaded {len(queries)} queries for evaluation.")

    if not queries:
        print("No queries found.")
        return

    ranks = []
    candidates_list = []
    wandb_table_data = []
    
    max_k = max(cfg.evaluation.k_values)
    save_k = cfg.evaluation.get("candidates_k", 1000)
    search_k = max(max_k, save_k) + 50
    scan_batch_size = 500000 

    for q_data in tqdm(queries, desc="Evaluating"):
        inputs = tokenizer(
            q_data["query_text"], padding=True, truncation=True, 
            max_length=cfg.model.max_length, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_vectors=True)
            q_vec = out.logits.cpu().numpy().flatten()
        
        faiss_q_vec = q_vec.reshape(1, -1).copy()
        faiss.normalize_L2(faiss_q_vec)
        
        found_dois = []
        if index is not None:
            D, I = index.search(faiss_q_vec, search_k)
            found_indices = I[0]
            found_dois = [doi_list[idx] for idx in found_indices if idx >= 0 and idx < len(doi_list)]
        
        ground_truth_dois = q_data["ground_truth_dois"]
        gt_indices = [doi_to_index[doi] for doi in ground_truth_dois if doi in doi_to_index]
        
        all_gt_ranks = []
        first_hit_rank = 0
        
        if not gt_indices:
            ranks.append(0)
            continue

        if cfg.evaluation.get("calc_full_rank", False):
            # 全件ランク計算
            all_gt_ranks = calculate_full_ranks(q_vec, gt_indices, corpus_embeddings, device, batch_size=scan_batch_size)
            all_gt_ranks.sort()
            first_hit_rank = all_gt_ranks[0]
        else:
            ground_truth_set = set(ground_truth_dois)
            for rank, doi in enumerate(found_dois, 1):
                if doi in ground_truth_set:
                    first_hit_rank = rank
                    all_gt_ranks.append(rank)
                    break
            if first_hit_rank == 0:
                # 圏外
                all_gt_ranks = [0] * len(ground_truth_dois)

        ranks.append(first_hit_rank)
        
        candidates_list.append({
            "query_doi": q_data["query_doi"],
            "query_text": q_data["query_text"],
            "ground_truth_dois": q_data["ground_truth_dois"],
            "retrieved_candidates": found_dois[:save_k],
            "first_hit_rank_retriever": first_hit_rank,
            "all_gt_ranks": all_gt_ranks
        })

        if cfg.logging.use_wandb:
            rank_str = ", ".join(map(str, all_gt_ranks[:10]))
            if len(all_gt_ranks) > 10: rank_str += "..."
            wandb_table_data.append([
                q_data["query_doi"],
                q_data["query_text"][:100],
                len(ground_truth_dois),
                first_hit_rank,
                rank_str,
                found_dois[0] if found_dois else "None"
            ])

    print("\n=== Evaluation Results (Retriever) ===")
    recall_scores = calculate_recall_at_k(ranks, cfg.evaluation.k_values)
    mrr = calculate_mrr(ranks)
    print(f"MRR: {mrr:.4f}")
    for k, score in recall_scores.items():
        print(f"Recall@{k}: {score:.4f}")
    
    out_file = os.path.join(cfg.data.output_dir, cfg.evaluation.result_file)
    
    output_data = {
        "mrr": mrr, 
        "recall": recall_scores,
        "details": candidates_list
    }
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    if cfg.evaluation.get("save_candidates", False):
        cand_file = os.path.join(cfg.data.output_dir, cfg.evaluation.candidates_file)
        with open(cand_file, 'w') as f:
            json.dump(candidates_list, f, indent=2)

    if cfg.logging.use_wandb:
        log_dict = {"mrr": float(mrr)}
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