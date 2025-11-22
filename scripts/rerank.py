# scripts/rerank.py

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import json
import sqlite3
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig
import wandb

sys.path.append(os.getcwd())
from src.modeling.cross_encoder import CrossEncoderMarginModel
from src.utils.cleaning import clean_text
from src.utils.metrics import calculate_recall_at_k, calculate_mrr

def get_abstracts_from_db(dois, db_path):
    """DOIリストに対応するアブストラクトをDBから一括取得"""
    doi_to_text = {}
    with sqlite3.connect(db_path) as conn:
        chunk_size = 900
        for i in range(0, len(dois), chunk_size):
            chunk = dois[i:i+chunk_size]
            placeholders = ','.join('?' for _ in chunk)
            rows = conn.execute(f"SELECT doi, abstract FROM papers WHERE doi IN ({placeholders})", chunk).fetchall()
            for row in rows:
                if row[1]:
                    doi_to_text[row[0]] = clean_text(row[1])
    return doi_to_text

@hydra.main(config_path="../configs", config_name="rerank", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Reranking ===")
    print(OmegaConf.to_yaml(cfg))
    
    # ★ WandB初期化
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 候補ロード
    print(f"Loading candidates from: {cfg.data.candidates_file}")
    with open(cfg.data.candidates_file, 'r') as f:
        candidates_data = json.load(f)
    
    # 2. モデルロード
    print(f"Loading Cross-Encoder: {cfg.model.path}")
    try:
        config = AutoConfig.from_pretrained(cfg.model.path)
        config.num_labels = 1
        model = CrossEncoderMarginModel.from_pretrained(cfg.model.path, config=config)
    except Exception as e:
        print(f"Loading from base due to error: {e}")
        config = AutoConfig.from_pretrained(cfg.model.base_name)
        config.num_labels = 1
        model = CrossEncoderMarginModel.from_pretrained(cfg.model.base_name, config=config)
        
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    # 3. リランキング実行
    final_ranks = []
    # ★ WandB用テーブルデータ
    wandb_table_data = []
    
    if not os.path.exists(cfg.data.output_dir):
        os.makedirs(cfg.data.output_dir)

    for query_item in tqdm(candidates_data, desc="Reranking Queries"):
        query_text = query_item["query_text"]
        candidate_dois = query_item["retrieved_candidates"]
        # 以前の順位（Retrieverの順位）を取得。0なら見つからなかった
        retriever_rank = query_item.get("first_hit_rank_retriever", 0)
        
        doi_to_text = get_abstracts_from_db(candidate_dois, cfg.data.db_path)
        valid_pairs = []
        for doi in candidate_dois:
            if doi in doi_to_text:
                valid_pairs.append((doi, doi_to_text[doi]))
        
        first_hit_rank = 0 # 初期値
        
        if valid_pairs:
            scores = []
            batch_size = cfg.model.batch_size
            
            for i in range(0, len(valid_pairs), batch_size):
                batch = valid_pairs[i : i + batch_size]
                batch_texts = [p[1] for p in batch]
                inputs = tokenizer(
                    [query_text] * len(batch), batch_texts,
                    padding=True, truncation=True,
                    max_length=cfg.model.max_length, return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    batch_scores = outputs.logits[0].cpu().numpy().flatten()
                    scores.extend(batch_scores)
            
            scored_candidates = []
            for j, (doi, _) in enumerate(valid_pairs):
                scored_candidates.append((doi, scores[j]))
                
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            sorted_dois = [x[0] for x in scored_candidates]
            
            ground_truth_set = set(query_item["ground_truth_dois"])
            for rank, doi in enumerate(sorted_dois, 1):
                if doi in ground_truth_set:
                    first_hit_rank = rank
                    break
        
        # リトリーバーで見つかっていない(0)場合、リランカーでも0のまま
        # リトリーバーで見つかった(>0)が、候補内(Top-K)に入っていなければ
        # ここでの first_hit_rank も 0 になる。
        
        # 最終ランクの決定:
        # もし候補内に正解があればその順位、なければ0
        final_ranks.append(first_hit_rank)

        # ★ WandB Tableに追加 (順位変動の可視化)
        if cfg.logging.use_wandb:
            # 改善幅 (正の値なら改善、負なら悪化)
            # 0 (圏外) の扱いに注意
            if retriever_rank > 0 and first_hit_rank > 0:
                improvement = retriever_rank - first_hit_rank
            elif retriever_rank == 0 and first_hit_rank > 0:
                improvement = 9999 # 圏外から発見（ありえないが）
            elif retriever_rank > 0 and first_hit_rank == 0:
                improvement = -9999 # 候補落ち
            else:
                improvement = 0 # どちらも圏外

            wandb_table_data.append([
                query_item["query_doi"],
                query_text[:100],
                retriever_rank,
                first_hit_rank,
                improvement
            ])

    # 4. 最終結果出力
    print("\n=== Reranking Results ===")
    recall_scores = calculate_recall_at_k(final_ranks, cfg.evaluation.k_values)
    mrr = calculate_mrr(final_ranks)
    print(f"MRR: {mrr:.4f}")
    
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mrr": mrr,
        "recall": recall_scores,
        "ranks": final_ranks
    }
    
    with open(os.path.join(cfg.data.output_dir, cfg.evaluation.result_file), 'w') as f:
        json.dump(results, f, indent=2)

    # ★ WandBへログ送信
    if cfg.logging.use_wandb:
        log_dict = {"mrr": mrr}
        for k, score in recall_scores.items():
            log_dict[f"recall@{k}"] = score
        wandb.log(log_dict)
        
        columns = ["Query DOI", "Query Text", "Retriever Rank", "Reranker Rank", "Improvement"]
        table = wandb.Table(data=wandb_table_data, columns=columns)
        wandb.log({"reranking_details": table})
        
        wandb.finish()

if __name__ == "__main__":
    main()