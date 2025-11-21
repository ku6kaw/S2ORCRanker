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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 候補リストのロード
    print(f"Loading candidates from: {cfg.data.candidates_file}")
    with open(cfg.data.candidates_file, 'r') as f:
        candidates_data = json.load(f)
    
    # 2. Cross-Encoderのロード
    print(f"Loading Cross-Encoder: {cfg.model.path}")
    try:
        config = AutoConfig.from_pretrained(cfg.model.path)
        # num_labels=1 を確認（回帰/ランキングスコア）
        config.num_labels = 1
        model = CrossEncoderMarginModel.from_pretrained(cfg.model.path, config=config)
    except Exception as e:
        print(f"Error loading model: {e}. Loading from base.")
        config = AutoConfig.from_pretrained(cfg.model.base_name)
        config.num_labels = 1
        model = CrossEncoderMarginModel.from_pretrained(cfg.model.base_name, config=config)
        
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    # 3. リランキング実行
    final_ranks = []
    
    if not os.path.exists(cfg.data.output_dir):
        os.makedirs(cfg.data.output_dir)

    for query_item in tqdm(candidates_data, desc="Reranking Queries"):
        query_text = query_item["query_text"]
        candidate_dois = query_item["retrieved_candidates"]
        
        # 候補の本文をDBから取得
        # (数が多い場合はここがボトルネックになる可能性あり。必要なら事前に一括取得)
        doi_to_text = get_abstracts_from_db(candidate_dois, cfg.data.db_path)
        
        # 本文がある候補のみペアを作成
        valid_pairs = [] # (doi, text)
        for doi in candidate_dois:
            if doi in doi_to_text:
                valid_pairs.append((doi, doi_to_text[doi]))
        
        if not valid_pairs:
            final_ranks.append(0)
            continue

        # バッチ処理でスコアリング
        scores = []
        batch_size = cfg.model.batch_size
        
        for i in range(0, len(valid_pairs), batch_size):
            batch = valid_pairs[i : i + batch_size]
            batch_texts = [p[1] for p in batch]
            
            # Cross-Encoder入力: [CLS] Query [SEP] Document [SEP]
            inputs = tokenizer(
                [query_text] * len(batch), # Queryをバッチ分複製
                batch_texts,
                padding=True, truncation=True,
                max_length=cfg.model.max_length,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                # forward(input_ids, ...) -> logits (score_positive, score_negative)
                # 評価時は正例ペアとして入力し、score_positiveを見る
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                # logits: (Batch, 2) -> (pos_score, neg_score)
                # 今回の実装では (score, None) が返ってくるはず（負例入力がないため）
                # CrossEncoderMarginModel.forward の戻り値を確認:
                # logits=(score_positive, score_negative)
                batch_scores = outputs.logits[0].cpu().numpy().flatten()
                scores.extend(batch_scores)
        
        # スコア順にソート
        # valid_pairs と scores は対応している
        scored_candidates = []
        for j, (doi, _) in enumerate(valid_pairs):
            scored_candidates.append((doi, scores[j]))
            
        # スコア降順ソート
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        sorted_dois = [x[0] for x in scored_candidates]
        
        # 再評価 (Rank計算)
        ground_truth_set = set(query_item["ground_truth_dois"])
        first_hit_rank = 0
        for rank, doi in enumerate(sorted_dois, 1):
            if doi in ground_truth_set:
                first_hit_rank = rank
                break
        
        # リトリーバーで見つからなかった(リストに含まれていなかった)場合は、
        # リランキング後も「見つからない(0)」とするか、
        # もとの順位を維持するか？ → 通常は候補に入らなければ0（失敗）。
        final_ranks.append(first_hit_rank)

    # 4. 最終結果出力
    print("\n=== Reranking Results ===")
    recall_scores = calculate_recall_at_k(final_ranks, cfg.evaluation.k_values)
    mrr = calculate_mrr(final_ranks)
    
    print(f"MRR: {mrr:.4f}")
    for k, score in recall_scores.items():
        print(f"Recall@{k}: {score:.4f}")
        
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mrr": mrr,
        "recall": recall_scores,
        "ranks": final_ranks
    }
    
    with open(os.path.join(cfg.data.output_dir, cfg.evaluation.result_file), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()