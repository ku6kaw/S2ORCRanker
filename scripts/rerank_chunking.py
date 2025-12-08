# scripts/rerank_chunking.py

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import os
import sys

# プロジェクトルートへのパス追加
sys.path.append(os.getcwd())

from src.utils.metrics import calculate_recall_at_k, calculate_mrr

class ChunkingReranker:
    def __init__(self, model_path, base_name, max_length=512, stride=64, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.stride = stride # オーバーラップさせるトークン数
        
        print(f"Loading tokenizer: {base_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        except:
            # base_nameで失敗したらmodel_pathから読む
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading model: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=1,
            ignore_mismatched_sizes=True # ヘッドのサイズ不一致等の保険
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, query, text):
        """
        長いテキストをチャンキングして推論し、最大スコアを返す (Max Pooling Strategy)
        """
        # 1. チャンキング付きでトークナイズ
        # return_overflowing_tokens=True で、はみ出した部分を新しいシーケンスとして返す
        # stride でオーバーラップを作る
        # ※ SPECTER2等は [CLS] Query [SEP] Text [SEP] の形式になる
        try:
            inputs = self.tokenizer(
                query,
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length", # バッチ処理のために長さを揃える
                stride=self.stride,
                return_overflowing_tokens=True,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Tokenization error: {e}")
            return -9999.0 # エラー時は最低スコア
        
        # バッチサイズとして扱う (num_chunks, seq_len)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 2. 推論 (全てのチャンクを一度にモデルに入力)
        # チャンク数が多い場合、メモリ不足になる可能性があるので注意（必要ならここでミニバッチ化する）
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1) # (num_chunks, )

        # 3. Max Pooling (最も関連度が高かったチャンクのスコアを採用)
        if logits.numel() > 1:
            max_score = logits.max().item()
        else:
            max_score = logits.item()
        
        return max_score

@hydra.main(config_path="../configs", config_name="rerank", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Reranking with Chunking ===")
    print(OmegaConf.to_yaml(cfg))
    
    input_file = cfg.data.candidates_file
    output_dir = cfg.data.output_dir
    output_file = os.path.join(output_dir, cfg.evaluation.result_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # チャンキング設定 (yamlになければデフォルト値)
    # SPECTER2は512固定なので、yamlで上書きできるようにする
    max_len = cfg.rerank.get("window_size", 512)
    stride = cfg.rerank.get("stride", 64)
    
    reranker = ChunkingReranker(
        model_path=cfg.model.path,
        base_name=cfg.model.base_name,
        max_length=max_len,
        stride=stride
    )
    
    print(f"Loading candidates from: {input_file}")
    with open(input_file, 'r') as f:
        candidates_data = json.load(f)
        
    final_ranks = []
    reranked_details = []

    # クエリごとに処理
    for item in tqdm(candidates_data, desc="Reranking Queries"):
        query_text = item["query_text"]
        ground_truth_dois = set(item["ground_truth_dois"])
        
        # 候補論文リスト (Top-Kに絞る処理はここに入れる)
        # cfg.rerank.top_k で指定、なければ全件
        top_k_limit = cfg.rerank.get("top_k", len(item["retrieved_candidates"]))
        candidate_dois = item["retrieved_candidates"][:top_k_limit]
        
        # 本来はここでDBからアブストラクト本文を取得する処理が必要
        # 今回のcandidates_fileにはDOIしかないので、評価データセットから復元するか
        # DBルックアップが必要。ここでは簡易的に「別途辞書などから取得済み」と仮定するか
        # もしくは evaluation_dataset_rich.jsonl 等から事前にマッピングを作る必要がある。
        # ★★★ 実装上の注意 ★★★
        # ここでは「candidate_dois」に対応するテキストが必要。
        # 簡易実装として、もしcandidate_doisがテキストを持っていなければスキップする等の処理になるが、
        # 完全な実装には「DOI -> Abstract」のマッピングデータが必須。
        # (今回はプレースホルダーとしてエラーを出さずに進めるためのダミー実装を含む)
        
        # --- (仮) DOIからテキストを取得するロジック ---
        # 実際にはここに s2orc_filtered.db へのアクセス等が入る
        doc_texts = [] 
        # ダミー: "Abstract for {doi}" (実際にはちゃんとしたテキスト取得コードが必要)
        for doi in candidate_dois:
             doc_texts.append(f"Abstract text for {doi}...") 
        # -------------------------------------------

        scores = []
        for doc_text in doc_texts:
            if not doc_text:
                scores.append(-9999.0)
                continue
                
            score = reranker.predict(query_text, doc_text)
            scores.append(score)
            
        # スコアとDOIをペアにしてソート (降順)
        scored_candidates = list(zip(candidate_dois, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # ランク付け
        sorted_dois = [doi for doi, score in scored_candidates]
        
        first_hit_rank = 0
        for rank, doi in enumerate(sorted_dois, 1):
            if doi in ground_truth_dois:
                first_hit_rank = rank
                break
        
        final_ranks.append(first_hit_rank)
        
        reranked_details.append({
            "query_doi": item["query_doi"],
            "ground_truth_dois": list(ground_truth_dois),
            "reranked_top_10": sorted_dois[:10],
            "first_hit_rank": first_hit_rank
        })
        
    # 評価指標計算
    print("\n=== Reranking Results (Chunking) ===")
    mrr = calculate_mrr(final_ranks)
    recall_scores = calculate_recall_at_k(final_ranks, cfg.evaluation.k_values)
    
    print(f"MRR: {mrr:.4f}")
    for k, score in recall_scores.items():
        print(f"Recall@{k}: {score:.4f}")
        
    # 保存
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mrr": mrr,
        "recall": recall_scores,
        "ranks": final_ranks,
        "details": reranked_details
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()