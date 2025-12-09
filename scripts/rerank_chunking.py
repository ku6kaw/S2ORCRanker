# scripts/rerank_chunking.py

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import json
import os
import sys

# Hydra起動前なので、標準の os.getcwd() を使用してプロジェクトルートをパスに追加
sys.path.append(os.getcwd())

from src.utils.metrics import calculate_recall_at_k, calculate_mrr
# ★追加: 学習時と同じモデルクラスをインポート（これが無いと重みが正しく読めません）
from src.modeling.cross_encoder import CrossEncoderMarginModel 

class ChunkingReranker:
    def __init__(self, model_path, base_name, max_length=512, stride=64, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.stride = stride 
        
        print(f"Loading tokenizer: {base_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        print(f"Loading model: {model_path}")
        try:
            # 学習時と同じクラス構造でロードする（必須）
            self.model = CrossEncoderMarginModel.from_pretrained(model_path)
            print("✅ Successfully loaded with CrossEncoderMarginModel")
        except Exception as e:
            # ここに来る＝何かがおかしい（重み不一致など）
            print(f"❌ Failed to load with CrossEncoderMarginModel: {e}")
            print("⚠️ WARNING: Falling back to AutoModel. Precision may drop significantly!")
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=1, ignore_mismatched_sizes=True
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, query, text):
        """
        長いテキストをチャンキングして推論し、最大スコアを返す (Max Pooling Strategy)
        """
        try:
            inputs = self.tokenizer(
                query,
                text,
                max_length=self.max_length,
                truncation=True, # ここはTrueでOK（strideが効くのは自前でやる場合だが、HFのstride機能を使うならこれでOK）
                padding="max_length",
                stride=self.stride,
                return_overflowing_tokens=True, # これにより、512を超える入力が複数のバッチに分割される
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Tokenization error: {e}")
            return -9999.0
        
        # return_overflowing_tokens=True の場合、input_ids は [num_chunks, seq_len] になる
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # モデルによっては token_type_ids が必要な場合がある（BERTなど）
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        with torch.no_grad():
            if token_type_ids is not None:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # CrossEncoderMarginModel は (score_pos, score_neg) のタプルを返す仕様にしたため
            if isinstance(outputs.logits, tuple):
                logits = outputs.logits[0] # score_pos
            else:
                logits = outputs.logits

            logits = logits.squeeze(-1) # (num_chunks,)

        # チャンクごとのスコアの最大値をとる (Max Pooling)
        if logits.numel() > 1:
            max_score = logits.max().item()
        else:
            max_score = logits.item()
        
        return max_score

@hydra.main(config_path="../configs", config_name="rerank", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Reranking with Chunking ===")
    print(OmegaConf.to_yaml(cfg))
    
    # パスの解決
    input_file = to_absolute_path(cfg.data.candidates_file)
    output_dir = to_absolute_path(cfg.data.output_dir)
    model_path = to_absolute_path(cfg.model.path)
    
    output_file = os.path.join(output_dir, cfg.evaluation.result_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    max_len = cfg.rerank.get("window_size", 512)
    stride = cfg.rerank.get("stride", 64)
    
    reranker = ChunkingReranker(
        model_path=model_path,
        base_name=cfg.model.base_name,
        max_length=max_len,
        stride=stride
    )
    
    print(f"Loading candidates from: {input_file}")
    with open(input_file, 'r') as f:
        candidates_data = json.load(f)
        
    final_ranks = []
    reranked_details = []

    for item in tqdm(candidates_data, desc="Reranking Queries"):
        query_text = item["query_text"]
        ground_truth_dois = set(item["ground_truth_dois"])
        
        top_k_limit = cfg.rerank.get("top_k", len(item["retrieved_candidates"]))
        candidate_dois = item["retrieved_candidates"][:top_k_limit]
        
        candidate_texts_map = item.get("candidate_texts", {})
        
        doc_texts = []
        for doi in candidate_dois:
            text = candidate_texts_map.get(doi, "")
            doc_texts.append(text)

        scores = []
        for doc_text in doc_texts:
            if not doc_text or len(doc_text) < 10:
                scores.append(-9999.0)
                continue
                
            score = reranker.predict(query_text, doc_text)
            scores.append(score)
            
        scored_candidates = list(zip(candidate_dois, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
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
        
    print("\n=== Reranking Results (Chunking) ===")
    mrr = calculate_mrr(final_ranks)
    recall_scores = calculate_recall_at_k(final_ranks, cfg.evaluation.k_values)
    
    print(f"MRR: {mrr:.4f}")
    for k, score in recall_scores.items():
        print(f"HitRate@{k}: {score:.4f}")
        
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mrr": mrr,
        "hit_rate": recall_scores,
        "ranks": final_ranks,
        "details": reranked_details
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()