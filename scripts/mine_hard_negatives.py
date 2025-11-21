# scripts/mine_hard_negatives.py

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import torch
import faiss
import json
import sqlite3
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# srcへのパスを通す
sys.path.append(os.getcwd())
from src.utils.cleaning import clean_text

def load_resources(cfg, device):
    """モデルとFaissインデックスをロード"""
    print(f"Loading model: {cfg.mining.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.mining.model_name)
    model = AutoModel.from_pretrained(cfg.mining.model_name).to(device)
    model.eval()

    print(f"Loading Faiss index: {cfg.data.faiss_index_file}")
    index = faiss.read_index(cfg.data.faiss_index_file)
    
    print(f"Loading DOI map: {cfg.data.doi_map_file}")
    with open(cfg.data.doi_map_file, 'r') as f:
        doi_map = json.load(f)
    # ID -> DOI の逆引き辞書を作成
    id_to_doi = {v: k for k, v in doi_map.items()}
    
    return tokenizer, model, index, id_to_doi

def get_abstracts_from_db(dois, db_path):
    """SQLiteからDOIに対応するアブストラクトを一括取得"""
    doi_to_text = {}
    
    # DB接続
    with sqlite3.connect(db_path) as conn:
        chunk_size = 900 # SQLiteのプレースホルダー上限対策
        
        # 必要なDOIのみをユニーク化してリスト化
        unique_dois = list(set(dois))
        
        for i in tqdm(range(0, len(unique_dois), chunk_size), desc="Querying DB"):
            chunk = unique_dois[i : i + chunk_size]
            placeholders = ','.join('?' for _ in chunk)
            query = f"SELECT doi, abstract FROM papers WHERE doi IN ({placeholders})"
            
            try:
                cursor = conn.execute(query, chunk)
                for row in cursor:
                    doi_to_text[row[0]] = row[1]
            except sqlite3.Error as e:
                print(f"Database error: {e}")
                
    return doi_to_text

@hydra.main(config_path="../configs", config_name="mining", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Hard Negative Mining ===")
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. データの読み込み
    print(f"Loading training data: {cfg.data.input_train_file}")
    df_train = pd.read_csv(cfg.data.input_train_file)
    
    # 正例ペアのみを抽出
    df_positives = df_train[df_train['label'] == 1].copy().reset_index(drop=True)
    print(f"Loaded {len(df_positives):,} positive pairs.")

    # アンカー -> 正解(Positive) のマッピング (除外用)
    anchor_to_positives = df_positives.groupby('abstract_a')['abstract_b'].apply(set).to_dict()
    unique_anchors = list(anchor_to_positives.keys())
    print(f"Unique anchors to query: {len(unique_anchors):,}")

    # 2. リソースのロード
    tokenizer, model, index, id_to_doi = load_resources(cfg, device)

    # 3. 検索実行 (Hard Negative候補のIDを取得)
    anchor_to_candidates = {}
    batch_size = cfg.mining.batch_size
    
    print("Starting dense retrieval...")
    for i in tqdm(range(0, len(unique_anchors), batch_size), desc="Mining"):
        batch_anchors = unique_anchors[i : i + batch_size]
        
        # ベクトル化
        inputs = tokenizer(
            batch_anchors, padding=True, truncation=True, 
            max_length=cfg.mining.max_length, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # CLSトークン or Mean Pooling (SciBERTは通常CLSでOKだが、ここではPoolerOutputを使用)
            embeddings = outputs.pooler_output.cpu().numpy().astype(np.float32)
        
        # Faiss検索
        # 検索数は (NegRatio + 余裕分) ではなく、多めに取ってフィルタリングする
        distances, indices = index.search(embeddings, cfg.mining.search_top_k)
        
        # 結果を格納
        for j, anchor_text in enumerate(batch_anchors):
            result_ids = indices[j]
            candidate_dois = [id_to_doi[rid] for rid in result_ids if rid != -1 and rid in id_to_doi]
            anchor_to_candidates[anchor_text] = candidate_dois

    # 4. テキスト取得
    print("Fetching texts for candidates...")
    # 全候補DOIをリスト化
    all_candidate_dois = []
    for dois in anchor_to_candidates.values():
        all_candidate_dois.extend(dois)
    
    doi_to_text_map = get_abstracts_from_db(all_candidate_dois, cfg.data.db_path)
    print(f"Retrieved {len(doi_to_text_map):,} abstracts from DB.")

    # 5. データセット構築 (フィルタリング & 結合)
    final_rows = []
    neg_ratio = cfg.mining.neg_ratio
    min_len = cfg.mining.min_text_length
    
    print(f"Constructing final dataset (Ratio 1:{neg_ratio})...")
    
    for _, row in tqdm(df_positives.iterrows(), total=len(df_positives), desc="Building Pairs"):
        anchor = row['abstract_a']
        positive = row['abstract_b']
        data_paper_doi = row.get('data_paper_doi', None)
        
        # 正例を追加
        final_rows.append({
            'abstract_a': anchor,
            'abstract_b': positive,
            'label': 1,
            'data_paper_doi': data_paper_doi
        })
        
        # 負例を追加
        candidates = anchor_to_candidates.get(anchor, [])
        true_positives = anchor_to_positives.get(anchor, set())
        
        added_count = 0
        for neg_doi in candidates:
            if added_count >= neg_ratio:
                break
            
            raw_text = doi_to_text_map.get(neg_doi, "")
            if not raw_text:
                continue
                
            # クリーニング
            neg_text = clean_text(raw_text)
            
            # フィルタリング
            if len(neg_text) < min_len: continue
            if neg_text == anchor: continue      # 自分自身
            if neg_text in true_positives: continue # 正解(False Negative)回避
            
            final_rows.append({
                'abstract_a': anchor,
                'abstract_b': neg_text,
                'label': 0,
                'data_paper_doi': None
            })
            added_count += 1
    
    # 6. 保存
    df_final = pd.DataFrame(final_rows)
    print(f"Dataset created. Shape: {df_final.shape}")
    print("Label distribution:")
    print(df_final['label'].value_counts())
    
    output_dir = os.path.dirname(cfg.data.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_final.to_csv(cfg.data.output_file, index=False)
    print(f"Saved to: {cfg.data.output_file}")

if __name__ == "__main__":
    main()