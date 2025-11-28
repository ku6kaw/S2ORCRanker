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
from transformers import AutoTokenizer, AutoConfig
import adapters # Adapter対応

# srcへのパスを通す
sys.path.append(os.getcwd())
from src.utils.cleaning import clean_text
# モデルクラスもインポート（ロードロジック共通化のため）
from src.modeling.bi_encoder import SiameseBiEncoder

def load_resources(cfg, device):
    """モデルとFaissインデックスをロード (Adapter対応版)"""
    
    # --- 1. モデルのロード (encode_corpus.pyと同様のロジック) ---
    print(f"Loading model config from: {cfg.mining.model_name}")
    # ベースモデル名の決定 (設定になければ SPECTER2 base と仮定)
    base_model_name = "allenai/specter2_base"
    
    try:
        config = AutoConfig.from_pretrained(cfg.mining.model_name)
    except:
        config = AutoConfig.from_pretrained(base_model_name)

    # 構造の初期化
    model = SiameseBiEncoder.from_pretrained(base_model_name, config=config)
    
    # Adapter構造の初期化
    adapter_name = cfg.mining.get("adapter_name", None)
    if adapter_name:
        print(f"🔄 Initializing Adapter structure: {adapter_name}")
        adapters.init(model.bert)
        loaded_name = model.bert.load_adapter(adapter_name, source="hf", set_active=True)
        model.bert.set_active_adapters(loaded_name)
        print(f"   Adapter '{loaded_name}' structure initialized.")

    # 重みのロード
    print(f"📂 Loading trained state_dict from: {cfg.mining.model_name}")
    state_dict_path = os.path.join(cfg.mining.model_name, "pytorch_model.bin")
    if not os.path.exists(state_dict_path):
        state_dict_path = os.path.join(cfg.mining.model_name, "model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path)
    else:
        state_dict = torch.load(state_dict_path, map_location="cpu")
    
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"   Missing keys: {len(keys.missing_keys)}")
    
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # --- 2. Faissインデックスのロード ---
    print(f"Loading Faiss index: {cfg.data.faiss_index_file}")
    index = faiss.read_index(cfg.data.faiss_index_file)
    
    # --- 3. DOIマップのロード (リスト対応修正) ---
    print(f"Loading DOI map: {cfg.data.doi_map_file}")
    with open(cfg.data.doi_map_file, 'r') as f:
        doi_data = json.load(f)
    
    # リストの場合 (encode_corpus.pyの出力) と 辞書の場合 を分岐処理
    if isinstance(doi_data, list):
        # リストのインデックスがそのままIDになる
        print(f"   DOI map is a list of {len(doi_data)} items.")
        id_to_doi = {i: doi for i, doi in enumerate(doi_data)}
    else:
        # 辞書 {doi: id} の場合 -> 逆引き {id: doi} に変換
        print(f"   DOI map is a dict of {len(doi_data)} items.")
        id_to_doi = {v: k for k, v in doi_data.items()}
    
    return tokenizer, model, index, id_to_doi

def get_abstracts_from_db(dois, db_path):
    """SQLiteからDOIに対応するアブストラクトを一括取得"""
    doi_to_text = {}
    
    with sqlite3.connect(db_path) as conn:
        chunk_size = 900 
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
    print("=== Starting Hard Negative Mining (Fixed) ===")
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

    # 3. 検索実行
    anchor_to_candidates = {}
    batch_size = cfg.mining.batch_size
    
    print("Starting dense retrieval...")
    for i in tqdm(range(0, len(unique_anchors), batch_size), desc="Mining"):
        batch_anchors = unique_anchors[i : i + batch_size]
        
        inputs = tokenizer(
            batch_anchors, padding=True, truncation=True, 
            max_length=cfg.mining.max_length, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # SiameseBiEncoderを使うため、pooler_outputではなくoutput_vectors=Trueの結果を使う
            outputs = model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                output_vectors=True
            )
            embeddings = outputs.logits.float().cpu().numpy()
        
        # Faiss検索
        distances, indices = index.search(embeddings, cfg.mining.search_top_k)
        
        for j, anchor_text in enumerate(batch_anchors):
            result_ids = indices[j]
            # IDが有効範囲内かつマップに存在するか確認
            candidate_dois = [id_to_doi[rid] for rid in result_ids if rid != -1 and rid in id_to_doi]
            anchor_to_candidates[anchor_text] = candidate_dois

    # 4. テキスト取得
    print("Fetching texts for candidates...")
    all_candidate_dois = []
    for dois in anchor_to_candidates.values():
        all_candidate_dois.extend(dois)
    
    doi_to_text_map = get_abstracts_from_db(all_candidate_dois, cfg.data.db_path)
    print(f"Retrieved {len(doi_to_text_map):,} abstracts from DB.")

    # 5. データセット構築
    final_rows = []
    neg_ratio = cfg.mining.neg_ratio
    min_len = cfg.mining.min_text_length
    
    print(f"Constructing final dataset (Ratio 1:{neg_ratio})...")
    
    for _, row in tqdm(df_positives.iterrows(), total=len(df_positives), desc="Building Pairs"):
        anchor = row['abstract_a']
        positive = row['abstract_b']
        data_paper_doi = row.get('data_paper_doi', None)
        
        # 正例
        final_rows.append({
            'abstract_a': anchor,
            'abstract_b': positive,
            'label': 1,
            'data_paper_doi': data_paper_doi
        })
        
        # 負例
        candidates = anchor_to_candidates.get(anchor, [])
        true_positives = anchor_to_positives.get(anchor, set())
        
        added_count = 0
        for neg_doi in candidates:
            if added_count >= neg_ratio:
                break
            
            raw_text = doi_to_text_map.get(neg_doi, "")
            if not raw_text:
                continue
                
            neg_text = clean_text(raw_text)
            
            # フィルタリング
            if len(neg_text) < min_len: continue
            if neg_text == anchor: continue
            if neg_text in true_positives: continue # False Negative回避
            
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