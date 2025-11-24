# scripts/encode_corpus.py (メモリ安全・高速化版)

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import sqlite3
import numpy as np
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig

sys.path.append(os.getcwd())
from src.modeling.bi_encoder import SiameseBiEncoder
from src.utils.cleaning import clean_text

def get_total_count(db_path):
    """DB内の論文総数を取得"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(doi) FROM papers")
        try:
            count = cursor.fetchone()[0]
        except TypeError:
            count = 0
    return count

def fetch_data_generator(db_path, batch_size, debug=False):
    """
    DBからデータをバッチサイズごとにジェネレートする（メモリ節約型）。
    """
    limit_clause = " LIMIT 5000" if debug else ""
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # カーソルで少しずつ読み込む
        cursor.execute(f"SELECT doi, abstract FROM papers WHERE abstract IS NOT NULL AND length(abstract) > 10{limit_clause}")
        
        batch_dois = []
        batch_texts = []
        
        while True:
            # fetchmanyでバッチサイズ分だけ取得（メモリ効率良）
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
                
            for row in rows:
                doi, abstract = row
                cleaned = clean_text(abstract)
                if cleaned:
                    batch_dois.append(doi)
                    batch_texts.append(cleaned)
            
            # クリーニングで減った分、batch_sizeに満たない場合があるがそのまま送る
            if batch_dois:
                yield batch_dois, batch_texts
                batch_dois = []
                batch_texts = []

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Memory-Safe Corpus Encoding ===")
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # FP16 (Mixed Precision) の有効化
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        print("🚀 FP16 (Mixed Precision) Enabled for speedup.")

    if not os.path.exists(cfg.data.output_dir):
        os.makedirs(cfg.data.output_dir)
    
    embeddings_path = os.path.join(cfg.data.output_dir, cfg.data.embeddings_file)
    doi_map_path = os.path.join(cfg.data.output_dir, cfg.data.doi_map_file)

    # 1. モデルロード
    print(f"Loading model from: {cfg.model.path}")
    config = AutoConfig.from_pretrained(cfg.model.path)
    model = SiameseBiEncoder.from_pretrained(cfg.model.path, config=config)
    model.to(device)
    model.eval()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    # 2. 総数カウント & Memmap準備
    total_papers = get_total_count(cfg.data.db_path)
    if cfg.get("debug", False):
        total_papers = 5000
        
    print(f"Total papers (estimate): {total_papers:,}")
    hidden_size = config.hidden_size 
    
    print(f"Creating memmap file at {embeddings_path}...")
    all_embeddings = np.memmap(
        embeddings_path, 
        dtype='float32', 
        mode='w+', 
        shape=(total_papers, hidden_size)
    )

    # 3. 推論ループ
    batch_size = cfg.model.batch_size
    # データジェネレータを使用（メモリ安全）
    data_gen = fetch_data_generator(cfg.data.db_path, batch_size, debug=cfg.get("debug", False))
    
    doi_list = [] 
    current_idx = 0
    
    # 推論
    with torch.no_grad():
        # tqdmのトータルは概算
        pbar = tqdm(data_gen, total=(total_papers // batch_size) + 1, desc="Encoding")
        
        for batch_dois, batch_texts in pbar:
            if not batch_texts:
                continue

            # トークナイズ
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=cfg.model.max_length, 
                return_tensors="pt"
            ).to(device)
            
            # FP16 推論 (高速化の肝)
            with torch.amp.autocast('cuda', enabled=use_fp16):
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_vectors=True
                )
                # float32に戻してCPUへ
                embeddings = outputs.logits.float().cpu().numpy()
            
            # 書き込み
            n_samples = len(embeddings)
            
            # memmapのサイズ超過チェック（推定より多かった場合）
            if current_idx + n_samples > total_papers:
                # 必要ならリサイズ等の処理も可能だが、今回は切り捨てるか、
                # または余裕を持って確保しておく設計にする。
                # 簡易的にここで打ち切る（総数はget_total_countで正確なはずだが、debug時は注意）
                if cfg.get("debug", False):
                    break
                
                # 実運用でサイズ不足が起きた場合の緊急回避（はみ出した分は無視）
                n_samples = total_papers - current_idx
                embeddings = embeddings[:n_samples]
                batch_dois = batch_dois[:n_samples]

            all_embeddings[current_idx : current_idx + n_samples] = embeddings
            doi_list.extend(batch_dois)
            current_idx += n_samples
            
            if current_idx >= total_papers:
                break

    print(f"Encoding complete. Valid vectors: {current_idx:,}")

    # 4. DOIマップ保存
    print(f"Saving DOI map to {doi_map_path}...")
    with open(doi_map_path, 'w') as f:
        json.dump(doi_list, f)

    # ディスクへのフラッシュ
    all_embeddings.flush()
    print("Done.")

if __name__ == "__main__":
    main()