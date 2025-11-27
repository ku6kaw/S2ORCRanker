# scripts/encode_corpus.py

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import sqlite3
import numpy as np
import json
import math
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
import adapters

sys.path.append(os.getcwd())
from src.modeling.bi_encoder import SiameseBiEncoder
from src.utils.cleaning import clean_text

def get_db_stats(db_path):
    """DBの行数と最大rowidを取得"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(rowid), COUNT(*) FROM papers")
        max_id, count = cursor.fetchone()
    return max_id, count

class SQLiteDataset(IterableDataset):
    def __init__(self, db_path, max_rowid, debug=False):
        self.db_path = db_path
        self.max_rowid = max_rowid
        self.debug = debug

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            start = 0
            end = self.max_rowid
        else:
            per_worker = int(math.ceil((self.max_rowid + 1) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self.max_rowid + 1)

        if self.debug:
            end = min(start + 1000, end)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = f"""
                SELECT doi, abstract 
                FROM papers 
                WHERE rowid >= ? AND rowid < ? AND abstract IS NOT NULL AND length(abstract) > 10
            """
            cursor.execute(query, (start, end))
            
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                
                for doi, abstract in rows:
                    cleaned_text = clean_text(abstract)
                    if cleaned_text:
                        yield doi, cleaned_text

class CollateFn:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        dois = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return dois, inputs

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Optimized Corpus Encoding (Bus Error Fix & Correct Adapter Loading) ===")
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        print("🚀 FP16 (Mixed Precision) Enabled.")

    if not os.path.exists(cfg.data.output_dir):
        os.makedirs(cfg.data.output_dir)
    
    embeddings_path = os.path.join(cfg.data.output_dir, cfg.data.embeddings_file)
    doi_map_path = os.path.join(cfg.data.output_dir, cfg.data.doi_map_file)

    # --- 1. モデルロード (修正版) ---
    print(f"Loading model config from: {cfg.model.path}")
    config = AutoConfig.from_pretrained(cfg.model.path)
    
    # 【重要】まずベースモデル名から構造だけ初期化する
    # (checkpointから直接呼ぶと、Adapter層がない状態で重みをロードしてしまい、警告が出るため)
    model = SiameseBiEncoder.from_pretrained(cfg.model.base_name, config=config)
    
    # アダプター構造の注入
    adapter_name = cfg.model.get("adapter_name", None)
    if adapter_name:
        print(f"🔄 Initializing Adapter structure: {adapter_name}")
        adapters.init(model.bert)
        # Hubから構造定義だけ読み込んでレイヤーを追加
        loaded_name = model.bert.load_adapter(adapter_name, source="hf", set_active=True)
        model.bert.set_active_adapters(loaded_name)
        print(f"   Adapter '{loaded_name}' structure initialized.")

    # 【重要】最後に学習済みの重み(Full State Dict)をロードして上書きする
    print(f"📂 Loading trained state_dict from: {cfg.model.path}")
    state_dict_path = os.path.join(cfg.model.path, "pytorch_model.bin")
    if not os.path.exists(state_dict_path):
        # safetensorsの場合のフォールバック
        state_dict_path = os.path.join(cfg.model.path, "model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path)
    else:
        state_dict = torch.load(state_dict_path, map_location="cpu")
    
    # モデルにロード (strict=Falseで不整合によるクラッシュを防ぎつつ、必要なキーをロード)
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"   Missing keys: {len(keys.missing_keys)}")
    print(f"   Unexpected keys: {len(keys.unexpected_keys)}")
    if len(keys.missing_keys) > 0:
        # classifier_head などは推論に不要なのでMissingでもOKな場合が多いが、念のため表示
        print(f"   ⚠️ Warning: Some keys were missing (e.g. {keys.missing_keys[:3]}...)")
    
    model.to(device)
    model.eval()
    
    # 【重要】バスエラー回避のためコンパイルは無効化
    # if hasattr(torch, "compile"): ... 

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    # --- 2. DB統計取得とDataset準備 ---
    print("Analyzing Database...")
    max_rowid, total_count = get_db_stats(cfg.data.db_path)
    print(f"Max RowID: {max_rowid}, Total Count: {total_count:,}")
    
    dataset = SQLiteDataset(cfg.data.db_path, max_rowid, debug=cfg.get("debug", False))
    
    batch_size = cfg.model.batch_size
    
    # 【重要】バスエラー対策のため num_workers=0 (メインプロセスで読み込み)
    num_workers = 0 
    
    collate_fn = CollateFn(tokenizer, cfg.model.max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        # prefetch_factor=2 # num_workers=0 のときはprefetch_factorは使えないため削除
    )

    # --- 3. Memmap準備 ---
    output_shape = (total_count, config.hidden_size) if not cfg.get("debug", False) else (1000, config.hidden_size)
    
    print(f"Creating memmap file at {embeddings_path}...")
    if not cfg.get("debug", False):
        required_space_gb = (total_count * config.hidden_size * 4) / (1024**3)
        print(f"   Required disk space: approx {required_space_gb:.2f} GB")

    all_embeddings = np.memmap(
        embeddings_path, 
        dtype='float32', 
        mode='w+', 
        shape=output_shape
    )

    # --- 4. 推論ループ ---
    doi_list = []
    current_idx = 0
    
    print("Starting inference...")
    total_batches = (output_shape[0] // batch_size) + 1
    
    with torch.no_grad():
        for batch_dois, batch_inputs in tqdm(dataloader, total=total_batches, desc="Encoding"):
            input_ids = batch_inputs['input_ids'].to(device, non_blocking=True)
            attention_mask = batch_inputs['attention_mask'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=use_fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_vectors=True
                )
                embeddings = outputs.logits.float().cpu().numpy()
            
            n_samples = len(embeddings)
            
            if current_idx + n_samples > output_shape[0]:
                 break
            
            all_embeddings[current_idx : current_idx + n_samples] = embeddings
            doi_list.extend(batch_dois)
            current_idx += n_samples

    print(f"Encoding complete. Valid vectors: {current_idx:,}")

    print(f"Saving DOI map to {doi_map_path}...")
    with open(doi_map_path, 'w') as f:
        json.dump(doi_list, f)
        
    all_embeddings.flush()
    print("Done.")

if __name__ == "__main__":
    main()