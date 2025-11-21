# scripts/encode_corpus.py

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

# srcへのパスを通す
sys.path.append(os.getcwd())
from src.modeling.bi_encoder import SiameseBiEncoder
from src.utils.cleaning import clean_text

def get_total_count(db_path):
    """DB内の論文総数を取得"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(doi) FROM papers")
        count = cursor.fetchone()[0]
    return count

def fetch_data_generator(db_path, batch_size):
    """
    DBからデータをバッチサイズごとにジェネレートする。
    メモリ効率のため、一度に全て読み込まずカーソルを使用。
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT doi, abstract FROM papers")
        
        batch_dois = []
        batch_texts = []
        
        for row in cursor:
            doi, abstract = row
            
            # テキストがない、あるいは短すぎる場合はスキップ（必要に応じて調整）
            if not abstract or len(abstract) < 10:
                continue
                
            batch_dois.append(doi)
            # クリーニング処理（学習時と同じ前処理を適用）
            batch_texts.append(clean_text(abstract))
            
            if len(batch_dois) >= batch_size:
                yield batch_dois, batch_texts
                batch_dois = []
                batch_texts = []
        
        # 残りのデータ
        if batch_dois:
            yield batch_dois, batch_texts

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print("=== Starting Corpus Encoding ===")
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 出力ディレクトリの準備
    if not os.path.exists(cfg.data.output_dir):
        os.makedirs(cfg.data.output_dir)
    
    embeddings_path = os.path.join(cfg.data.output_dir, cfg.data.embeddings_file)
    doi_map_path = os.path.join(cfg.data.output_dir, cfg.data.doi_map_file)

    # 2. モデルとトークナイザのロード
    print(f"Loading model from: {cfg.model.path}")
    # Configをロードしてモデル構造を特定
    config = AutoConfig.from_pretrained(cfg.model.path)
    
    # ★重要: SiameseBiEncoderとしてロード
    # head_typeは推論時はあまり関係ないが、学習時の設定を引き継ぐ
    model = SiameseBiEncoder.from_pretrained(cfg.model.path, config=config)
    model.to(device)
    model.eval()
    
    # トークナイザ（学習済みモデルのディレクトリ、なければベース名から）
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    except:
        print(f"Tokenizer not found in model path, loading from base: {cfg.model.base_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    # 3. 保存用 memmap の準備
    print(f"Counting total papers in {cfg.data.db_path}...")
    total_papers = get_total_count(cfg.data.db_path)
    print(f"Total papers: {total_papers:,}")
    
    # SciBERTの次元数 (768)
    hidden_size = config.hidden_size 
    
    # memmapファイルを作成 (書き込みモード)
    # 形状: (総論文数, 768)
    # ※ データクリーニングでスキップされる分があるため、少し大きめに確保するか、
    #    あるいは一旦リストに貯めてから最後に保存する手もあるが、
    #    1100万件だとメモリが死ぬので、memmapで確保して、実際の件数に合わせてtruncateするのが安全。
    #    今回は簡易的に「最大数」で確保し、最後にメタデータで有効件数を管理する。
    
    print(f"Creating memmap file at {embeddings_path}...")
    all_embeddings = np.memmap(
        embeddings_path, 
        dtype='float32', 
        mode='w+', 
        shape=(total_papers, hidden_size)
    )

    # 4. 推論実行ループ
    batch_size = cfg.model.batch_size
    
    if cfg.get("debug", False):
        print("\n" + "="*50)
        print(" ⚠️  DEBUG MODE ENABLED: Processing only 10 batches!")
        print("="*50 + "\n")
        # ジェネレータを作り直す必要はないが、ループ回数を制限する
        total_papers = batch_size * 10
        
    data_gen = fetch_data_generator(cfg.data.db_path, batch_size)
    
    doi_list = [] # インデックス -> DOI のマッピング用
    current_idx = 0
    
    # tqdmのトータルは概算
    with torch.no_grad():
        # tqdmのtotalを調整
        pbar = tqdm(data_gen, total=(total_papers // batch_size) + 1, desc="Encoding")
        
        # ★修正: enumerateでインデックス(i)とデータ((dois, texts))を同時に取得し、1つのループで処理する
        for i, (batch_dois, batch_texts) in enumerate(pbar):
            
            # ▼▼▼ 1. デバッグモード時の脱出判定 ▼▼▼
            if cfg.get("debug", False) and i >= 10:
                print("Debug limit reached. Stopping.")
                break
            # ▲▲▲ -------------------------------- ▲▲▲

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
            
            # 推論 (output_vectors=True でベクトルのみ取得)
            # SiameseBiEncoder.forward の仕様に合わせる
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_vectors=True # ★ここがポイント
            )
            
            # ベクトル取得 (Batch, 768)
            embeddings = outputs.logits.cpu().numpy()
            
            # memmapに書き込み
            n_samples = len(batch_dois)
            all_embeddings[current_idx : current_idx + n_samples] = embeddings
            
            # DOIリスト更新
            doi_list.extend(batch_dois)
            current_idx += n_samples

    print(f"Encoding complete. Valid vectors: {current_idx:,}")

    # 5. DOIマップの保存
    # インデックス(行番号) -> DOI の辞書
    # リストのインデックスがそのままmemmapの行番号に対応
    print(f"Saving DOI map to {doi_map_path}...")
    
    # JSONシリアライズ可能な形式に変換（念のため）
    with open(doi_map_path, 'w') as f:
        json.dump(doi_list, f)

    # 6. Memmapのトリミング（もしスキップされたデータがあってサイズが合わない場合）
    if current_idx < total_papers:
        print(f"Trimming memmap from {total_papers} to {current_idx}...")
        # 新しいサイズで再度memmapを開き直すことはできないので、
        # 別途メタデータとして「有効行数」を記録するか、
        # 評価スクリプト側で doi_list の長さを信じるように設計する。
        # ここでは「doi_listの長さ ＝ 有効なベクトル数」とする運用にします。
        pass
        
    # ディスクへのフラッシュ
    all_embeddings.flush()
    print("Done.")

if __name__ == "__main__":
    main()