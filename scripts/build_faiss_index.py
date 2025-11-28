# scripts/build_faiss_index.py
import numpy as np
import faiss
import sys
import os

# 設定: 前回の評価実験の出力パスに合わせてください
EMBEDDING_DIR = "data/processed/embeddings/SPECTER2_Adapter"
INPUT_NPY = os.path.join(EMBEDDING_DIR, "corpus_embeddings.npy")
OUTPUT_INDEX = os.path.join(EMBEDDING_DIR, "corpus.faiss")

# 次元の設定 (SPECTER2 base は 768)
DIMENSION = 768

def main():
    print(f"Loading embeddings from {INPUT_NPY}...")
    if not os.path.exists(INPUT_NPY):
        print(f"Error: {INPUT_NPY} not found.")
        return

    # メモリマップモードで開く
    vecs = np.memmap(INPUT_NPY, dtype='float32', mode='r')
    num_vecs = len(vecs) // DIMENSION
    print(f"Total vectors: {num_vecs:,}")
    
    # 形状を整える
    vecs = vecs.reshape(num_vecs, DIMENSION)

    # インデックスの構築 (Inner Product = Cosine Similarity)
    print("Building Faiss index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(DIMENSION)

    # メモリ効率のためバッチ処理で正規化して追加
    batch_size = 50000
    for i in range(0, num_vecs, batch_size):
        # バッチをメモリに読み込む
        batch = np.array(vecs[i : i + batch_size])
        # 正規化 (L2ノルム=1にする) -> 内積がコサイン類似度になる
        faiss.normalize_L2(batch)
        index.add(batch)
        if i % (batch_size * 5) == 0:
            print(f"  Added {i + len(batch):,} vectors...")

    print(f"Saving index to {OUTPUT_INDEX}...")
    faiss.write_index(index, OUTPUT_INDEX)
    print("Done.")

if __name__ == "__main__":
    main()