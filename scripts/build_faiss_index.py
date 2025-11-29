# scripts/build_faiss_index.py
import numpy as np
import faiss
import sys
import os

def main():
    if len(sys.argv) > 1:
        EMBEDDING_DIR = sys.argv[1]
    else:
        print("Usage: python scripts/build_faiss_index.py <embedding_dir>")
        return

    INPUT_NPY = os.path.join(EMBEDDING_DIR, "corpus_embeddings.npy")
    OUTPUT_INDEX = os.path.join(EMBEDDING_DIR, "corpus.faiss")
    DIMENSION = 768

    print(f"Loading embeddings from {INPUT_NPY}...")
    if not os.path.exists(INPUT_NPY):
        print(f"Error: {INPUT_NPY} not found.")
        return

    # float16で読み込み
    vecs = np.memmap(INPUT_NPY, dtype='float16', mode='r')
    num_vecs = len(vecs) // DIMENSION
    print(f"Total vectors: {num_vecs:,}")
    
    vecs = vecs.reshape(num_vecs, DIMENSION)

    print("Building Faiss index (ScalarQuantizer QT_fp16)...")
    # ▼▼▼ 修正: float16のまま保存できるインデックスを使用 ▼▼▼
    # IndexFlatIP は float32 になるため、ScalarQuantizerで float16 を指定
    index = faiss.IndexScalarQuantizer(DIMENSION, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    
    # ScalarQuantizerは学習が必要だが、QT_fp16は統計情報を取らないためtrain不要な場合が多い
    # 念のため少量のデータでtrainを呼んでおく（QT_fp16なら実質何もしない）
    train_size = min(50000, num_vecs)
    train_data = np.array(vecs[:train_size]).astype('float32')
    index.train(train_data)

    # データの追加
    batch_size = 50000
    for i in range(0, num_vecs, batch_size):
        # Indexへの追加時は float32 のnumpy配列として渡す必要がある
        # (内部でfp16に圧縮して保存される)
        batch = np.array(vecs[i : i + batch_size]).astype('float32')
        
        faiss.normalize_L2(batch)
        index.add(batch)
        
        if i % (batch_size * 5) == 0:
            print(f"  Added {i + len(batch):,} vectors...")

    print(f"Saving index to {OUTPUT_INDEX}...")
    faiss.write_index(index, OUTPUT_INDEX)
    print("Done.")

if __name__ == "__main__":
    main()