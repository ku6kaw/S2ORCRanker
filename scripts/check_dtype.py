import numpy as np
import os
import sys

def check_npy_dtype(file_path):
    """NPYファイルのdtypeと形状を確認する"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    try:
        # mmapモードでヘッダーだけ読み込む（巨大なファイルでも一瞬で終わります）
        data = np.load(file_path, mmap_mode='r')
        print(f"File: {file_path}")
        print(f"  - Shape: {data.shape}")
        print(f"  - Dtype: {data.dtype}")
        
        if data.dtype == 'float16':
            print("  ✅ This is FLOAT16 (Half Precision)")
        elif data.dtype == 'float32':
            print("  ⚠️ This is FLOAT32 (Single Precision)")
        else:
            print(f"  ❓ Unknown dtype: {data.dtype}")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    print("-" * 40)

if __name__ == "__main__":
    # 引数でファイルパスが渡された場合
    if len(sys.argv) > 1:
        target_files = sys.argv[1:]
    else:
        # デフォルトで確認したい主要なファイルをリストアップ
        target_files = [
            "data/processed/embeddings/Pretrained_SPECTER2/corpus_embeddings.npy",
            "data/processed/embeddings/SPECTER2_Adapter/corpus_embeddings.npy",
            "data/processed/embeddings/SPECTER2_HardNeg/corpus_embeddings.npy",
            "data/processed/embeddings/SPECTER2_HardNeg_round2/corpus_embeddings.npy",
             "data/processed/embeddings/SPECTER2_Round2_Mining/corpus_embeddings.npy"
        ]

    print("=== Checking .npy file types ===\n")
    for path in target_files:
        check_npy_dtype(path)