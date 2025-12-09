import json
import csv
import random
import os
import pandas as pd
from tqdm import tqdm

# --- 設定 ---
# 1. 入力ファイル
# 先ほど作成した「テキスト付き」の候補ファイルを指定します
INPUT_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_30k_HITABLE_WITH_TEXT.json"

# 2. 出力ファイル
OUTPUT_CSV = "data/processed/training_dataset_hard_negatives.csv"

# 1クエリあたりに作成するHard Negativeの数
# 増やすとデータ数は増えますが、難易度の高い（上位の）ものだけに絞るなら小さめに
NEGATIVES_PER_QUERY = 5

def build_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("Please run scripts/hydrate_candidates.py first.")
        return

    print(f"Loading candidates from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    rows = []
    skipped_count = 0
    print("Building dataset...")

    for item in tqdm(data):
        query_text = item.get("query_text", "")
        ground_truth_dois = set(item.get("ground_truth_dois", []))
        
        # 候補リスト (Retrieverのスコア順に並んでいる前提)
        candidates = item.get("retrieved_candidates", [])
        
        # テキストマップ (DOI -> Abstract)
        candidate_texts = item.get("candidate_texts", {})
        
        if not query_text or not candidates:
            skipped_count += 1
            continue

        # --- 1. Positive (正解) の特定 ---
        # 候補リストの中に含まれる正解を探す
        # (HITABLEデータなので必ずあるはずだが、テキストがあるかも確認)
        valid_pos_doi = None
        for doi in candidates:
            if doi in ground_truth_dois:
                if candidate_texts.get(doi): # 本文があるかチェック
                    valid_pos_doi = doi
                    break # 最も上位の正解を1つ採用
        
        # もし候補内に正解がなくても、GTリストにはあるはずなのでそちらも確認
        if not valid_pos_doi:
            for doi in ground_truth_dois:
                if candidate_texts.get(doi):
                    valid_pos_doi = doi
                    break
        
        if not valid_pos_doi:
            # 正解の本文が見つからない場合はスキップ
            skipped_count += 1
            continue

        pos_text = candidate_texts[valid_pos_doi]

        # --- 2. Hard Negative (不正解) の特定 ---
        # 「候補リストの上位にある」かつ「正解ではない」ものを抽出
        hard_negatives = []
        for doi in candidates:
            if doi not in ground_truth_dois: # ★ここで確実に正解を除外
                text = candidate_texts.get(doi, "")
                # テキストがあり、かつ短すぎないもの
                if text and len(text) > 50:
                    hard_negatives.append(doi)
        
        if not hard_negatives:
            skipped_count += 1
            continue

        # 上位のHard Negativeからいくつか選ぶ
        # リストは既にスコア順なので、先頭に近いほど「難しい負例」
        # Top-50の中からランダムに選ぶ、あるいはTop-Nをそのまま使う
        num_negs = min(len(hard_negatives), NEGATIVES_PER_QUERY)
        selected_neg_dois = hard_negatives[:num_negs] # とにかく上位を使う（一番難しい）
        # あるいはランダム性を入れたい場合:
        # selected_neg_dois = random.sample(hard_negatives[:50], num_negs)

        # --- 3. データ行の追加 ---
        # Positive行とNegative行をセットで追加する
        # (dataset.pyはこれをTripletに変換して使う)
        
        for neg_doi in selected_neg_dois:
            neg_text = candidate_texts[neg_doi]
            
            # Label 1: Query + Positive
            rows.append({
                "abstract_a": query_text,
                "abstract_b": pos_text,
                "label": 1,
                "data_paper_doi": valid_pos_doi
            })
            
            # Label 0: Query + Negative
            rows.append({
                "abstract_a": query_text,
                "abstract_b": neg_text,
                "label": 0,
                "data_paper_doi": neg_doi
            })

    # --- 保存 ---
    if not rows:
        print("❌ No valid rows created. Check data integrity.")
        return

    df = pd.DataFrame(rows)
    print(f"\nCreated {len(df)} rows.")
    print(f"Skipped queries: {skipped_count}")
    
    # 念のためシャッフル
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Done.")
    
    # 確認表示
    print("\n--- Data Preview ---")
    print(df.head(2))
    print("\nChecking label distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    build_dataset()