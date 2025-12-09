import pandas as pd
import textwrap

# 検証対象のファイル
csv_file = "data/processed/training_dataset_hard_negatives_round2.csv"

def inspect_labels():
    print(f"Loading {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("File not found.")
        return

    # Label=0 (負例として学習されるもの) を抽出
    negatives = df[df['label'] == 0]
    
    if len(negatives) == 0:
        print("No label=0 found. Dataset might be formatted differently.")
        return

    print(f"\nTotal rows with Label=0 (Negative): {len(negatives)}")
    print("Checking random samples from Label=0...\n")
    print("="*80)

    # ランダムに5件表示して確認
    samples = negatives.sample(n=min(5, len(negatives)), random_state=42)

    for idx, row in samples.iterrows():
        print(f"Row Index: {idx}")
        print("-" * 20)
        
        # クエリ (検索する人の入力)
        query = row.get('abstract_a', '')[:200]
        print(f"[Query (User Input)]: \n{query}...")
        
        print("-" * 20)
        
        # 候補 (負例として扱われている論文)
        candidate = row.get('abstract_b', '')
        
        # データセット論文っぽいキーワードが含まれているかチェック
        keywords = ["dataset", "database", "corpus", "benchmark", "repository", "new data"]
        warning = ""
        if any(k in candidate.lower() for k in keywords):
            warning = "⚠️  Contains 'dataset' keywords!"

        print(f"[Label=0 (TREATED AS NEGATIVE/INCORRECT)]: {warning}")
        print(textwrap.fill(candidate[:500], width=80)) # 長すぎるので冒頭500文字
        print("..." if len(candidate) > 500 else "")
        print("\n" + "="*80 + "\n")

    print("【判定方法】")
    print("もし [Label=0] の内容が「本来検索されるべき正解（データセット論文）」であれば、")
    print("データセットのラベルは逆（間違い）です。")

if __name__ == "__main__":
    inspect_labels()