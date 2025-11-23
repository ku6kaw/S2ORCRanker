# scripts/convert_legacy_maps.py
import json
import os
import glob

LEGACY_DIR = "data/processed/legacy_embeddings"

def convert_map(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 辞書型の場合のみ変換 (リスト型ならスキップ)
    if isinstance(data, dict):
        print(f"Converting: {os.path.basename(file_path)} ...")
        # index (value) でソートして DOI (key) のリストを作成
        new_list = [k for k, v in sorted(data.items(), key=lambda item: item[1])]
        
        # 上書き保存（心配なら _converted.json に変えてください）
        with open(file_path, 'w') as f:
            json.dump(new_list, f)
        print(f"  -> Done. Count: {len(new_list)}")
    else:
        print(f"Skipping: {os.path.basename(file_path)} (Already list format)")

if __name__ == "__main__":
    # legacy_embeddings フォルダ内の全jsonを対象
    json_files = glob.glob(os.path.join(LEGACY_DIR, "*_doi_map.json"))
    if not json_files:
        print(f"No json files found in {LEGACY_DIR}")
    
    for p in json_files:
        convert_map(p)