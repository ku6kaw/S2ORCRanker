import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import math

# --- 設定 ---
INPUT_FILE = "data/processed/evaluation_dataset_metadata_full.jsonl"
OUTPUT_DIR = "reports/figures/sorted_networks"

# 生成する画像の枚数
NUM_IMAGES = 10
# 1枚あたりのクエリ数
QUERIES_PER_IMAGE = 5

def visualize_sorted_batches():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return
    
    # 出力ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data from {INPUT_FILE}...")
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # --- 1. ソート処理 ---
    # 正解データ (ground_truth_details) の数で降順にソート
    print("Sorting queries by ground truth count (descending)...")
    data.sort(key=lambda x: len(x.get("ground_truth_details", [])), reverse=True)
    
    # Top 50 (10枚 * 5個) を抽出
    total_target = NUM_IMAGES * QUERIES_PER_IMAGE
    target_data = data[:total_target]
    
    print(f"Processing top {len(target_data)} queries...")

    # --- 2. 画像生成ループ ---
    for img_idx in range(NUM_IMAGES):
        # バッチの範囲を決定
        start_idx = img_idx * QUERIES_PER_IMAGE
        end_idx = start_idx + QUERIES_PER_IMAGE
        batch = target_data[start_idx:end_idx]
        
        if not batch:
            break
            
        # 1行5列のキャンバスを作成 (横長)
        fig, axes = plt.subplots(1, QUERIES_PER_IMAGE, figsize=(25, 5))
        if QUERIES_PER_IMAGE == 1: axes = [axes] # 1個の場合の対応

        print(f"Generating image {img_idx+1}/{NUM_IMAGES} (Rank {start_idx+1} - {end_idx})...")

        for i, item in enumerate(batch):
            ax = axes[i]
            
            # データの取得
            rank = start_idx + i + 1
            query_doi = item.get("query_doi", "Q")
            q_title_full = item.get("query_title") or query_doi
            q_title = (q_title_full[:30] + '..') if len(q_title_full) > 30 else q_title_full
            
            gt_details = item.get("ground_truth_details", [])
            
            # --- グラフ構築 ---
            G = nx.Graph()
            
            # 中心ノード (クエリ)
            G.add_node("Query", color='#1f77b4', size=1500, label="Q") # 青
            
            overlap_count = 0
            
            # 周辺ノード (正解)
            for j, gt in enumerate(gt_details):
                gt_id = f"G{j+1}"
                
                # 著者被り判定
                q_auth = set(item.get("query_authors", []))
                g_auth = set(gt.get("authors", []))
                is_overlap = not q_auth.isdisjoint(g_auth)
                
                if is_overlap:
                    overlap_count += 1
                
                # 色分け: 被りあり=赤, なし=緑
                node_color = '#d62728' if is_overlap else '#2ca02c'
                
                G.add_node(gt_id, color=node_color, size=300, label="")
                G.add_edge("Query", gt_id, color=node_color)

            # --- レイアウトと描画 ---
            # 星型になるようにバネモデルを使用
            pos = nx.spring_layout(G, k=0.5, seed=42)
            
            # 色とサイズの抽出
            node_colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
            node_sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            
            nx.draw(G, pos, ax=ax, 
                    node_color=node_colors, 
                    node_size=node_sizes, 
                    edge_color=edge_colors,
                    width=1.5,
                    with_labels=False)
            
            # 中心ラベル "Q"
            nx.draw_networkx_labels(G, pos, ax=ax, labels={"Query": "Q"}, font_color="white", font_size=12, font_weight="bold")

            # タイトル設定
            title_color = "red" if overlap_count > 0 else "black"
            ax.set_title(f"Rank #{rank}\nGT Count: {len(gt_details)}\n(Overlap: {overlap_count})", 
                         fontsize=11, color=title_color, fontweight="bold")
            
            # クエリタイトルを下に表示
            ax.text(0.5, -0.1, q_title, transform=ax.transAxes, 
                    ha='center', fontsize=9, wrap=True)

            # 枠線を消す
            ax.axis('off')

        # 保存
        filename = f"batch_{img_idx+1:02d}_rank_{start_idx+1}-{end_idx}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100)
        plt.close()

    print(f"✅ All {NUM_IMAGES} images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_sorted_batches()