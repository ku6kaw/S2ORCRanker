import json
import os
from tqdm import tqdm

# --- 設定 ---
SUMMARY_FILE = "data/processed/evaluation_summary_50_final.json"
RETRIEVER_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/candidates_for_reranking_30k_HITABLE_WITH_TEXT.json"
RERANKER_FILE = "data/processed/embeddings/SPECTER2_HardNeg_round2/evaluation_results_reranked_SPECTER2_Cross_Chunking_HITABLE.json"
OUTPUT_FILE = "data/processed/evaluation_summary_50_ranked.json"

def normalize_doi(doi):
    """DOIの正規化（小文字化・空白削除）"""
    if not doi: return ""
    return doi.lower().strip()

def append_ranks_fixed():
    if not os.path.exists(SUMMARY_FILE):
        print("Summary file not found.")
        return

    print("Loading Retriever results...")
    retriever_rank_map = {}
    with open(RETRIEVER_FILE, 'r') as f:
        ret_data = json.load(f)
        for item in ret_data:
            q_doi = normalize_doi(item["query_doi"])
            # 候補DOIも正規化して辞書に登録
            # { candidate_doi_lower: rank }
            retriever_rank_map[q_doi] = {
                normalize_doi(c): i+1 
                for i, c in enumerate(item["retrieved_candidates"])
            }

    print("Loading Reranker results...")
    reranker_rank_map = {}
    if os.path.exists(RERANKER_FILE):
        with open(RERANKER_FILE, 'r') as f:
            rerank_data = json.load(f)
            details = rerank_data.get("details", [])
            for item in details:
                q_doi = normalize_doi(item["query_doi"])
                top_cands = item.get("reranked_top_10", [])
                reranker_rank_map[q_doi] = {
                    normalize_doi(c): i+1 
                    for i, c in enumerate(top_cands)
                }
    
    print(f"Loading Summary file: {SUMMARY_FILE}")
    with open(SUMMARY_FILE, 'r') as f:
        summary_data = json.load(f)

    print("Appending rank information (with normalization)...")
    
    match_count_ret = 0
    match_count_rer = 0
    total_candidates = 0
    
    for item in tqdm(summary_data):
        # クエリDOIを正規化してキーにする
        q_doi_raw = item["query_paper"]["doi"]
        q_doi_key = normalize_doi(q_doi_raw)
        
        # マップ取得
        ret_map = retriever_rank_map.get(q_doi_key, {})
        rerank_map = reranker_rank_map.get(q_doi_key, {})
        
        for candidate in item["candidates"]:
            total_candidates += 1
            c_doi_raw = candidate["doi"]
            c_doi_key = normalize_doi(c_doi_raw)
            
            # Retriever Rank
            r_rank = ret_map.get(c_doi_key)
            candidate["retriever_rank"] = r_rank
            if r_rank is not None:
                match_count_ret += 1
                
            # Reranker Rank
            rr_rank = rerank_map.get(c_doi_key)
            candidate["reranker_rank"] = rr_rank
            if rr_rank is not None:
                match_count_rer += 1

    print("-" * 50)
    print(f"Total Candidates Processed: {total_candidates}")
    print(f"Retriever Rank Matched: {match_count_ret} ({match_count_ret/total_candidates:.1%})")
    print(f"Reranker Rank Matched:  {match_count_rer} ({match_count_rer/total_candidates:.1%})")
    print("-" * 50)

    if match_count_ret == 0:
        print("⚠️ Warning: No Retriever ranks matched. Please check if Query DOIs match between files.")
        print(f"Sample Query DOI in Summary: {normalize_doi(summary_data[0]['query_paper']['doi'])}")
        print(f"Sample Query DOIs in Retriever Map: {list(retriever_rank_map.keys())[:3]}")

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # 確認用表示
    print("\n=== Sample Data Check ===")
    for cand in summary_data[0]["candidates"][:3]:
        print(f"DOI: {cand['doi']}")
        print(f"  -> Ret Rank: {cand.get('retriever_rank')}")
        print(f"  -> Rer Rank: {cand.get('reranker_rank')}")

if __name__ == "__main__":
    append_ranks_fixed()