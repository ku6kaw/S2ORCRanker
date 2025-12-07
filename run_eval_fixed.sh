# --- 共通設定 (10万件, FIXED版) ---
# 結果ファイルと候補ファイルに _FIXED をつけ、前回の結果と区別します
COMMON_ARGS="evaluation.gpu_search=false evaluation.queries_per_dataset=1"

# ==========================================
# 1. SPECTER2 (Fine-tuned, Random Neg, No Adapter)
# ==========================================
python scripts/evaluate.py $COMMON_ARGS \
    logging.run_name="eval_Bi_SPECTER2_No_Adapter_Random_Fixed" \
    model.path="models/checkpoints/bi_encoder/Bi_SPECTER2_MNRL_RandomNeg/best_model" \
    model.base_name="allenai/specter2_base" \
    data.output_dir="data/processed/embeddings/SPECTER2_MNRL"

# ==========================================
# 2. SPECTER2 Adapter Pretrained (Zero-shot)
# ==========================================
python scripts/evaluate.py $COMMON_ARGS \
    logging.run_name="eval_Bi_SPECTER2_Adapter_Pretrained_Fixed" \
    model.path="allenai/specter2_base" \
    model.base_name="allenai/specter2_base" \
    +model.adapter_name="allenai/specter2" \
    data.output_dir="data/processed/embeddings/Pretrained_SPECTER2"

# ==========================================
# 3. SPECTER2 Adapter Random Neg (Fine-tuned with Adapter)
# ==========================================
python scripts/evaluate.py $COMMON_ARGS \
    logging.run_name="eval_Bi_SPECTER2_Adapter_Full_Fixed" \
    model.path="models/checkpoints/bi_encoder/Bi_SPECTER2_MNRL_AdapterInit_Full/best_model" \
    model.base_name="allenai/specter2_base" \
    +model.adapter_name="allenai/specter2" \
    data.output_dir="data/processed/embeddings/SPECTER2_Adapter"

# ==========================================
# 4. SPECTER2 Adapter Hard Neg (Round 2)
# ==========================================
python scripts/evaluate.py $COMMON_ARGS \
    logging.run_name="eval_Bi_SPECTER2_Adapter_HardNeg_R2_Fixed" \
    model.path="models/checkpoints/bi_encoder/Bi_SPECTER2_MNRL_HardNeg_round2/best_model" \
    model.base_name="allenai/specter2_base" \
    +model.adapter_name="allenai/specter2" \
    data.output_dir="data/processed/embeddings/SPECTER2_HardNeg_round2"