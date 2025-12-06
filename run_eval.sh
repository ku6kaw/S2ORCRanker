python scripts/encode_corpus.py \
    logging.run_name="encode_Bi_SPECTER2_Adapter_Full" \
    model.path="models/checkpoints/bi_encoder/Bi_SPECTER2_MNRL_AdapterInit_Full/best_model" \
    model.base_name="allenai/specter2_base" \
    +model.adapter_name="allenai/specter2" \
    model.batch_size=256 \
    data.output_dir="data/processed/embeddings/SPECTER2_Adapter"

# SPECTER2 Adapter (AdapterInit_Full)
python scripts/evaluate.py \
    logging.run_name="eval_Bi_SPECTER2_Adapter_Full_100k" \
    model.path="models/checkpoints/bi_encoder/Bi_SPECTER2_MNRL_AdapterInit_Full/best_model" \
    model.base_name="allenai/specter2_base" \
    +model.adapter_name="allenai/specter2" \
    data.output_dir="data/processed/embeddings/SPECTER2_Adapter"

