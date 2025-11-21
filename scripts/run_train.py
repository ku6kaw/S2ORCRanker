# scripts/run_train.py

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer, TrainingArguments, AutoConfig
from torch.optim import AdamW

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())

from src.modeling.bi_encoder import SiameseBiEncoder
from src.modeling.cross_encoder import CrossEncoderMarginModel
from src.training.dataset import TextRankingDataset
from src.training.trainer import BiEncoderPairTrainer, MarginRankingTrainer

def create_optimizer_grouped_parameters(model, base_lr, head_lr, weight_decay):
    """
    モデルのパラメータを4つのグループに分け、学習率とWeight Decayを適用する。
    1. Head (Decayあり)
    2. Head (Decayなし)
    3. Base (Decayあり)
    4. Base (Decayなし)
    """
    # Weight Decayを適用しないパラメータ名
    no_decay = ["bias", "LayerNorm.weight"]
    
    # 分類ヘッドとみなすパラメータ名のキーワード
    # Bi-Encoder: "classifier_head"
    # Cross-Encoder (Longformer): "scorer.classifier" 等
    head_keywords = ["classifier", "head", "score"]

    optimizer_grouped_parameters = []
    
    # 全パラメータを走査して振り分け
    head_params_decay = []
    head_params_no_decay = []
    base_params_decay = []
    base_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # ヘッドかどうか判定
        is_head = any(k in name for k in head_keywords)
        # Decayなしかどうか判定
        is_no_decay = any(nd in name for nd in no_decay)

        if is_head:
            if is_no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)
        else:
            if is_no_decay:
                base_params_no_decay.append(param)
            else:
                base_params_decay.append(param)

    # グループ定義の作成
    if head_params_decay:
        optimizer_grouped_parameters.append({
            "params": head_params_decay,
            "weight_decay": weight_decay,
            "lr": head_lr,
            "name": "head_decay"
        })
    if head_params_no_decay:
        optimizer_grouped_parameters.append({
            "params": head_params_no_decay,
            "weight_decay": 0.0,
            "lr": head_lr,
            "name": "head_no_decay"
        })
    if base_params_decay:
        optimizer_grouped_parameters.append({
            "params": base_params_decay,
            "weight_decay": weight_decay,
            "lr": base_lr,
            "name": "base_decay"
        })
    if base_params_no_decay:
        optimizer_grouped_parameters.append({
            "params": base_params_no_decay,
            "weight_decay": 0.0,
            "lr": base_lr,
            "name": "base_no_decay"
        })

    return optimizer_grouped_parameters

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print(f"=== Starting Training: {cfg.model.type} ===")
    print(OmegaConf.to_yaml(cfg))
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    dataset_handler = TextRankingDataset(cfg, tokenizer)
    tokenized_datasets = dataset_handler.load_and_prepare()
    
    print(f"Loading model: {cfg.model.name}")
    model_config = AutoConfig.from_pretrained(cfg.model.name)
    
    # ▼▼▼ 変更点: Trainer選択ロジック ▼▼▼
    if cfg.model.type == "cross_encoder":
        model_config.num_labels = 1
        model = CrossEncoderMarginModel.from_pretrained(cfg.model.name, config=model_config)
        trainer_cls = MarginRankingTrainer
    else:
        # Bi-Encoder (分類ヘッドあり)
        # config.yamlの head_type はモデル内部で使用されるが、
        # Trainerは常にスコア(BCE Loss)を扱うものを使用する
        model = SiameseBiEncoder.from_pretrained(cfg.model.name, config=model_config)
        trainer_cls = BiEncoderPairTrainer 

    # オプティマイザ作成
    base_lr = cfg.training.learning_rate
    head_lr = cfg.training.get("head_learning_rate", base_lr)
    print(f"Optimizer Setup: Base LR={base_lr}, Head LR={head_lr}")
    
    optimizer_grouped_parameters = create_optimizer_grouped_parameters(
        model, base_lr, head_lr, cfg.training.weight_decay
    )
    optimizer = AdamW(optimizer_grouped_parameters)

    args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.epochs,
        learning_rate=base_lr, 
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_strategy="steps",
        logging_steps=cfg.training.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        margin=cfg.training.margin,
        optimizers=(optimizer, None)
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {cfg.training.output_dir}/best_model")
    trainer.save_model(os.path.join(cfg.training.output_dir, "best_model"))

if __name__ == "__main__":
    main()