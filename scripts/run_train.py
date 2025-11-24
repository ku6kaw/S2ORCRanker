# scripts/run_train.py

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer, TrainingArguments, AutoConfig, EarlyStoppingCallback
from torch.optim import AdamW
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())

from src.modeling.bi_encoder import SiameseBiEncoder
from src.modeling.cross_encoder import CrossEncoderMarginModel
from src.training.dataset import TextRankingDataset
from src.training.trainer import BiEncoderPairTrainer, MarginRankingTrainer, ContrastiveTrainer

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

def compute_metrics(eval_pred):
    """
    Bi-Encoder (BCE Loss) 用の評価指標計算
    """
    predictions, labels = eval_pred
    # predictionsはロジット(スコア)なので、0を閾値として0/1に変換
    # (Sigmoidを通すと 0.5 が閾値になるのと同義)
    preds = (predictions > 0).astype(int).reshape(-1)
    labels = labels.astype(int).reshape(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print(f"=== Starting Training: {cfg.model.type} ===")
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    dataset_handler = TextRankingDataset(cfg, tokenizer)
    tokenized_datasets = dataset_handler.load_and_prepare()
    
    print(f"Loading model: {cfg.model.name}")
    model_config = AutoConfig.from_pretrained(cfg.model.name)
    
    # --- ★ Trainerとモデルの選択ロジック (修正) ---
    compute_metrics_func = None

    if cfg.model.type == "cross_encoder":
        # Cross-Encoderの場合
        model_config.num_labels = 1
        model = CrossEncoderMarginModel.from_pretrained(cfg.model.name, config=model_config)
        trainer_cls = MarginRankingTrainer
        
    else: # bi_encoder
        # 設定からhead_typeを取得 (デフォルトはranknet)
        head_type = cfg.model.get("head_type", "ranknet")
        print(f"Bi-Encoder Head Type: {head_type}")
        
        # モデル初期化時に head_type を渡す
        model = SiameseBiEncoder.from_pretrained(
            cfg.model.name, 
            config=model_config, 
            head_type=head_type
        )
        
        if head_type == "none":
            # ヘッドなし -> 距離学習 (ContrastiveTrainer)
            print("Using ContrastiveTrainer (Distance-based)")
            trainer_cls = ContrastiveTrainer
            # Contrastiveの場合、BCE用のmetrics計算は適用できないためNoneのままにする
        else:
            # ヘッドあり -> 分類学習 (BiEncoderPairTrainer)
            print("Using BiEncoderPairTrainer (Classification Head)")
            trainer_cls = BiEncoderPairTrainer
            compute_metrics_func = compute_metrics # 以前定義した関数

    # 4. オプティマイザ作成
    base_lr = cfg.training.learning_rate
    head_lr = cfg.training.get("head_learning_rate", base_lr)
    
    optimizer_grouped_parameters = create_optimizer_grouped_parameters(
        model, base_lr, head_lr, cfg.training.weight_decay
    )
    optimizer = AdamW(optimizer_grouped_parameters)

    # 5. Arguments
    output_dir = cfg.training.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    args = TrainingArguments(
        output_dir=output_dir,
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
        report_to="wandb" if cfg.logging.use_wandb else "none",
        run_name=cfg.logging.run_name, 
        remove_unused_columns=False,
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    
    # ★ コールバックの準備
    callbacks = []
    if cfg.training.get("patience"):
        print(f"Early stopping enabled with patience: {cfg.training.patience}")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.training.patience))
        
    # 6. Trainer初期化
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        margin=cfg.training.margin,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics_func,
        callbacks=callbacks
    )
    
    print("Starting training...")
    trainer.train()
    
    best_model_path = os.path.join(output_dir, "best_model")
    print(f"Saving best model to {best_model_path}")
    trainer.save_model(best_model_path)
    
    if cfg.logging.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()