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
import adapters

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())

from src.modeling.bi_encoder import SiameseBiEncoder
from src.modeling.cross_encoder import CrossEncoderMarginModel
from src.training.dataset import TextRankingDataset, CrossEncoderTripletCollator # ★修正
from src.training.trainer import BiEncoderPairTrainer, MarginRankingTrainer, ContrastiveTrainer, MultipleNegativesRankingTrainer
from src.utils.optimization import create_optimizer_grouped_parameters

def compute_metrics(eval_pred):
    """
    Bi-Encoder (BCE Loss / Contrastive) 用の評価指標計算
    Cross-Encoder (MarginRankingLoss) では通常metrics計算は難しいのでスキップされることが多いが、
    便宜上エラーにならないようにしておく。
    """
    predictions, labels = eval_pred
    
    # predictionsがタプルの場合 (Contrastive, MarginRankingなど)
    if isinstance(predictions, tuple):
        # MarginRankingTrainerは (pos_score, neg_score) を返すよう実装した場合
        # ここでの精度計算は定義が難しいのでダミーを返すか、
        # Pos > Neg となっている割合（Accuracy）を計算する
        pos_scores = predictions[0] # (batch,)
        if len(predictions) > 1:
            neg_scores = predictions[1] # (batch,)
            # Pos > Neg なら正解 (1), 逆なら不正解 (0)
            # numpy配列であることを想定
            acc = (pos_scores > neg_scores).mean()
            return {"accuracy": acc}
        else:
            return {}
            
    # 通常の分類 (Logits)
    preds = (predictions > 0).astype(int).reshape(-1)
    if labels is not None:
        labels = labels.astype(int).reshape(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}
    
    return {}

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
    
    # --- Trainerとモデルの選択ロジック ---
    compute_metrics_func = None

    if cfg.model.type == "cross_encoder":
        # Cross-Encoder (Reranker)
        model_config.num_labels = 1
        model = CrossEncoderMarginModel.from_pretrained(cfg.model.name, config=model_config)
        trainer_cls = MarginRankingTrainer
        # Cross-Encoderでは正解率(Pos > Negの割合)を計算させるとモニタリングに良い
        compute_metrics_func = compute_metrics
        
    else: # bi_encoder
        head_type = cfg.model.get("head_type", "ranknet")
        loss_type = cfg.training.get("loss_type", "pair_score")
        
        print(f"Bi-Encoder Head: {head_type}, Loss: {loss_type}")
        
        model = SiameseBiEncoder.from_pretrained(
            cfg.model.name, 
            config=model_config, 
            head_type=head_type
        )
        
        adapter_name = cfg.model.get("adapter_name", None)
        if adapter_name:
            print(f"🔄 Loading Adapter: {adapter_name}")
            adapters.init(model.bert)
            try:
                loaded_name = model.bert.load_adapter(adapter_name, source="hf", set_active=True)
            except:
                loaded_name = model.bert.load_adapter(adapter_name, set_active=True)
            
            model.bert.set_active_adapters(loaded_name)
            print(f"✅ Adapter '{loaded_name}' activated.")
            
            if cfg.model.get("freeze_base", False):
                print("❄️  Freezing base model parameters (Training Adapter only)")
                for param in model.bert.parameters():
                    param.requires_grad = False
                for name, param in model.named_parameters():
                    if "adapter" in name or "classifier_head" in name:
                        param.requires_grad = True

        if loss_type == "mnrl":
            print("Using MultipleNegativesRankingTrainer")
            trainer_cls = MultipleNegativesRankingTrainer
        elif head_type == "none":
            print("Using ContrastiveTrainer")
            trainer_cls = ContrastiveTrainer
            compute_metrics_func = compute_metrics
        else:
            print("Using BiEncoderPairTrainer")
            trainer_cls = BiEncoderPairTrainer
            compute_metrics_func = compute_metrics

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
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=True,
        report_to="wandb" if cfg.logging.use_wandb else "none",
        run_name=cfg.logging.run_name, 
        remove_unused_columns=False,
        metric_for_best_model="loss", # MarginRankingLossは小さい方が良い
        greater_is_better=False,
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1)
    )
    
    callbacks = []
    if cfg.training.get("patience"):
        print(f"Early stopping enabled with patience: {cfg.training.patience}")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.training.patience))
        
    # ★追加: コレーターの準備
    data_collator = None
    if cfg.model.type == "cross_encoder":
        print("Using CrossEncoderTripletCollator")
        data_collator = CrossEncoderTripletCollator(tokenizer)

    # 6. Trainer初期化
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "tokenizer": tokenizer,
        "optimizers": (optimizer, None),
        "compute_metrics": compute_metrics_func,
        "callbacks": callbacks,
        "data_collator": data_collator # ★追加
    }

    if trainer_cls == MultipleNegativesRankingTrainer:
        trainer_kwargs["scale"] = cfg.training.get("scale", 20.0)
    elif trainer_cls in [ContrastiveTrainer, MarginRankingTrainer]:
        trainer_kwargs["margin"] = cfg.training.margin
    
    trainer = trainer_cls(**trainer_kwargs)
    
    print("Starting training...")
    trainer.train()
    
    best_model_path = os.path.join(output_dir, "best_model")
    print(f"Saving best model to {best_model_path}")
    trainer.save_model(best_model_path)
    
    if cfg.logging.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()