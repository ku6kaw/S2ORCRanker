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
from src.training.dataset import TextRankingDataset
from src.training.trainer import BiEncoderPairTrainer, MarginRankingTrainer, ContrastiveTrainer, MultipleNegativesRankingTrainer

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


    """
    Contrastive Loss (距離学習) 用の評価指標計算。
    距離が threshold 以下なら「類似（Positive）」と判定する。
    """
    # ContrastiveTrainerは (vec_a, vec_b) を返すので、これを受け取る必要があるが、
    # Hugging Face Trainerの compute_metrics は (predictions, label_ids) を受け取る仕様。
    # predictions は logits のタプルになるはず。
    
    logits, labels = eval_pred
    # logits は (vec_a, vec_b) のタプル
    vec_a = torch.tensor(logits[0])
    vec_b = torch.tensor(logits[1])
    
    # 距離計算 (Euclidean)
    distances = torch.nn.functional.pairwise_distance(vec_a, vec_b).numpy()
    
    # 予測: 距離が閾値以下なら 1 (Positive), それ以外は 0 (Negative)
    # ※ Contrastiveでは「距離が近いほど似ている」ため
    preds = (distances < threshold).astype(int)
    labels = labels.astype(int).reshape(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mean_distance': float(distances.mean()) # 平均距離もログに残すと便利
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
        head_type = cfg.model.get("head_type", "ranknet")
        # ★追加: 損失関数のタイプ設定 (デフォルトは互換性のため pair_score)
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
            
            # adaptersライブラリで初期化
            adapters.init(model.bert)
            
            # アダプターをロードし、その内部名(例: '[PRX]')を取得
            loaded_name = model.bert.load_adapter(adapter_name, source="hf", set_active=True)
            
            # 念のため明示的にアクティブ化
            model.bert.set_active_adapters(loaded_name)
            print(f"✅ Adapter '{loaded_name}' activated.")
            
            if cfg.model.get("freeze_base", False):
                print("❄️  Freezing base model parameters (Training Adapter only)")
                # 全パラメータを一旦凍結
                for param in model.bert.parameters():
                    param.requires_grad = False
                
                # アダプター部分と分類ヘッド(もしあれば)のみ解凍
                for name, param in model.named_parameters():
                    if "adapter" in name or "classifier_head" in name:
                        param.requires_grad = True
                        
                # 確認: 学習対象パラメータ数
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                all_params = sum(p.numel() for p in model.parameters())
                print(f"   Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
        
        if loss_type == "mnrl":
            # Multiple Negatives Ranking Loss
            print("Using MultipleNegativesRankingTrainer (Batch Negatives)")
            trainer_cls = MultipleNegativesRankingTrainer
        
        elif head_type == "none":
            # Contrastive
            print("Using ContrastiveTrainer (Distance-based)")
            trainer_cls = ContrastiveTrainer
        else:
            # RankNet / BCE
            print("Using BiEncoderPairTrainer (Classification Head)")
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
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    
    # ★ コールバックの準備
    callbacks = []
    if cfg.training.get("patience"):
        print(f"Early stopping enabled with patience: {cfg.training.patience}")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.training.patience))
        
    # 6. Trainer初期化
    # 共通の引数
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "tokenizer": tokenizer,
        "optimizers": (optimizer, None),
        "compute_metrics": compute_metrics_func,
        "callbacks": callbacks
    }

    # Trainerクラスに応じた追加引数
    if trainer_cls == MultipleNegativesRankingTrainer:
        trainer_kwargs["scale"] = cfg.training.get("scale", 20.0)
    elif trainer_cls in [ContrastiveTrainer, MarginRankingTrainer]:
        trainer_kwargs["margin"] = cfg.training.margin
    
    # Trainer初期化
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