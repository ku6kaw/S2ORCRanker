# src/training/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class ContrastiveTrainer(Trainer):
    """
    Bi-Encoder用: Contrastive Loss (距離学習) を使用するTrainer。
    モデル出力 (vec_a, vec_b) を期待する。
    """
    def __init__(self, *args, margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None # ContrastiveTrainerの仕様によってはlabelsがない場合も考慮

        outputs = model(**inputs)
        
        # SiameseBiEncoder(head_type="none") は logits=(vec_a, vec_b) を返す
        vec_a, vec_b = outputs.logits
        
        distance = F.pairwise_distance(vec_a, vec_b)
        
        loss_positive = distance.pow(2)
        loss_negative = F.relu(self.margin - distance).pow(2)
        
        # ラベル定義: 1=Positive, 0=Negative
        loss = (labels.float() * loss_positive) + ((1 - labels.float()) * loss_negative)
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

class BiEncoderPairTrainer(Trainer):
    """
    Bi-Encoder用: 分類ヘッドあり (RankNet/BCE) を使用するTrainer。
    モデル出力 score を期待する。
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        # SiameseBiEncoder(head_type="ranknet") は logits=score を返す
        scores = outputs.logits.squeeze(-1)
        
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(scores, labels.float())

        return (loss, outputs) if return_outputs else loss

class MarginRankingTrainer(Trainer):
    """
    Cross-Encoder用: Margin Ranking Loss を使用するTrainer。
    Datasetからは {input_ids_pos, ..., input_ids_neg, ...} が返ってくる想定。
    """
    def __init__(self, *args, margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.loss_fct = nn.MarginRankingLoss(margin=self.margin)

    def _get_logits(self, outputs):
        """モデル出力から安全にlogitsを取り出すヘルパー関数"""
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        # 万が一 logits 自体がタプルの場合も考慮
        if isinstance(logits, tuple):
            logits = logits[0]
            
        return logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            inputs.pop("labels")
            
        # 1. Positive Pair の推論
        pos_inputs = {
            "input_ids": inputs["input_ids_pos"],
            "attention_mask": inputs["attention_mask_pos"]
        }
        if "token_type_ids_pos" in inputs:
            pos_inputs["token_type_ids"] = inputs["token_type_ids_pos"]

        pos_outputs = model(**pos_inputs)
        pos_logits = self._get_logits(pos_outputs) # ★修正: 安全に取り出す
        pos_scores = pos_logits.squeeze(-1) 

        # 2. Negative Pair の推論
        neg_inputs = {
            "input_ids": inputs["input_ids_neg"],
            "attention_mask": inputs["attention_mask_neg"]
        }
        if "token_type_ids_neg" in inputs:
            neg_inputs["token_type_ids"] = inputs["token_type_ids_neg"]

        neg_outputs = model(**neg_inputs)
        neg_logits = self._get_logits(neg_outputs) # ★修正: 安全に取り出す
        neg_scores = neg_logits.squeeze(-1)

        # 3. Loss 計算
        # target=1 means pos should be higher than neg
        target = torch.ones_like(pos_scores)
        loss = self.loss_fct(pos_scores, neg_scores, target)

        return (loss, pos_outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        評価ループ用
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        
        # logitsを取得して返す
        logits = self._get_logits(outputs)
        return (loss, logits, None)
    
class MultipleNegativesRankingTrainer(Trainer):
    """
    Bi-Encoder用: Multiple Negatives Ranking Loss (MNRL)
    """
    def __init__(self, *args, scale=20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale 
        self.cross_entropy = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            inputs.pop("labels")
        
        outputs = model(**inputs)
        
        if isinstance(outputs.logits, tuple) and len(outputs.logits) == 3:
            vec_a, vec_p, vec_n = outputs.logits
            has_hard_neg = True
        else:
            vec_a, vec_p = outputs.logits
            has_hard_neg = False

        scores_p = torch.matmul(vec_a, vec_p.transpose(0, 1)) * self.scale
        
        if has_hard_neg:
            scores_n = torch.matmul(vec_a, vec_n.transpose(0, 1)) * self.scale
            scores = torch.cat([scores_p, scores_n], dim=1)
        else:
            scores = scores_p

        labels = torch.arange(scores.size(0), device=scores.device)
        
        loss = self.cross_entropy(scores, labels)

        return (loss, outputs) if return_outputs else loss