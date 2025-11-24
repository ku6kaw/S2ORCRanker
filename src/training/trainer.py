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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
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
    def compute_loss(self, model, inputs, return_outputs=False):
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
    (Pos, Neg) のTriplet入力を期待する。
    """
    def __init__(self, *args, margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.loss_fct = nn.MarginRankingLoss(margin=self.margin)

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            inputs.pop("labels")
            
        outputs = model(**inputs)
        # CrossEncoderMarginModelは (pos_score, neg_score) を返す
        score_pos, score_neg = outputs.logits
        
        target = torch.ones_like(score_pos)
        loss = self.loss_fct(score_pos, score_neg, target)

        return (loss, outputs) if return_outputs else loss