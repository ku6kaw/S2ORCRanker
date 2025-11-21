# src/training/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class BiEncoderPairTrainer(Trainer):
    """
    分類ヘッド付きBi-Encoder用Trainer。
    モデルが出力するスコア(logits)と、正解ラベル(0/1)でBCEWithLogitsLossを計算します。
    """
    # ▼▼▼ 追加: marginを受け取るための__init__ ▼▼▼
    def __init__(self, *args, margin=1.0, **kwargs):
        # marginはBCEでは使わないが、run_train.pyから渡されるため
        # ここで受け取って消費し、親クラス(Trainer)には渡さないようにする
        super().__init__(*args, **kwargs)
        self.margin = margin # 一応保持しておく（使わない）

    def compute_loss(self, model, inputs, return_outputs=False):
        # ラベルを取り出す
        labels = inputs.pop("labels")
        
        # モデルの推論
        outputs = model(**inputs)
        
        # モデル出力は (batch_size, 1) の形状をしているため、(batch_size) に変形
        scores = outputs.logits.squeeze(-1)
        
        # BCEWithLogitsLoss (Sigmoid + BCE) で損失を計算
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(scores, labels.float())

        return (loss, outputs) if return_outputs else loss

class MarginRankingTrainer(Trainer):
    """Cross-Encoder用: Margin Ranking Loss を使用するTrainer"""
    def __init__(self, *args, margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.loss_fct = nn.MarginRankingLoss(margin=self.margin)

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            inputs.pop("labels")
            
        outputs = model(**inputs)
        # (pos_score, neg_score) のペア
        score_pos, score_neg = outputs.logits
        
        target = torch.ones_like(score_pos)
        loss = self.loss_fct(score_pos, score_neg, target)

        return (loss, outputs) if return_outputs else loss