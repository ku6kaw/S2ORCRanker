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
    (Pos, Neg) のTriplet入力を期待する。
    """
    def __init__(self, *args, margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.loss_fct = nn.MarginRankingLoss(margin=self.margin)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            inputs.pop("labels")
            
        outputs = model(**inputs)
        # CrossEncoderMarginModelは (pos_score, neg_score) を返す
        score_pos, score_neg = outputs.logits
        
        target = torch.ones_like(score_pos)
        loss = self.loss_fct(score_pos, score_neg, target)

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        評価ループで1バッチごとの予測（とLoss計算）を行う関数。
        デフォルトではモデルがlossを返さないとloss=Noneになるため、
        ここで明示的に compute_loss を呼ぶ。
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs.logits, None)
    
class MultipleNegativesRankingTrainer(Trainer):
    """
    Bi-Encoder用: Multiple Negatives Ranking Loss (MNRL)
    (Anchor, Positive, HardNegative) の3つを受け取り、
    バッチ内全ての他サンプル + HardNegative を負例として学習する。
    """
    def __init__(self, *args, scale=20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale # 類似度をスケーリングする値 (Temperatureの逆数)
        self.cross_entropy = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            inputs.pop("labels")
        
        outputs = model(**inputs)
        
        # SiameseBiEncoderは head_type="none" の場合、(vec_a, vec_b, vec_c) を返す想定
        # vec_c (Negative) がない場合は (vec_a, vec_b)
        if isinstance(outputs.logits, tuple) and len(outputs.logits) == 3:
            vec_a, vec_p, vec_n = outputs.logits
            has_hard_neg = True
        else:
            vec_a, vec_p = outputs.logits
            has_hard_neg = False

        # 1. AnchorとPositive/Negativeの類似度行列を計算
        # (Batch, Dim) @ (Dim, Batch) -> (Batch, Batch)
        # scores_p[i][j] = Anchor[i] と Positive[j] の類似度
        scores_p = torch.matmul(vec_a, vec_p.transpose(0, 1)) * self.scale
        
        if has_hard_neg:
            # scores_n[i][j] = Anchor[i] と HardNegative[j] の類似度
            scores_n = torch.matmul(vec_a, vec_n.transpose(0, 1)) * self.scale
            
            # 結合: [Positives (Batch), HardNegatives (Batch)]
            # 横方向に結合 -> (Batch, Batch * 2)
            scores = torch.cat([scores_p, scores_n], dim=1)
        else:
            scores = scores_p

        # 2. 正解ラベルの作成
        # i番目のAnchorの正解は、i番目のPositive (つまり対角成分)
        # ラベルは [0, 1, 2, ..., batch_size-1]
        labels = torch.arange(scores.size(0), device=scores.device)
        
        # 3. Cross Entropy Loss
        loss = self.cross_entropy(scores, labels)

        return (loss, outputs) if return_outputs else loss