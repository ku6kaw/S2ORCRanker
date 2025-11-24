# src/modeling/bi_encoder.py

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class SiameseBiEncoder(BertPreTrainedModel):
    """
    SciBERTなどをバックボーンとするSiameseモデル。
    head_type="ranknet": 分類ヘッドあり (RankNet/BCE用)
    head_type="none":    分類ヘッドなし (Contrastive/Triplet用)
    """
    def __init__(self, config, head_type="ranknet"): # ★引数追加 (デフォルトはranknetで互換性維持)
        super().__init__(config)
        self.head_type = head_type
        
        # エンコーダー部分 (SciBERT)
        self.bert = AutoModel.from_config(config)
        
        # 分類ヘッドの作成 (ranknetの場合のみ)
        if self.head_type == "ranknet":
            self.classifier_head = nn.Sequential(
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )
        
        self.post_init()

    def _get_vector(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.pooler_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids_b=None,
        attention_mask_b=None,
        input_ids_c=None,
        attention_mask_c=None,
        labels=None,
        output_vectors=False,
        **kwargs
    ):
        # 1. Anchor
        vec_a = self._get_vector(input_ids, attention_mask)
        
        # 推論用
        if output_vectors or (input_ids_b is None):
            return SequenceClassifierOutput(logits=vec_a)

        # 2. Positive / Candidate
        vec_b = self._get_vector(input_ids_b, attention_mask_b)
        
        # --- ★分岐: ヘッドなし (Contrastive) ---
        if self.head_type == "none":
            # ベクトルそのものを返す -> ContrastiveTrainerで距離計算する
            return SequenceClassifierOutput(loss=None, logits=(vec_a, vec_b))

        # --- ★分岐: ヘッドあり (RankNet) ---
        diff = torch.abs(vec_a - vec_b)
        prod = vec_a * vec_b
        features = torch.cat([vec_a, vec_b, diff, prod], dim=1)
        
        score_pos = self.classifier_head(features)

        # 3. Negativeがある場合 (Triplet的入力)
        score_neg = None
        if input_ids_c is not None:
            vec_c = self._get_vector(input_ids_c, attention_mask_c)
            diff_c = torch.abs(vec_a - vec_c)
            prod_c = vec_a * vec_c
            features_c = torch.cat([vec_a, vec_c, diff_c, prod_c], dim=1)
            score_neg = self.classifier_head(features_c)
            return SequenceClassifierOutput(loss=None, logits=(score_pos, score_neg))

        return SequenceClassifierOutput(loss=None, logits=score_pos)