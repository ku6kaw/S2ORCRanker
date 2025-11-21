# src/modeling/cross_encoder.py

import torch
import torch.nn as nn
from transformers import LongformerPreTrainedModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class CrossEncoderMarginModel(LongformerPreTrainedModel):
    """
    Longformerを使用したCross-Encoderモデル。
    MarginRankingLossでの学習用に、(positive_score, negative_score) のペアを出力します。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # AutoModelForSequenceClassificationを使ってバックボーンと分類ヘッドを初期化
        # num_labels=1 とすることで、スカラー値（スコア）を出力するように設定
        self.scorer = AutoModelForSequenceClassification.from_config(config)
        
        # 重みの初期化（親クラスのメソッドを使用）
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        labels=None, 
        **kwargs
    ):
        """
        Args:
            input_ids, attention_mask: 正例ペア（Anchor + Positive）の入力
            input_ids_neg, attention_mask_neg: 負例ペア（Anchor + Negative）の入力
        """
        # 正例ペアのスコア計算
        output_pos = self.scorer(input_ids=input_ids, attention_mask=attention_mask)
        score_positive = output_pos.logits

        # 負例ペアのスコア計算（もし入力されていれば）
        score_negative = None
        if input_ids_neg is not None:
            output_neg = self.scorer(input_ids=input_ids_neg, attention_mask=attention_mask_neg)
            score_negative = output_neg.logits

        # 損失計算はTrainer側（loss関数）で行うため、这里ではNoneを返すか、
        # 必要ならここで計算してもよいが、今回はlogitsとしてペアを返す設計にする
        
        return SequenceClassifierOutput(
            loss=None,
            logits=(score_positive, score_negative),
            hidden_states=None,
            attentions=None,
        )