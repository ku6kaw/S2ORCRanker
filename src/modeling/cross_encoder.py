# src/modeling/cross_encoder.py

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

class CrossEncoderMarginModel(BertPreTrainedModel):
    """
    BERTベースのCross-Encoderモデル。
    MarginRankingLossでの学習用に、(positive_score, negative_score) のペアを出力します。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # ★修正点: self.scorer ではなく、標準的な self.bert を使用
        # これにより "allenai/specter2_base" の重みが自動的にマッチしてロードされる
        self.bert = BertModel(config)
        
        # 分類ヘッド (1次元出力)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # 重みの初期化 (Headなどはランダム初期化、BERT部分はロード時に上書きされる)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        token_type_ids_neg=None,
        labels=None, 
        **kwargs
    ):
        """
        Args:
            input_ids, attention_mask: 正例ペア（Anchor + Positive）の入力
            input_ids_neg, attention_mask_neg: 負例ペア（Anchor + Negative）の入力
        """
        # --- 1. 正例ペアの推論 ---
        outputs_pos = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # [CLS]トークンのベクトルを取得
        cls_pos = outputs_pos.last_hidden_state[:, 0, :]
        # スコア計算
        score_positive = self.classifier(cls_pos)

        # --- 2. 負例ペアの推論（もし入力されていれば）---
        score_negative = None
        if input_ids_neg is not None:
            outputs_neg = self.bert(
                input_ids=input_ids_neg,
                attention_mask=attention_mask_neg,
                token_type_ids=token_type_ids_neg,
                return_dict=True
            )
            cls_neg = outputs_neg.last_hidden_state[:, 0, :]
            score_negative = self.classifier(cls_neg)

        # lossはTrainerで計算するのでNone、logitsにペアを入れて返す
        return SequenceClassifierOutput(
            loss=None,
            logits=(score_positive, score_negative),
            hidden_states=outputs_pos.hidden_states,
            attentions=outputs_pos.attentions,
        )