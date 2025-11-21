import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class SiameseBiEncoder(BertPreTrainedModel):
    """
    SciBERTなどをバックボーンとし、SBERT流の分類ヘッド(差・積の結合)を持つSiameseモデル。
    RankNet LossやSoftmax Lossなどで学習可能。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # エンコーダー部分 (SciBERT)
        self.bert = AutoModel.from_config(config)
        
        # SBERT流の分類ヘッド
        # 特徴量: [u, v, |u-v|, u*v] -> 4倍の次元 (768 * 4 = 3072)
        self.classifier_head = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1) # スカラー値（スコア）を出力
        )
        
        self.post_init()

    def _get_vector(self, input_ids, attention_mask):
        """アブストラクトをベクトル化する（CLSトークン使用）"""
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.pooler_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids_b=None,
        attention_mask_b=None,
        input_ids_c=None,      # Tripletなどで3つ目の入力がある場合
        attention_mask_c=None,
        labels=None,
        output_vectors=False,  # 推論用: Trueならスコアではなくベクトルを返す
        **kwargs
    ):
        # 1. Anchor (Query) のベクトル化
        vec_a = self._get_vector(input_ids, attention_mask)
        
        # --- 推論モード（ベクトル化のみ） ---
        # 検索インデックス作成時などは、ここでベクトルだけを返して終了
        if output_vectors or (input_ids_b is None):
            return SequenceClassifierOutput(logits=vec_a)

        # 2. Positive / Candidate のベクトル化
        vec_b = self._get_vector(input_ids_b, attention_mask_b)
        
        # --- ペアのスコア計算 (分類ヘッド使用) ---
        # 特徴量生成: u, v, |u-v|, u*v
        diff = torch.abs(vec_a - vec_b)
        prod = vec_a * vec_b
        features = torch.cat([vec_a, vec_b, diff, prod], dim=1)
        
        # ヘッドに通してスコア算出
        score_pos = self.classifier_head(features)

        # 3. Negativeがある場合 (Triplet的な入力の場合)
        score_neg = None
        if input_ids_c is not None:
            vec_c = self._get_vector(input_ids_c, attention_mask_c)
            
            diff_c = torch.abs(vec_a - vec_c)
            prod_c = vec_a * vec_c
            features_c = torch.cat([vec_a, vec_c, diff_c, prod_c], dim=1)
            
            score_neg = self.classifier_head(features_c)
            
            # 戻り値: (Positiveスコア, Negativeスコア)
            return SequenceClassifierOutput(loss=None, logits=(score_pos, score_neg))

        # 戻り値: (Positiveスコア)
        return SequenceClassifierOutput(loss=None, logits=score_pos)