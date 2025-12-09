import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import sys
import os

# --- モデル定義 (修正版: src/modeling/cross_encoder.py と同じ構造) ---
class CrossEncoderMarginModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # ★重要: self.bert という名前で定義することで、学習済み重みが正しく読み込まれる
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
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
        # 正例ペアの推論
        outputs_pos = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # [CLS]ベクトル (batch_size, hidden_size)
        cls_pos = outputs_pos.last_hidden_state[:, 0, :]
        score_positive = self.classifier(cls_pos)

        # 負例ペアの推論（もし入力されていれば）
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
            
        return SequenceClassifierOutput(
            loss=None,
            logits=(score_positive, score_negative),
            hidden_states=outputs_pos.hidden_states,
            attentions=outputs_pos.attentions,
        )

# --- 設定 ---
# 再学習したモデルのパスを指定してください
MODEL_PATH = "models/checkpoints/cross_encoder/Cross_SPECTER2_HardNeg_v2/best_model"
BASE_NAME = "allenai/specter2_base"

def main():
    print(f"Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found: {MODEL_PATH}")
        print("Please make sure you have finished training with the fixed model definition.")
        return

    # コンフィグとモデルのロード
    config = AutoConfig.from_pretrained(MODEL_PATH)
    try:
        model = CrossEncoderMarginModel.from_pretrained(MODEL_PATH, config=config)
    except Exception as e:
        print(f"Load Error: {e}")
        print("Hint: モデル定義の変数名(self.bert vs self.scorer)が学習時と一致していない可能性があります。")
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_NAME)
    model.eval()
    
    # --- テストデータ ---
    query = "Machine learning methods for protein structure prediction."
    
    # 1. 正解 (Positive)
    pos_doc = "This study introduces AlphaFold, a deep learning system that predicts the 3D structure of proteins with high accuracy."
    
    # 2. 簡単な不正解 (Easy Negative) - 全く関係ない話題
    easy_neg_doc = "The Roman Empire was one of the largest empires in history, spanning the Mediterranean."
    
    # 3. 紛らわしい不正解 (Hard Negative) - 単語は似ているがタスクが違う
    hard_neg_doc = "We apply machine learning techniques to predict stock market trends using financial data."

    docs = [pos_doc, easy_neg_doc, hard_neg_doc]
    labels = ["Positive", "Easy Neg", "Hard Neg"]
    
    print("\n--- Scoring Check ---")
    print(f"Query: {query}\n")
    
    for label, doc in zip(labels, docs):
        inputs = tokenizer(
            query, 
            doc, 
            return_tensors="pt", 
            truncation="longest_first", 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            # CrossEncoderMarginModelは (score_pos, score_neg) のタプルを返す
            # ここでは入力が1つだけなので、tuple[0] にスコアが入っている
            logits_tuple = outputs.logits
            score = logits_tuple[0].item()
            
        print(f"[{label:<8}] Score: {score:.4f}")

    print("\nExpected Result: Positive > Hard Neg > Easy Neg")

if __name__ == "__main__":
    main()