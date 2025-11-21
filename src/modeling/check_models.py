import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from src.modeling.cross_encoder import CrossEncoderMarginModel
from src.modeling.bi_encoder import SiameseBiEncoder

try:
    from transformers import LongformerPreTrainedModel
except ImportError:
    from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel


def test_cross_encoder():
    print("\n=== Testing Cross-Encoder (Longformer) ===")
    model_name = "allenai/longformer-base-4096"
    try:
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        
        model = CrossEncoderMarginModel(config)
        print("✅ Model initialized successfully.")
        
        # ダミー入力
        input_ids = torch.randint(0, 1000, (2, 16))
        mask = torch.ones((2, 16))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=mask,
            input_ids_neg=input_ids, attention_mask_neg=mask
        )
        
        print(f"✅ Forward pass successful.")
        print(f"   Pos Score shape: {outputs.logits[0].shape}")

    except Exception as e:
        print(f"❌ Error: {e}")

def test_bi_encoder():
    print("\n=== Testing Bi-Encoder (SciBERT with Head) ===")
    model_name = "allenai/scibert_scivocab_uncased"
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        # 分類ヘッド付きで初期化されるか確認
        model = SiameseBiEncoder(config)
        print("✅ Model initialized successfully.")
        
        input_ids = torch.randint(0, 1000, (2, 16))
        mask = torch.ones((2, 16))
        
        # 1. ペア入力（スコア計算）
        out_pair = model(input_ids=input_ids, attention_mask=mask, input_ids_b=input_ids, attention_mask_b=mask)
        print(f"✅ Pair Scoring output shape: {out_pair.logits.shape} (Expected: [2, 1])")

        # 2. ベクトル化のみ（インデックス作成用）
        out_vec = model(input_ids=input_ids, attention_mask=mask, output_vectors=True)
        print(f"✅ Vectorization output shape: {out_vec.logits.shape} (Expected: [2, 768])")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cross_encoder()
    test_bi_encoder()