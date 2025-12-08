import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# 学習済みモデルのパス
MODEL_PATH = "models/checkpoints/cross_encoder/Cross_SPECTER2_HardNeg/best_model"
BASE_NAME = "allenai/specter2_base"

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(BASE_NAME)
    model.eval()
    
    # テストケース
    query = "Machine learning for biology"
    
    # 1. 明らかな正解 (Positive)
    pos_doc = "This paper proposes a deep learning method for protein structure prediction in biology."
    
    # 2. 明らかな不正解 (Easy Negative)
    neg_doc = "The history of the Roman Empire is very interesting."
    
    # 3. 紛らわしい不正解 (Hard Negative)
    hard_neg_doc = "Machine learning is widely used in finance for stock prediction."

    docs = [pos_doc, neg_doc, hard_neg_doc]
    labels = ["Positive", "Easy Neg", "Hard Neg"]
    
    print("\n--- Scoring Check ---")
    for label, doc in zip(labels, docs):
        inputs = tokenizer(
            query, 
            doc, 
            return_tensors="pt", 
            truncation="longest_first", 
            max_length=512
        )
        
        with torch.no_grad():
            score = model(**inputs).logits.item()
            
        print(f"[{label}] Score: {score:.4f}")

if __name__ == "__main__":
    main()