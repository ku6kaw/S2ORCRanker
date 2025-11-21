import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

class TextRankingDataset:
    """
    論文ランキングタスク用のデータセットハンドラ。
    CSVの読み込み、GroupShuffleSplitによる分割、トークナイズを行います。
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load_and_prepare(self):
        """データを読み込み、分割し、トークナイズしたDatasetDictを返す"""
        print(f"Loading dataset from: {self.config.data.train_file}")
        df = pd.read_csv(self.config.data.train_file)
        
        # デバッグモード時のデータ削減処理
        if self.config.get("debug", False):
            print("\n" + "="*50)
            print(" ⚠️  DEBUG MODE ENABLED: Truncating dataset to 100 samples!")
            print("="*50 + "\n")
            df = df.head(100)

        df = df.dropna(subset=['abstract_a', 'abstract_b', 'label'])
        df['label'] = df['label'].astype(int)

        # --- データ形式の変換 (必要に応じて) ---
        if self.config.data.format == "triplet":
            print("Converting dataset to Triplets (Anchor, Positive, Negative)...")
            df = self._convert_to_triplets(df)
        
        # --- データ分割 (GroupShuffleSplit) ---
        print("Performing Group Shuffle Split based on 'anchor' (or 'abstract_a')...")
        
        group_col = 'anchor' if self.config.data.format == "triplet" else 'abstract_a'
        groups = df[group_col]
        
        # データ数が少なすぎて分割できない場合のエラー回避
        n_splits = 1
        if len(df) < 5:
             print("Data too small for split, using same data for train/val")
             train_df = df
             val_df = df
        else:
            try:
                gss = GroupShuffleSplit(
                    n_splits=n_splits, 
                    test_size=self.config.data.val_size, 
                    random_state=self.config.seed
                )
                train_idx, val_idx = next(gss.split(df, groups=groups))
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
            except Exception as e:
                print(f"Split failed (likely due to debug size): {e}. Using full data for both.")
                train_df = df
                val_df = df
        
        print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}")

        # Hugging Face Dataset に変換
        raw_datasets = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df)
        })

        # --- トークナイズ ---
        print("Tokenizing...")
        
        # ▼▼▼ 修正点: remove_columns に渡す値を list() で明示的に変換 ▼▼▼
        cols_to_remove = list(df.columns)
        
        # Dataset.from_pandas でインデックス列が追加されている場合があるので安全のため削除リストに追加
        if "__index_level_0__" in raw_datasets['train'].column_names:
            cols_to_remove.append("__index_level_0__")

        tokenized_datasets = raw_datasets.map(
            self._get_tokenize_function(),
            batched=True,
            num_proc=self.config.data.num_proc,
            remove_columns=cols_to_remove # list(df.columns)
        )
        tokenized_datasets.set_format("torch")
        
        return tokenized_datasets

    def _convert_to_triplets(self, df):
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        
        neg_map = neg_df.groupby('abstract_a')['abstract_b'].apply(list).to_dict()
        
        triplets = []
        for _, row in pos_df.iterrows():
            anchor = row['abstract_a']
            positive = row['abstract_b']
            
            hard_negatives = neg_map.get(anchor, [])
            if hard_negatives:
                negative = np.random.choice(hard_negatives)
                triplets.append({
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative
                })
        
        return pd.DataFrame(triplets)

    def _get_tokenize_function(self):
        max_len = self.config.model.max_length
        
        if self.config.model.type == "cross_encoder":
            def tokenize_triplet(examples):
                tokenized_pos = self.tokenizer(
                    examples["anchor"], examples["positive"],
                    padding="max_length", truncation=True, max_length=max_len
                )
                tokenized_neg = self.tokenizer(
                    examples["anchor"], examples["negative"],
                    padding="max_length", truncation=True, max_length=max_len
                )
                return {
                    "input_ids": tokenized_pos["input_ids"],
                    "attention_mask": tokenized_pos["attention_mask"],
                    "input_ids_neg": tokenized_neg["input_ids"],
                    "attention_mask_neg": tokenized_neg["attention_mask"],
                }
            return tokenize_triplet

        else: # bi_encoder
            def tokenize_pair(examples):
                tokenized_a = self.tokenizer(
                    examples["abstract_a"], padding="max_length", truncation=True, max_length=max_len
                )
                tokenized_b = self.tokenizer(
                    examples["abstract_b"], padding="max_length", truncation=True, max_length=max_len
                )
                return {
                    "input_ids": tokenized_a["input_ids"],
                    "attention_mask": tokenized_a["attention_mask"],
                    "input_ids_b": tokenized_b["input_ids"],
                    "attention_mask_b": tokenized_b["attention_mask"],
                    "labels": examples["label"]
                }
            return tokenize_pair