# src/training/dataset.py

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

        # 必要な列の欠損除去
        # フォーマットによって必要な列が異なるため、簡易チェック
        cols_to_check = ['query', 'positive'] if 'query' in df.columns else ['abstract_a', 'abstract_b']
        df = df.dropna(subset=cols_to_check)
        
        if 'label' in df.columns:
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

        # ▼▼▼ Triplet形式でない場合のみ、MNRL用に正例フィルタリングを行う ▼▼▼
        loss_type = self.config.training.get("loss_type", "pair_score")
        data_format = self.config.data.get("format", "pair")

        # MNRLかつ、Hard Negativeを使わない(pair形式の)場合のみフィルタリング
        # (Cross-Encoderの場合はMNRLを使わないので影響なし)
        if self.config.model.type == "bi_encoder" and loss_type == "mnrl" and data_format != "triplet":
            print(f"\n[Dataset] ℹ️  Filtering dataset for MNRL (format={data_format})")
            print("          Keeping only Positive samples (label=1) for In-Batch Negatives")
            original_len = len(df)
            df = df[df['label'] == 1]
            print(f"          Rows filtered: {original_len} -> {len(df)}")
            
            if len(df) == 0:
                raise ValueError("No positive samples found! MNRL requires positive pairs.")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # --- データ形式の変換 (必要に応じて) ---
        if self.config.data.format == "triplet":
            # 既にTripletのカラム(anchor, positive, negative)があるなら変換不要
            if not {'anchor', 'positive', 'negative'}.issubset(df.columns):
                print("Converting dataset to Triplets (Anchor, Positive, Negative)...")
                df = self._convert_to_triplets(df)
            
            # Triplet変換後にデータが残っているか確認
            if len(df) == 0:
                raise ValueError("Triplet conversion resulted in empty dataset! Check if Hard Negatives (label=0) exist in input file.")
        
        # --- データ分割 (GroupShuffleSplit) ---
        print("Performing Group Shuffle Split...")
        
        # グループ化のキー決定 (Query ID的なもの)
        if self.config.data.format == "triplet":
            group_col = 'anchor' if 'anchor' in df.columns else 'query'
        else:
            group_col = 'abstract_a' if 'abstract_a' in df.columns else 'query'

        groups = df[group_col]
        
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
        
        cols_to_remove = list(df.columns)
        if "__index_level_0__" in raw_datasets['train'].column_names:
            cols_to_remove.append("__index_level_0__")

        tokenized_datasets = raw_datasets.map(
            self._get_tokenize_function(),
            batched=True,
            num_proc=self.config.data.num_proc,
            remove_columns=cols_to_remove
        )
        tokenized_datasets.set_format("torch")
        
        return tokenized_datasets

    def _convert_to_triplets(self, df):
        # カラム名の標準化 (query/positive/negative がなければ abstract_a/b から作る)
        if 'query' in df.columns and 'positive' in df.columns and 'negative' in df.columns:
            return df # 既にTriplet形式

        # 以下は abstract_a, abstract_b, label 形式からの変換
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        
        if len(neg_df) == 0:
            print("⚠️ Warning: No negatives found for Triplet conversion. Returning empty DataFrame.")
            return pd.DataFrame()

        neg_map = neg_df.groupby('abstract_a')['abstract_b'].apply(list).to_dict()
        
        triplets = []
        for _, row in pos_df.iterrows():
            anchor = row['abstract_a']
            positive = row['abstract_b']
            
            hard_negatives = neg_map.get(anchor, [])
            if hard_negatives:
                # 負例の中からランダムに1つ選んで Triplet を作る
                negative = np.random.choice(hard_negatives)
                triplets.append({
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative
                })
        
        return pd.DataFrame(triplets)

    def _get_tokenize_function(self):
        max_len = self.config.model.max_length
        
        # Case 1: Cross-Encoder (Triplet)
        if self.config.model.type == "cross_encoder":
            def tokenize_cross_triplet(examples):
                # カラム名の揺らぎ吸収
                anchors = examples.get("anchor", examples.get("query"))
                positives = examples.get("positive")
                negatives = examples.get("negative")

                # Positive Pair: [CLS] Query [SEP] Positive [SEP]
                # truncation="longest_first": クエリと文書の長い方を削り、両方が入るように調整する
                tokenized_pos = self.tokenizer(
                    anchors, positives,
                    padding="max_length", 
                    truncation="longest_first", # ★重要変更点
                    max_length=max_len
                )
                
                # Negative Pair: [CLS] Query [SEP] Negative [SEP]
                tokenized_neg = self.tokenizer(
                    anchors, negatives,
                    padding="max_length", 
                    truncation="longest_first", # ★重要変更点
                    max_length=max_len
                )
                
                return {
                    "input_ids_pos": tokenized_pos["input_ids"],
                    "attention_mask_pos": tokenized_pos["attention_mask"],
                    "input_ids_neg": tokenized_neg["input_ids"],
                    "attention_mask_neg": tokenized_neg["attention_mask"],
                }
            return tokenize_cross_triplet

        # Case 2: Bi-Encoder (Triplet for MNRL with Hard Negatives)
        elif self.config.data.format == "triplet":
            def tokenize_bi_triplet(examples):
                anchors = examples.get("anchor", examples.get("query"))
                positives = examples.get("positive")
                negatives = examples.get("negative")

                # Bi-Encoderは個別にエンコードするため truncation=True (末尾切り捨て) でOK
                tokenized_a = self.tokenizer(
                    anchors, padding="max_length", truncation=True, max_length=max_len
                )
                tokenized_p = self.tokenizer(
                    positives, padding="max_length", truncation=True, max_length=max_len
                )
                tokenized_n = self.tokenizer(
                    negatives, padding="max_length", truncation=True, max_length=max_len
                )
                return {
                    "input_ids": tokenized_a["input_ids"],
                    "attention_mask": tokenized_a["attention_mask"],
                    "input_ids_b": tokenized_p["input_ids"],
                    "attention_mask_b": tokenized_p["attention_mask"],
                    "input_ids_c": tokenized_n["input_ids"],
                    "attention_mask_c": tokenized_n["attention_mask"],
                    "labels": [0] * len(anchors) 
                }
            return tokenize_bi_triplet

        # Case 3: Bi-Encoder (Pair)
        else: 
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
        
class CrossEncoderTripletCollator:
    """
    Cross-Encoder用のカスタムコレーター。
    input_ids_pos / input_ids_neg を別々にパディングしてバッチ化する。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pos_features = []
        neg_features = []
        
        for f in features:
            # Positive側を取り出して標準キーに戻す
            pos_features.append({
                "input_ids": f["input_ids_pos"],
                "attention_mask": f["attention_mask_pos"]
            })
            # Negative側を取り出して標準キーに戻す
            neg_features.append({
                "input_ids": f["input_ids_neg"],
                "attention_mask": f["attention_mask_neg"]
            })
        
        # 別々にパディング実行
        batch_pos = self.tokenizer.pad(pos_features, padding=True, return_tensors="pt")
        batch_neg = self.tokenizer.pad(neg_features, padding=True, return_tensors="pt")
        
        # 結合して返す
        return {
            "input_ids_pos": batch_pos["input_ids"],
            "attention_mask_pos": batch_pos["attention_mask"],
            "input_ids_neg": batch_neg["input_ids"],
            "attention_mask_neg": batch_neg["attention_mask"],
        }