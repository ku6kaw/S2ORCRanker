# src/utils/optimization.py

import torch

def create_optimizer_grouped_parameters(model, base_lr, head_lr, weight_decay):
    """
    モデルのパラメータを4つのグループに分け、学習率とWeight Decayを適用する。
    1. Head (Decayあり)
    2. Head (Decayなし)
    3. Base (Decayあり)
    4. Base (Decayなし)

    Args:
        model: PyTorchモデル
        base_lr: ベースモデル（バックボーン）の学習率
        head_lr: 分類ヘッド（新規層）の学習率
        weight_decay: Weight Decayの値

    Returns:
        optimizer_grouped_parameters: オプティマイザに渡すパラメータグループのリスト
    """
    # Weight Decayを適用しないパラメータ名（BiasやLayerNorm）
    no_decay = ["bias", "LayerNorm.weight"]
    
    # 分類ヘッドとみなすパラメータ名のキーワード
    # Bi-Encoder: "classifier_head"
    # Cross-Encoder (Longformer/BGE): "scorer.classifier", "classifier", "score" 等
    head_keywords = ["classifier", "head", "score"]

    # 全パラメータを走査して振り分け
    head_params_decay = []
    head_params_no_decay = []
    base_params_decay = []
    base_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # ヘッドかどうか判定
        is_head = any(k in name for k in head_keywords)
        # Decayなしかどうか判定
        is_no_decay = any(nd in name for nd in no_decay)

        if is_head:
            if is_no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)
        else:
            if is_no_decay:
                base_params_no_decay.append(param)
            else:
                base_params_decay.append(param)

    # グループ定義の作成
    optimizer_grouped_parameters = []

    if head_params_decay:
        optimizer_grouped_parameters.append({
            "params": head_params_decay,
            "weight_decay": weight_decay,
            "lr": head_lr,
            "name": "head_decay"
        })
    if head_params_no_decay:
        optimizer_grouped_parameters.append({
            "params": head_params_no_decay,
            "weight_decay": 0.0,
            "lr": head_lr,
            "name": "head_no_decay"
        })
    if base_params_decay:
        optimizer_grouped_parameters.append({
            "params": base_params_decay,
            "weight_decay": weight_decay,
            "lr": base_lr,
            "name": "base_decay"
        })
    if base_params_no_decay:
        optimizer_grouped_parameters.append({
            "params": base_params_no_decay,
            "weight_decay": 0.0,
            "lr": base_lr,
            "name": "base_no_decay"
        })

    return optimizer_grouped_parameters