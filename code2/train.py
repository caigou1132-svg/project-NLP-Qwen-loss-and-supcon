"""
方案A：LoRA 微调训练模块（含 SupCon Loss 支持）
=====================================================
核心训练逻辑，支持纯 CE loss 和 CE + λ×SupCon 联合 loss。
与数据模块解耦：只接收 DataLoader，不关心数据来源。
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score

from code2.config import LoRAExperimentConfig, RESULTS_DIR, resolve_model_path
from code2.supcon_loss import SupConLoss, pool_hidden_states


def build_model_and_peft(model_id: str, config: LoRAExperimentConfig, device: torch.device):
    """
    加载预训练模型 + 注入 LoRA。

    注意：启用 SupCon 时需要模型输出 hidden_states。
    """
    print(f"  [train] 加载模型: {model_id}")

    use_bf16_safe = config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16_safe else torch.float32
    model_kwargs = {
        "num_labels": config.num_classes,
        "trust_remote_code": True,
        "dtype": model_dtype,
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        resolve_model_path(model_id), **model_kwargs,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )

    model = get_peft_model(model, peft_config)
    model = model.to(device)
    model.print_trainable_parameters()

    return model, peft_config


def train_one_epoch(
    model, train_loader: DataLoader, optimizer, scheduler,
    device: torch.device, supcon_criterion: SupConLoss = None,
    supcon_lambda: float = 0.0, use_bf16: bool = False,
    pooling_strategy: str = "mean",
) -> dict:
    """
    训练一个 epoch。

    当 supcon_lambda > 0 时：loss = CE_loss + λ * SupCon_loss
    当 supcon_lambda = 0 时：loss = CE_loss（退化为标准 LoRA 微调）
    """
    model.train()
    total_ce_loss = 0.0
    total_supcon_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播；SupCon 时需要 hidden_states
        extra_kwargs = {}
        if supcon_lambda > 0:
            extra_kwargs["output_hidden_states"] = True
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **extra_kwargs)
        ce_loss = outputs.loss
        total_loss = ce_loss

        # === SupCon Loss ===
        # 转 fp32 计算：bf16 下 exp/log 易 NaN（temperature=0.1 放大数值风险）
        if supcon_lambda > 0 and supcon_criterion is not None:
            hidden_states = outputs.hidden_states
            features = pool_hidden_states(hidden_states, attention_mask, strategy=pooling_strategy).float()
            supcon_loss = supcon_criterion(features, labels)
            total_loss = ce_loss + supcon_lambda * supcon_loss
            total_supcon_loss += supcon_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        total_ce_loss += ce_loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_ce_loss = total_ce_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)

    metrics = {"ce_loss": avg_ce_loss, "accuracy": acc}
    if supcon_lambda > 0:
        metrics["supcon_loss"] = total_supcon_loss / len(train_loader)
        metrics["total_loss"] = metrics["ce_loss"] + supcon_lambda * metrics["supcon_loss"]

    return metrics


@torch.no_grad()
def evaluate(model, val_loader: DataLoader, device: torch.device, use_bf16: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(val_loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return {"loss": avg_loss, "accuracy": acc, "f1_macro": f1}


def run_single_experiment(
    model_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LoRAExperimentConfig,
    device: torch.device,
    few_shot_size: int = -1,
    seed: int = 42,
) -> dict:
    # 设置随机种子（多种子实验的核心）
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    experiment_name = model_id.replace("/", "_")
    supcon_label = f"_supcon{config.supcon_lambda}" if config.supcon_lambda > 0 else "_CEonly"
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}{supcon_label} (seed={seed})")
    print(f"{'='*60}")

    model, peft_config = build_model_and_peft(model_id, config, device)

    # bf16 安全标志：与 build_model_and_peft 的 dtype 判断保持一致
    use_bf16_safe = config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # SupCon
    supcon_criterion = None
    if config.supcon_lambda > 0:
        supcon_criterion = SupConLoss(temperature=config.supcon_temperature)

    best_val_acc = 0.0
    best_val_metrics = None
    history = []

    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            supcon_criterion=supcon_criterion,
            supcon_lambda=config.supcon_lambda,
            use_bf16=use_bf16_safe,
            pooling_strategy=getattr(config, "pooling_strategy", "mean"),
        )
        val_metrics = evaluate(model, val_loader, device, use_bf16=use_bf16_safe)

        train_str = f"CE Loss: {train_metrics['ce_loss']:.4f}"
        if 'supcon_loss' in train_metrics:
            train_str += f", SupCon Loss: {train_metrics['supcon_loss']:.4f}"
        train_str += f", Acc: {train_metrics['accuracy']:.4f}"
        print(f"  Train - {train_str}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")

        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_metrics = {k: v for k, v in val_metrics.items()}

    del model
    torch.cuda.empty_cache()

    result = {
        "model_id": model_id,
        "few_shot_size": few_shot_size,
        "lora_r": config.lora_r,
        "supcon_lambda": config.supcon_lambda,
        "supcon_temperature": config.supcon_temperature,
        "pooling_strategy": getattr(config, "pooling_strategy", "mean"),
        "seed": seed,
        "best_val_accuracy": best_val_acc,
        "best_val_metrics": best_val_metrics if best_val_metrics else val_metrics,
        "history": history,
        "lora_config": {"r": config.lora_r, "alpha": config.lora_alpha, "dropout": config.lora_dropout},
    }

    return result


def save_experiment_result(result: dict, config: LoRAExperimentConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    safe_name = result["model_id"].replace("/", "_")
    supcon_tag = f"_supcon{result['supcon_lambda']}" if result['supcon_lambda'] > 0 else "_CEonly"
    fs = result.get("few_shot_size", "full")
    fs_tag = f"_fs{fs}"
    r_val = result.get("lora_r", config.lora_r)
    r_tag = f"_r{r_val}"
    seed = result.get("seed", 42)
    seed_tag = f"_seed{seed}"
    pool_strategy = result.get("pooling_strategy", "mean")
    pool_tag = f"_{pool_strategy}" if pool_strategy != "mean" else ""
    path = os.path.join(config.output_dir, f"result_{safe_name}{fs_tag}{r_tag}{supcon_tag}{seed_tag}{pool_tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[train] 结果已保存: {path}")
