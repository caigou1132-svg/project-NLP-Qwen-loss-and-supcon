"""
v3 数据加载模块：支持 decoder + encoder 双架构
=====================================================
基于 code2/data_loader.py，新增：
  - 按 model_id 自动选择正确的 tokenizer
  - encoder 模型不需要 label_aware batch（bidirectional attention 无 causal 限制）
"""

import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from transformers import AutoTokenizer
from tqdm import tqdm

from code3.config import resolve_model_path, get_architecture


class LabelAwareBatchSampler(Sampler):
    """Label-aware batch 采样器。与 code2 完全一致的实现。"""

    def __init__(self, labels, batch_size, min_per_class=2, seed=42):
        self.labels = np.asarray(labels, dtype=int)
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.n_classes = self.labels.max() + 1
        self.rng = np.random.RandomState(seed)

        self.class_indices = [np.where(self.labels == c)[0] for c in range(self.n_classes)]
        self.per_batch_per_class = min_per_class
        self.fixed_per_batch = self.n_classes * self.per_batch_per_class
        if self.fixed_per_batch > batch_size:
            raise ValueError(f"batch_size({batch_size}) 不足以容纳 "
                             f"{self.n_classes}×{min_per_class}={self.fixed_per_batch} 个固定样本")
        self.remainder_per_batch = batch_size - self.fixed_per_batch

    def __iter__(self):
        shuffled = [self.rng.permutation(idxs).tolist() for idxs in self.class_indices]
        min_avail = min(len(idxs) // self.per_batch_per_class for idxs in shuffled)
        n_batches = max(1, min_avail)

        all_remaining = []
        for c in range(self.n_classes):
            used = n_batches * self.per_batch_per_class
            all_remaining.extend(shuffled[c][used:])
        self.rng.shuffle(all_remaining)
        rem_ptr = 0

        for batch_idx in range(n_batches):
            batch = []
            for c in range(self.n_classes):
                start = batch_idx * self.per_batch_per_class
                batch.extend(shuffled[c][start:start + self.per_batch_per_class])
            need = self.remainder_per_batch
            if rem_ptr + need <= len(all_remaining):
                batch.extend(all_remaining[rem_ptr:rem_ptr + need])
                rem_ptr += need
            elif rem_ptr < len(all_remaining):
                batch.extend(all_remaining[rem_ptr:])
                rem_ptr = len(all_remaining)
            self.rng.shuffle(batch)
            yield batch

        if rem_ptr < len(all_remaining):
            tail = all_remaining[rem_ptr:]
            self.rng.shuffle(tail)
            if tail:
                yield tail

    def __len__(self):
        min_avail = min(len(idxs) // self.per_batch_per_class for idxs in self.class_indices)
        n_full = max(1, min_avail)
        total_samples = sum(len(idxs) for idxs in self.class_indices)
        used_fixed = n_full * self.fixed_per_batch
        has_tail = total_samples > used_fixed
        return n_full + (1 if has_tail else 0)


class THUCNewsDataset(Dataset):
    def __init__(self, data_dir, split_ratio=0.8, train=True, seed=42):
        self.data_dir = data_dir
        self.samples = []

        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and not d.startswith(".") and not d.startswith("__")
        ])
        self.class_names = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        all_samples = []
        for cls_name in classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".txt"):
                    all_samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

        random.seed(seed)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * split_ratio)
        self.samples = all_samples[:split_idx] if train else all_samples[split_idx:]
        print(f"  [THUCNews] {'train' if train else 'val'}: {len(self.samples)} 样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="gbk") as f:
                text = f.read()
        return text, label

    def get_labels(self):
        return [label for _, label in self.samples]


def load_and_prepare_dataset(config, model_id: str = None):
    """
    加载 THUCNews 数据 + 构建 tokenized dataset。

    Args:
        config: LoRAExperimentConfig
        model_id: 用于加载 tokenizer 的模型 ID，默认取 config.models_to_compare[0]
    Returns:
        dict with keys: train, val, tokenizer, classes, architecture
    """
    if model_id is None:
        model_id = config.models_to_compare[0]

    architecture = get_architecture(model_id)
    print(f"[data_loader] 架构: {architecture} | 从本地加载: {config.data_dir}")

    raw_train = THUCNewsDataset(config.data_dir, train=True)
    raw_val = THUCNewsDataset(config.data_dir, train=False)

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(model_id), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class TokenizedDataset(Dataset):
        def __init__(self, raw_ds, tokenizer, max_len, pre_tokenize=False):
            self.max_len = max_len
            self.pre_tokenize = pre_tokenize
            if pre_tokenize:
                self._cached = []
                for idx in tqdm(range(len(raw_ds)), desc="  预Tokenize(Val)", leave=False):
                    text, label = raw_ds[idx]
                    encoded = tokenizer(
                        text, truncation=True, padding="max_length", max_length=max_len,
                        return_tensors="pt",
                    )
                    self._cached.append({
                        "input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                        "labels": torch.tensor(label, dtype=torch.long),
                    })
            else:
                self.raw = raw_ds
                self.tokenizer = tokenizer

        def __len__(self):
            if self.pre_tokenize:
                return len(self._cached)
            return len(self.raw)

        def __getitem__(self, idx):
            if self.pre_tokenize:
                return self._cached[idx]
            text, label = self.raw[idx]
            encoded = self.tokenizer(
                text, truncation=True, padding="max_length", max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
            }

        def get_labels(self):
            if self.pre_tokenize:
                return [item["labels"].item() for item in self._cached]
            return self.raw.get_labels()

    train_set = TokenizedDataset(raw_train, tokenizer, config.max_seq_length, pre_tokenize=False)
    val_set = TokenizedDataset(raw_val, tokenizer, config.max_seq_length, pre_tokenize=True)

    print(f"[data_loader] 类别: {raw_train.class_names}")
    print(f"[data_loader] Tokenization 完成")
    return {
        "train": train_set, "val": val_set, "tokenizer": tokenizer,
        "classes": raw_train.class_names, "architecture": architecture,
    }


def create_dataloaders(train_dataset, val_dataset, config, few_shot_size=-1, seed=42):
    if few_shot_size > 0:
        g = torch.Generator()
        g.manual_seed(seed + few_shot_size)

        if getattr(config, "stratified_few_shot", False) and hasattr(train_dataset, "get_labels"):
            labels = train_dataset.get_labels()
            n_classes = config.num_classes
            per_class = few_shot_size // n_classes
            remainder = few_shot_size - per_class * n_classes
            indices = []
            for cls in range(n_classes):
                cls_indices = [i for i, l in enumerate(labels) if l == cls]
                n_sample = per_class + (1 if cls < remainder else 0)
                if len(cls_indices) < n_sample:
                    print(f"  [data_loader] 警告: 类别 {cls} 仅 {len(cls_indices)} 样本, "
                          f"需要 {n_sample}, 将取全部可用样本")
                    n_sample = len(cls_indices)
                cls_sample = torch.randperm(len(cls_indices), generator=g)[:n_sample].tolist()
                indices.extend([cls_indices[i] for i in cls_sample])
        else:
            indices = torch.randperm(len(train_dataset), generator=g)[:few_shot_size].tolist()

        train_subset = Subset(train_dataset, indices)
    else:
        train_subset = train_dataset

    use_label_aware = getattr(config, "label_aware_batch", False) and few_shot_size > 0
    if use_label_aware:
        if isinstance(train_subset, Subset):
            raw_labels = train_dataset.get_labels()
            subset_labels = [raw_labels[i] for i in train_subset.indices]
        else:
            subset_labels = train_subset.get_labels()
        batch_sampler = LabelAwareBatchSampler(
            subset_labels, config.batch_size, min_per_class=2, seed=seed
        )
        train_loader = DataLoader(
            train_subset, batch_sampler=batch_sampler,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_subset, batch_size=config.batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
