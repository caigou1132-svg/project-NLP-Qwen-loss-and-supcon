"""
监督对比损失（Supervised Contrastive Loss）+ 池化策略
=======================================================
基于 code2，新增 "cls_token" pooling 支持 encoder-only 模型（BERT 等）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """监督对比损失。与 code2 完全一致的实现。"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        diag_mask = 1.0 - torch.eye(batch_size, device=features.device)
        pos_mask = pos_mask * diag_mask

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix) * diag_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_per_sample = pos_mask.sum(dim=1)
        pos_per_sample = torch.clamp(pos_per_sample, min=1e-8)

        loss = -(pos_mask * log_prob).sum(dim=1) / pos_per_sample
        loss = loss.mean()
        return loss


def pool_hidden_states(hidden_states, attention_mask, strategy: str = "mean"):
    """
    对最后一层 hidden states 做池化。

    Args:
        hidden_states: tuple of (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)，1=有效token，0=padding
        strategy: "mean" | "last_token" | "cls_token"
    Returns:
        pooled: (batch, hidden_dim)
    """
    last_hidden = hidden_states[-1]  # (batch, seq_len, hidden_dim)

    if strategy == "cls_token":
        # encoder-only: 取第 0 个 token（[CLS]）的 hidden state
        return last_hidden[:, 0, :]

    if strategy == "last_token":
        # decoder-only: 取每个样本最后一个有效 token
        seq_lens = attention_mask.sum(dim=1).long() - 1
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_indices, seq_lens]
        return pooled

    # default: "mean" — attention_mask 加权均值池化
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled
