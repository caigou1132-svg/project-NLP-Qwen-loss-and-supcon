"""
监督对比损失（Supervised Contrastive Loss）实现
=================================================
基于 Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
论文链接: https://arxiv.org/abs/2004.11362

核心思想：
  不是只用交叉熵让模型输出正确的类别概率，
  而是额外要求：同类样本在 embedding 空间里聚在一起，异类推远。

公式（简化）：
  L_supcon(i) = -1/|P(i)| Σ log[ exp(z_i·z_p / τ) / Σ exp(z_i·z_a / τ) ]
  其中 P(i) = 与 i 同类的样本集合，τ = 温度系数。

最终 loss = CE_loss + lambda * SupCon_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    监督对比损失。

    使用方式：
        criterion = SupConLoss(temperature=0.1)
        loss = criterion(features, labels)   # features: (batch, dim), labels: (batch,)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalized embeddings, shape (batch_size, feature_dim)
            labels:   ground-truth labels, shape (batch_size,)
        Returns:
            scalar loss
        """
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)

        # ---- 1. L2 归一化 ----
        features = F.normalize(features, dim=1)

        # ---- 2. 相似度矩阵（除以 temperature）----
        #    sim_matrix[i][j] = z_i · z_j / τ
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # ---- 3. 构建正样本 mask ----
        #    正样本 = 同标签且不是自身
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()        # 同类的样本
        # 去掉对角线（自己和自己的相似度不算正样本）
        diag_mask = 1.0 - torch.eye(batch_size, device=features.device)
        pos_mask = pos_mask * diag_mask

        # ---- 4. 数值稳定性：减去每行最大值 ----
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # ---- 5. 计算 log-softmax ----
        #    分母 = Σ_{a∈A(i)} exp(sim_matrix[i][a])
        exp_sim = torch.exp(sim_matrix) * diag_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # ---- 6. 只取正样本对的 log-prob，求平均 ----
        pos_per_sample = pos_mask.sum(dim=1)  # 每个样本有几个同类
        pos_per_sample = torch.clamp(pos_per_sample, min=1e-8)

        loss = -(pos_mask * log_prob).sum(dim=1) / pos_per_sample
        loss = loss.mean()

        return loss


def pool_hidden_states(hidden_states, attention_mask, strategy: str = "mean"):
    """
    对最后一层 hidden states 做池化，排除 padding token。

    Qwen2 等 decoder-only 模型没有传统 [CLS] token。

    Args:
        hidden_states: tuple of (batch, seq_len, hidden_dim)，取最后一层
        attention_mask: (batch, seq_len)，1=有效token，0=padding
        strategy: "mean" | "last_token"
    Returns:
        pooled: (batch, hidden_dim)
    """
    last_hidden = hidden_states[-1]  # (batch, seq_len, hidden_dim)

    if strategy == "last_token":
        # 取每个样本最后一个有效 token 的 hidden state
        # seq_lens: (batch,)，每个样本的有效 token 数
        seq_lens = attention_mask.sum(dim=1).long() - 1  # 索引从 0 开始
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_indices, seq_lens]  # (batch, hidden_dim)
        return pooled

    # default: mean pooling
    mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled
