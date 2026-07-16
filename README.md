# LoRA + SupCon：少样本文本分类的跨架构效应分析

> **A Systematic Empirical Study on Supervised Contrastive Learning for Few-Shot Text Classification across Decoder-Only and Encoder-Only Architectures**
>
> 最后更新：2026-07-15

***

## 项目简介

本项目系统研究\*\*监督对比损失（Supervised Contrastive Loss, SupCon）\*\*在 LoRA 微调框架下对两类预训练语言模型的增益规律：

| 架构           | 代表模型              | 序列建模方式                  | SupCon pooling 策略 |
| ------------ | ----------------- | ----------------------- | ----------------- |
| Decoder-only | Qwen2.5-1.5B      | Causal Attention        | Mean / Last-Token |
| Encoder-only | BERT-base-chinese | Bidirectional Attention | \[CLS] Token      |

在 THUCNews 14 分类数据集上，控制完全相同的实验条件（数据划分、超参数、LoRA rank），系统消融以下维度：

- **架构对比**：decoder vs encoder
- **Pooling 策略**：mean / last-token / \[CLS] token
- **Batch 构造**：标准随机 shuffle vs label-aware batch sampling
- **多种子验证**：5 个随机种子 + Cohen's d 效应量

核心理论框架为**双重稀释假说**（Dual Dilution Hypothesis）：decoder-only 模型在 SupCon 场景下同时面临 causal mask 导致的 embedding 信息稀释和 batch 内正样本不足导致的对比信号退化，而 encoder-only 模型仅面临后者。

## 项目结构

```
├── project-a-llm-lora/
│   ├── code/          # v1: batch=8 基线实验（32组）
│   ├── code2/         # v2: batch=28 + label-aware（9组，已废弃）
│   └── code3/         # v3: epoch补偿 + 跨架构对比（25组，当前版本）
├── docs/
│   └── 08-experiment-log.md   # 完整实验日志与结果分析
└── README.md
```

## 实验状态

| 版本 | 实验数 | 状态      | 说明                        |
| -- | --- | ------- | ------------------------- |
| v1 | 32  | 已完成     | batch=8，seed 方差主导结果       |
| v2 | 9   | 已完成（无效） | batch=28 epoch=3 导致欠拟合    |
| v3 | 25  | 代码就绪    | batch=28 epoch=11 + 跨架构对比 |

## 依赖

- Python 3.10+
- PyTorch 2.5+
- Transformers, PEFT
- scikit-learn, pandas, matplotlib
- CUDA GPU（推荐 RTX 3090 24GB+）

## 引用

若本项目对您的研究有帮助，请引用：

```bibtex
@misc{lora-supcon-crossarch-2026,
  author = {杨新盛},
  title  = {LoRA + SupCon: A Systematic Empirical Study on Supervised 
            Contrastive Learning for Few-Shot Text Classification 
            across Decoder-Only and Encoder-Only Architectures},
  year   = {2026},
  note   = {https://github.com/caigou1132-SVG/project-NLP-Qwen}
}
```

## 许可与商用条例

**代码许可**：本项目代码采用 [MIT License](https://opensource.org/licenses/MIT) 开源，允许自由使用、修改、分发，包括商业用途，但需保留原始版权声明和许可声明。

**实验数据与结果**：本项目使用的 THUCNews 数据集版权归清华大学自然语言处理实验室所有，仅限学术研究使用。实验结果数据（`results/` 目录下的 JSON 及图表）采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可——允许非商业用途的共享与改编，需署名。

**商用咨询**：如计划将本项目方法或代码用于商业产品，请联系作者获取授权。

**免责声明**：本项目为学术实证研究，不保证方法在任意数据集上的泛化性能。
