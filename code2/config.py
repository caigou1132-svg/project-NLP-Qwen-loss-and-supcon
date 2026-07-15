"""
方案A：LLM LoRA 微调对比实验 — 全局配置
=============================================
所有可调参数集中在这里，其他模块从此导入，避免散落各处。
"""

import os
from dataclasses import dataclass, field
from typing import List

# 国内环境（AutoDL等）访问 HuggingFace 需使用镜像站。
# 必须在任何 transformers/huggingface_hub 导入之前设置。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============================================
# 项目根目录 — 自动检测，无需手动修改
# ============================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)          # d:\work\lunwen
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOCAL_MODELS_DIR = os.path.join(_WORKSPACE_ROOT, "models")  # 本地模型存放目录


def resolve_model_path(model_id: str) -> str:
    """如果本地存在对应模型目录，返回本地路径；否则返回原始 model_id。"""
    local_name = model_id.replace("/", "_")
    local_path = os.path.join(LOCAL_MODELS_DIR, local_name)
    if os.path.isdir(local_path):
        return local_path
    return model_id


@dataclass
class LoRAExperimentConfig:
    """
    实验配置。

    你可以修改以下任何参数。建议先用默认值跑通，
    确认流程没问题后再调整。

    重要约束：
    - 本实验在 CPU 上极其缓慢。强烈建议使用 CUDA GPU。
    - 如果显存不足，优先减小 batch_size 或换更小的模型。
    """

    # ---- 数据集 ----
    # 使用本地 THUCNews（清华大学新闻分类，14类）
    # 数据路径: data/THUCNews/类别/*.txt
    dataset_name: str = "THUCNews"        # 本地数据集名
    dataset_config: str = ""              # 不使用 HuggingFace
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "THUCNews")
    max_seq_length: int = 256
    num_classes: int = 14                 # THUCNews 14 类

    # ---- 训练参数 ----
    num_epochs: int = 3                  # 训练轮数（LoRA 通常 3-5 轮就收敛）
    batch_size: int = 28                 # 增大至 28：14类每 batch 各类期望 2 样本，SupCon 正样本 mask 生效前提
    learning_rate: float = 2e-4          # 学习率（LoRA 常用 1e-4 ~ 5e-4）
    warmup_ratio: float = 0.1            # warmup 步数占比
    weight_decay: float = 0.01
    use_bf16: bool = True                # bf16 混合精度：加速约 50%，显存省约 40%
    stratified_few_shot: bool = True     # 少样本分层采样：每类等量，避免类不均衡假象
    label_aware_batch: bool = True       # label-aware batch 构造：确保每 batch 各类至少 2 样本，不依赖随机 shuffle

    # ---- LoRA 参数 ----
    lora_r: int = 16                     # LoRA rank（v2 聚焦 r=16，最优配置）
    lora_alpha: int = 32                 # LoRA alpha（2×r）
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj"              # 对 Qwen2/ChatGLM 系列有效
    ])

    # LoRA rank 消融实验：固定 best_λ，测不同 r 的表现
    lora_ranks: List[int] = field(default_factory=lambda: [4, 8, 16])
    # alpha 自动设为 2*r，保持一致

    # ---- 模型列表 ----
    # 你要对比的模型。每个模型大约需要 2-7GB 显存（q4 量化版本）。
    # 注意：模型 ID 来自 HuggingFace Hub，需要网络连接。
    # 如果下载失败，请检查网络或使用镜像（hf-mirror.com）。
    models_to_compare: List[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-0.5B",            # 最小，适合快速验证
        "Qwen/Qwen2.5-1.5B",            # 中等
    ])

    # 如果你想加更多模型，取消下面的注释：
    # "google/gemma-2-2b-it",
    # "meta-llama/Llama-3.2-1B",
    # "THUDM/chatglm3-6b",

    # ---- SupCon（监督对比损失）参数 ----
    supcon_lambda: float = 0.0            # SupCon loss 的权重。0 = 不启用
    supcon_temperature: float = 0.1       # 温度系数（越小越"硬"，通常 0.05~0.5）
    pooling_strategy: str = "mean"        # hidden state 池化策略："mean" | "last_token"
    # "mean": attention_mask 加权均值（当前 baseline）
    # "last_token": 取每个样本最后一个有效 token 的 hidden state（causal decoder 下理论上更合理）

    # ---- 对比实验设计 ----
    # 对比维度：
    #   1. 不同模型（models_to_compare）
    #   2. 不同训练样本数：full / 500 / 100（少样本场景）
    #   3. 不同 λ 值（消融实验，见 supcon_lambdas）
    few_shot_sizes: List[int] = field(default_factory=lambda: [100, 500, -1])
    # -1 表示使用全部训练数据

    # 消融实验用的 λ 列表（你的论文的 ablation study）
    supcon_lambdas: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0])
    # 0.0 = 纯 CE baseline

    # ---- 输出 ----
    output_dir: str = RESULTS_DIR

    # ---- 随机种子 ----
    seed: int = 42


# ============================================
# 如果你没有 GPU，可以将 device 强制设为 "cpu"，
# 但训练会慢到不可接受。建议用 Google Colab 免费 GPU。
# ============================================
DEFAULT_CONFIG = LoRAExperimentConfig()
