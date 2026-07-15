"""
v3 实验配置：epoch 补偿 + decoder/encoder 跨架构对比
========================================================
基于 v2 的全部改进，增加：
  - num_epochs=11（batch=28 下等价 v1 的 189 steps）
  - 5 种子替代 3 种子（增强统计检验力）
  - encoder-only 模型支持（BERT-base 中文）
  - 架构自动检测 + 对应 pooling 策略
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOCAL_MODELS_DIR = os.path.join(_WORKSPACE_ROOT, "models")


def resolve_model_path(model_id: str) -> str:
    local_name = model_id.replace("/", "_")
    local_path = os.path.join(LOCAL_MODELS_DIR, local_name)
    if os.path.isdir(local_path):
        return local_path
    return model_id


# ============================================================
# 架构定义：decoder-only vs encoder-only
# ============================================================
MODEL_ARCHITECTURES: Dict[str, str] = {
    "Qwen/Qwen2.5-0.5B": "decoder",
    "Qwen/Qwen2.5-1.5B": "decoder",
    "google-bert/bert-base-chinese": "encoder",
}


def get_architecture(model_id: str) -> str:
    """根据 model_id 返回架构类型："decoder" | "encoder"。"""
    if model_id in MODEL_ARCHITECTURES:
        return MODEL_ARCHITECTURES[model_id]
    lower = model_id.lower()
    if "bert" in lower or "roberta" in lower or "electra" in lower:
        return "encoder"
    return "decoder"


def get_target_modules(model_id: str, architecture: str = None) -> List[str]:
    """返回对应架构的 LoRA target_modules。"""
    arch = architecture or get_architecture(model_id)
    if arch == "encoder":
        return ["query", "value"]
    return ["q_proj", "v_proj"]


def get_default_pooling(architecture: str) -> str:
    """返回对应架构的默认 pooling 策略。"""
    if architecture == "encoder":
        return "cls_token"
    return "mean"


@dataclass
class LoRAExperimentConfig:
    # ---- 数据集 ----
    dataset_name: str = "THUCNews"
    dataset_config: str = ""
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "THUCNews")
    max_seq_length: int = 256
    num_classes: int = 14

    # ---- 训练参数 ----
    num_epochs: int = 11                 # v2: 3 → v3: 11（batch=28 下 ≈189 steps，对齐 v1）
    batch_size: int = 28
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    use_bf16: bool = True
    stratified_few_shot: bool = True
    label_aware_batch: bool = True

    # ---- LoRA 参数 ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # ---- 模型列表 ----
    models_to_compare: List[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-1.5B",
        "google-bert/bert-base-chinese",
    ])

    # ---- SupCon 参数 ----
    supcon_lambda: float = 0.0
    supcon_temperature: float = 0.1
    pooling_strategy: str = "mean"

    # ---- 对比实验设计 ----
    few_shot_sizes: List[int] = field(default_factory=lambda: [500])

    # ---- 输出 ----
    output_dir: str = RESULTS_DIR

    # ---- 随机种子 ----
    seed: int = 42


# ============================================================
# v3 实验管理：multi-seed 种子列表
# ============================================================
MULTI_SEEDS = [42, 123, 456, 789, 1011]  # 5 seeds for statistical power

DEFAULT_CONFIG = LoRAExperimentConfig()
