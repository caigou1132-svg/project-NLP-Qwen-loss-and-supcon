"""
v2 核心验证实验：增大 batch size + label-aware sampling + pooling 消融
========================================================================
聚焦 1.5B r=16，fs=500，3种子。CE-only 统一跑（不受 pooling 影响），
SupCon 做 mean vs last_token pooling 消融。

实验矩阵（9 组）：
- CE baseline：3 种子（pooling=N/A）
- SupCon λ=1.0 + mean pooling：3 种子
- SupCon λ=1.0 + last_token pooling：3 种子

用法（服务器）：
  rm -f project-a-llm-lora/results/result_*.json
  PYTHONPATH="/root/autodl-tmp/lunwen/project-a-llm-lora" \
    nohup python -m code2.run_v2 > experiment_v2.log 2>&1 &
"""
import sys
import os
import torch

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_PATH = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _CODE_PATH)

from code2.config import LoRAExperimentConfig
from code2.data_loader import load_and_prepare_dataset, create_dataloaders
from code2.train import run_single_experiment, save_experiment_result
from code2.evaluate import (
    load_all_results, print_comparison_table,
    plot_lambda_sensitivity, plot_supcon_vs_baseline, plot_pooling_ablation,
)


def _result_exists(config, model_id, few_shot_size, supcon_lambda, seed=42):
    safe_name = model_id.replace("/", "_")
    supcon_tag = f"_supcon{supcon_lambda}" if supcon_lambda > 0 else "_CEonly"
    fs_tag = f"_fs{few_shot_size}"
    r_tag = f"_r{config.lora_r}"
    seed_tag = f"_seed{seed}"
    pool_strategy = getattr(config, "pooling_strategy", "mean")
    pool_tag = f"_{pool_strategy}" if pool_strategy != "mean" else ""
    path = os.path.join(config.output_dir, f"result_{safe_name}{fs_tag}{r_tag}{supcon_tag}{seed_tag}{pool_tag}.json")
    return os.path.exists(path)


def _run_seed_experiment(config, data, seed, label):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"\n  [{label}] fs=500 | seed={seed}")
    train_loader, val_loader = create_dataloaders(
        data["train"], data["val"], config, few_shot_size=500, seed=seed
    )
    result = run_single_experiment(
        model_id=config.models_to_compare[0], train_loader=train_loader,
        val_loader=val_loader, config=config, device=device,
        few_shot_size=500, seed=seed,
    )
    save_experiment_result(result, config)


def main():
    config = LoRAExperimentConfig()

    # ==== v2 核心设定 ====
    config.models_to_compare = ["Qwen/Qwen2.5-1.5B"]
    config.few_shot_sizes = [500]
    config.lora_r = 16
    config.lora_alpha = 32
    config.batch_size = 28
    config.label_aware_batch = True

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_v2] 设备: {device}")
    if device.type == "cuda":
        print(f"[run_v2] GPU: {torch.cuda.get_device_name(0)}, "
              f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[run_v2] batch_size={config.batch_size}, label_aware={config.label_aware_batch}, r={config.lora_r}")

    data = load_and_prepare_dataset(config)
    MULTI_SEEDS = [42, 123, 456]

    # ==== 阶段 1：CE baseline（λ=0，pooling 不影响 CE，只跑 3 种子）====
    print("\n" + "=" * 70)
    print("  阶段 1：CE baseline（pooling=N/A）| 1.5B r=16 | 3 seeds")
    print("=" * 70)

    config.supcon_lambda = 0.0
    config.pooling_strategy = "mean"
    for seed in MULTI_SEEDS:
        if _result_exists(config, config.models_to_compare[0], 500, 0.0, seed):
            print(f"  [跳过] CE baseline seed={seed}")
            continue
        _run_seed_experiment(config, data, seed, "CE baseline")

    # ==== 阶段 2：SupCon λ=1.0 + mean pooling ====
    print("\n" + "=" * 70)
    print("  阶段 2：SupCon λ=1.0 + mean pooling | 1.5B r=16 | 3 seeds")
    print("=" * 70)

    config.supcon_lambda = 1.0
    config.pooling_strategy = "mean"
    for seed in MULTI_SEEDS:
        if _result_exists(config, config.models_to_compare[0], 500, 1.0, seed):
            print(f"  [跳过] SupCon mean seed={seed}")
            continue
        _run_seed_experiment(config, data, seed, "SupCon mean")

    # ==== 阶段 3：SupCon λ=1.0 + last_token pooling ====
    print("\n" + "=" * 70)
    print("  阶段 3：SupCon λ=1.0 + last_token pooling | 1.5B r=16 | 3 seeds")
    print("=" * 70)

    config.pooling_strategy = "last_token"
    for seed in MULTI_SEEDS:
        if _result_exists(config, config.models_to_compare[0], 500, 1.0, seed):
            print(f"  [跳过] SupCon last_token seed={seed}")
            continue
        _run_seed_experiment(config, data, seed, "SupCon last_token")

    # ==== 汇总 ====
    print("\n\n" + "=" * 70)
    print("  v2 全部实验完成！")
    print("=" * 70)

    df = load_all_results(config.output_dir)
    print_comparison_table(df)
    plot_lambda_sensitivity(df)
    plot_supcon_vs_baseline(df)
    plot_pooling_ablation(df)

    print(f"\n结果目录: {config.output_dir}")
    print(f"产出图表: lambda_sensitivity.png, supcon_vs_baseline.png, pooling_ablation.png")


if __name__ == "__main__":
    main()
