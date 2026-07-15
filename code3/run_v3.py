"""
v3 核心验证实验：epoch 补偿 + decoder/encoder 跨架构对比
=============================================================
实验矩阵（25 组）：
  阶段 1：decoder（Qwen2.5-1.5B, batch=28, label_aware, epoch=11）
    CE baseline：5 种子
    SupCon λ=1.0 + mean pooling：5 种子
    SupCon λ=1.0 + last_token pooling：5 种子

  阶段 2：encoder（bert-base-chinese, batch=28, shuffle, epoch=11）
    CE baseline：5 种子
    SupCon λ=1.0 + cls_token pooling：5 种子

用法（服务器）：
  rm -f project-a-llm-lora/results/result_*.json
  PYTHONPATH="/root/autodl-tmp/lunwen/project-a-llm-lora" \
    nohup python -m code3.run_v3 > experiment_v3.log 2>&1 &
"""
import sys
import os
import math
import json
import torch

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_PATH = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _CODE_PATH)

from code3.config import (
    LoRAExperimentConfig, MULTI_SEEDS,
    get_architecture, get_target_modules, get_default_pooling,
)
from code3.data_loader import load_and_prepare_dataset, create_dataloaders
from code3.train import run_single_experiment, save_experiment_result
from code3.evaluate import (
    load_all_results, print_comparison_table,
    plot_cross_architecture, plot_pooling_ablation, print_statistical_note,
)


def _result_exists(config, model_id, few_shot_size, supcon_lambda, seed=42, pooling_override=None):
    safe_name = model_id.replace("/", "_")
    supcon_tag = f"_supcon{supcon_lambda}" if supcon_lambda > 0 else "_CEonly"
    fs_tag = f"_fs{few_shot_size}"
    r_tag = f"_r{config.lora_r}"
    seed_tag = f"_seed{seed}"
    arch = get_architecture(model_id)
    pool_strategy = pooling_override or getattr(config, "pooling_strategy", "mean")
    default_pool = "cls_token" if arch == "encoder" else "mean"
    pool_tag = f"_{pool_strategy}" if pool_strategy != default_pool else ""
    path = os.path.join(config.output_dir, f"result_{safe_name}{fs_tag}{r_tag}{supcon_tag}{seed_tag}{pool_tag}.json")
    if not os.path.exists(path):
        return False
    # 完整性校验：防止写入中途被 kill 产生的半截 JSON 被永久跳过
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("best_val_accuracy") is None:
            print(f"  [警告] 结果文件不完整，将重跑: {os.path.basename(path)}")
            return False
        return True
    except (json.JSONDecodeError, KeyError):
        print(f"  [警告] 结果文件损坏，将重跑: {os.path.basename(path)}")
        return False


def _run_seed_experiment(config, data, seed, label):
    # 种子由 run_single_experiment 统一设置（torch + numpy + random + cuda）
    model_id = config.models_to_compare[0]
    print(f"\n  [{label}] fs=500 | seed={seed}")
    train_loader, val_loader = create_dataloaders(
        data["train"], data["val"], config, few_shot_size=500, seed=seed
    )
    result = run_single_experiment(
        model_id=model_id, train_loader=train_loader,
        val_loader=val_loader, config=config, device=device,
        few_shot_size=500, seed=seed,
    )
    save_experiment_result(result, config)


def main():
    config = LoRAExperimentConfig()

    # ==== v3 核心设定 ====
    config.few_shot_sizes = [500]
    config.lora_r = 16
    config.lora_alpha = 32
    config.batch_size = 28
    config.num_epochs = 11            # batch=28 下 ≈198 steps，对齐 v1 的 189 steps

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_v3] 设备: {device}")
    if device.type == "cuda":
        print(f"[run_v3] GPU: {torch.cuda.get_device_name(0)}, "
              f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[run_v3] batch_size={config.batch_size}, num_epochs={config.num_epochs}, "
          f"total_steps/epoch≈{math.ceil(500 / config.batch_size)}, "
          f"total_steps≈{math.ceil(500 / config.batch_size) * config.num_epochs}")

    # ================================================================
    # 阶段 1：Decoder（Qwen2.5-1.5B）+ label_aware batch
    # ================================================================
    DECODER_MODEL = "Qwen/Qwen2.5-1.5B"
    config.models_to_compare = [DECODER_MODEL]
    config.label_aware_batch = True
    config.target_modules = get_target_modules(DECODER_MODEL)

    print("\n" + "=" * 70)
    print(f"  阶段 1：Decoder | {DECODER_MODEL} | batch=28 label_aware | epoch=11")
    print("=" * 70)

    data_decoder = load_and_prepare_dataset(config, model_id=DECODER_MODEL)

    # 1a: CE baseline
    print("\n--- 1a: CE baseline | 5 seeds ---")
    config.supcon_lambda = 0.0
    config.pooling_strategy = "mean"
    for seed in MULTI_SEEDS:
        _run_seed_experiment(config, data_decoder, seed, "CE baseline")

    # 1b: SupCon λ=1.0 + mean pooling
    print("\n--- 1b: SupCon λ=1.0 + mean pooling | 5 seeds ---")
    config.supcon_lambda = 1.0
    config.pooling_strategy = "mean"
    for seed in MULTI_SEEDS:
        _run_seed_experiment(config, data_decoder, seed, "SupCon mean")

    # 1c: SupCon λ=1.0 + last_token pooling
    print("\n--- 1c: SupCon λ=1.0 + last_token pooling | 5 seeds ---")
    config.pooling_strategy = "last_token"
    for seed in MULTI_SEEDS:
        _run_seed_experiment(config, data_decoder, seed, "SupCon last_token")

    # ================================================================
    # 阶段 2：Encoder（bert-base-chinese）+ 标准 shuffle
    # ================================================================
    ENCODER_MODEL = "google-bert/bert-base-chinese"
    config.models_to_compare = [ENCODER_MODEL]
    config.label_aware_batch = False    # encoder 不需要 label_aware
    config.target_modules = get_target_modules(ENCODER_MODEL)

    print("\n" + "=" * 70)
    print(f"  阶段 2：Encoder | {ENCODER_MODEL} | batch=28 shuffle | epoch=11")
    print("=" * 70)

    data_encoder = load_and_prepare_dataset(config, model_id=ENCODER_MODEL)

    # 2a: CE baseline（cls_token pooling，BERT 默认行为）
    print("\n--- 2a: CE baseline | 5 seeds ---")
    config.supcon_lambda = 0.0
    config.pooling_strategy = "cls_token"
    for seed in MULTI_SEEDS:
        _run_seed_experiment(config, data_encoder, seed, "CE baseline (BERT)")

    # 2b: SupCon λ=1.0 + cls_token pooling
    print("\n--- 2b: SupCon λ=1.0 + cls_token pooling | 5 seeds ---")
    config.supcon_lambda = 1.0
    config.pooling_strategy = "cls_token"
    for seed in MULTI_SEEDS:
        _run_seed_experiment(config, data_encoder, seed, "SupCon cls_token (BERT)")

    # ================================================================
    # 汇总与可视化
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  v3 全部实验完成！")
    print("=" * 70)

    df = load_all_results(config.output_dir)
    print_comparison_table(df)
    print_statistical_note(df)
    plot_cross_architecture(df)
    plot_pooling_ablation(df)

    print(f"\n结果目录: {config.output_dir}")
    print("产出图表: cross_architecture.png, pooling_ablation.png")


if __name__ == "__main__":
    main()
