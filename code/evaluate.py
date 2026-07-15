"""
方案A：评估与对比模块（含 SupCon 消融实验可视化）
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from code.config import RESULTS_DIR

try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def load_all_results(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    if not os.path.exists(results_dir):
        print(f"[evaluate] 结果目录不存在: {results_dir}")
        return pd.DataFrame()

    for fname in os.listdir(results_dir):
        if not fname.startswith("result_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "model_id": data.get("model_id", fname),
            "few_shot_size": data.get("few_shot_size", -1),
            "lora_r": data.get("lora_r", 8),
            "supcon_lambda": data.get("supcon_lambda", 0.0),
            "seed": data.get("seed", 42),
            "best_val_accuracy": data.get("best_val_accuracy", None),
            "best_f1_macro": data.get("best_val_metrics", {}).get("f1_macro", None),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("best_val_accuracy", ascending=False)
    return df


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    对多种子实验按 (model_id, few_shot_size, lora_r, supcon_lambda) 聚合，
    返回均值±标准差。单种子的配置保留原值、std 为 NaN。
    """
    if df.empty or "seed" not in df.columns:
        return df
    group_cols = ["model_id", "few_shot_size", "lora_r", "supcon_lambda"]
    agg = df.groupby(group_cols).agg(
        acc_mean=("best_val_accuracy", "mean"),
        acc_std=("best_val_accuracy", "std"),
        f1_mean=("best_f1_macro", "mean"),
        f1_std=("best_f1_macro", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    agg.rename(columns={"acc_mean": "best_val_accuracy", "f1_mean": "best_f1_macro"}, inplace=True)
    return agg


def print_comparison_table(df: pd.DataFrame):
    if df.empty:
        print("[evaluate] 没有实验结果。请先运行 run_all.py。")
        return

    print("\n" + "=" * 80)
    print("          实验结果汇总（含 SupCon λ 消融）")
    print("=" * 80)
    print(f"{'模型':<28} {'λ':>6} {'最佳 Acc':>10} {'最终 F1':>10}")
    print("-" * 80)
    for _, row in df.iterrows():
        name = row["model_id"].split("/")[-1][:26]
        lam = f"{row['supcon_lambda']:.1f}"
        acc = f"{row['best_val_accuracy']:.4f}" if row["best_val_accuracy"] is not None else "N/A"
        f1 = f"{row['best_f1_macro']:.4f}" if row["best_f1_macro"] is not None else "N/A"
        print(f"{name:<28} {lam:>6} {acc:>10} {f1:>10}")
    print("=" * 80)

    # 如果存在多种子数据，追加均值±标准差汇总
    if "seed" in df.columns and df["seed"].nunique() > 1:
        agg = aggregate_seeds(df)
        multi = agg[agg["n_seeds"] > 1]
        if not multi.empty:
            print("\n--- 多种子均值 ± 标准差（仅列出 n_seeds>1 的配置）---")
            print(f"{'配置':<52} {'n':>3} {'Acc均值':>10} {'Acc ±std':>12} {'F1均值':>10} {'F1 ±std':>12}")
            print("-" * 100)
            for _, row in multi.iterrows():
                cfg = f"{row['model_id'].split('/')[-1]} fs={row['few_shot_size']} r={row['lora_r']} λ={row['supcon_lambda']}"
                n = int(row["n_seeds"])
                acc_m = f"{row['best_val_accuracy']:.4f}" if pd.notna(row["best_val_accuracy"]) else "N/A"
                acc_s = f"{row['acc_std']:.4f}" if pd.notna(row["acc_std"]) else "N/A"
                f1_m = f"{row['best_f1_macro']:.4f}" if pd.notna(row["best_f1_macro"]) else "N/A"
                f1_s = f"{row['f1_std']:.4f}" if pd.notna(row["f1_std"]) else "N/A"
                print(f"{cfg:<52} {n:>3} {acc_m:>10} {acc_s:>12} {f1_m:>10} {f1_s:>12}")
            print("=" * 100)


def plot_lambda_sensitivity(df: pd.DataFrame, save_path: str = None):
    """
    论文核心图表：λ vs Accuracy 折线图（消融实验）。
    多种子实验先聚合为均值，用误差棒展示标准差。
    """
    if df.empty:
        return

    # 先对多种子聚合，再固定 few_shot_size==500, lora_r==8（主设定）
    df_agg = aggregate_seeds(df)
    df_agg = df_agg[(df_agg["few_shot_size"] == 500) & (df_agg["lora_r"] == 8)]

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "lambda_sensitivity.png")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8E44AD"]
    for i, model_id in enumerate(df_agg["model_id"].unique()):
        model_df = df_agg[df_agg["model_id"] == model_id].sort_values("supcon_lambda")
        if len(model_df) < 2:
            continue
        short_name = model_id.split("/")[-1]
        color = colors[i % len(colors)]
        vals = model_df["best_val_accuracy"].values
        stds = model_df.get("acc_std", pd.Series([0] * len(vals))).values
        ax.errorbar(model_df["supcon_lambda"], vals, yerr=stds,
                    marker="o", linewidth=2, markersize=8, color=color,
                    capsize=4, label=short_name)

    ax.set_xlabel("SupCon Lambda (λ)")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Ablation Study: Effect of SupCon Loss Weight λ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] λ 敏感性曲线已保存: {save_path}")


def plot_supcon_vs_baseline(df: pd.DataFrame, save_path: str = None):
    """
    Baseline vs Best SupCon 分组柱状图（含误差棒）。
    先对多种子聚合，再取各模型 λ=0 均值 vs 最优 λ 均值。
    """
    if df.empty:
        return

    # 先聚合多种子，再固定 few_shot_size==500, lora_r==8（主设定）
    df_agg = aggregate_seeds(df)
    df_agg = df_agg[(df_agg["few_shot_size"] == 500) & (df_agg["lora_r"] == 8)]

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "supcon_vs_baseline.png")

    models = df_agg["model_id"].unique()
    x = np.arange(len(models))
    width = 0.35

    baseline_vals, baseline_errs = [], []
    best_supcon_vals, best_supcon_errs = [], []
    best_lambdas = []

    for m in models:
        mdf = df_agg[df_agg["model_id"] == m]
        bl = mdf[mdf["supcon_lambda"] == 0.0]
        if len(bl) > 0:
            baseline_vals.append(bl["best_val_accuracy"].values[0])
            baseline_errs.append(bl.get("acc_std", pd.Series([0])).values[0] if pd.notna(bl.get("acc_std", pd.Series([np.nan])).values[0]) else 0)
        else:
            baseline_vals.append(np.nan)
            baseline_errs.append(0)

        non_zero = mdf[mdf["supcon_lambda"] > 0]
        if len(non_zero) > 0:
            best_row = non_zero.loc[non_zero["best_val_accuracy"].idxmax()]
            best_supcon_vals.append(best_row["best_val_accuracy"])
            best_supcon_errs.append(best_row.get("acc_std", 0) if pd.notna(best_row.get("acc_std", np.nan)) else 0)
            best_lambdas.append(best_row["supcon_lambda"])
        else:
            best_supcon_vals.append(np.nan)
            best_supcon_errs.append(0)
            best_lambdas.append(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, baseline_vals, width, yerr=baseline_errs,
                   label="CE Only (Baseline)", color="#4C72B0", capsize=4)
    bars2 = ax.bar(x + width/2, best_supcon_vals, width, yerr=best_supcon_errs,
                   label="CE + SupCon (Best λ)", color="#55A868", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=25, ha="right")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Baseline vs SupCon-Enhanced LoRA Fine-tuning")
    ax.legend()
    ax.set_ylim(bottom=0)

    for i, (v, lam) in enumerate(zip(best_supcon_vals, best_lambdas)):
        if lam > 0 and not np.isnan(v):
            ax.annotate(f"λ={lam}", (x[i] + width/2, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color="#55A868")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Baseline vs SupCon 对比图已保存: {save_path}")


def plot_rank_ablation(df: pd.DataFrame, save_path: str = None):
    """
    LoRA rank × SupCon 交互消融图。
    先聚合多种子，再对比 λ=0 和 λ=1.0 下 rank 的影响。
    """
    if df.empty:
        return

    # 先聚合多种子
    df_agg = aggregate_seeds(df)

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "rank_ablation.png")

    best_lam = 1.0
    relevant = df_agg[(df_agg["supcon_lambda"].isin([0.0, best_lam])) & (df_agg["few_shot_size"] == 500)]
    if relevant.empty:
        relevant = df_agg[df_agg["supcon_lambda"].isin([0.0])]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    linestyles = ["-", "--"]

    for i, model_id in enumerate(relevant["model_id"].unique()):
        for j, lam in enumerate([0.0, best_lam]):
            sub = relevant[(relevant["model_id"] == model_id) & (relevant["supcon_lambda"] == lam)]
            if len(sub) < 1:
                continue
            sub = sub.sort_values("lora_r")
            short_name = model_id.split("/")[-1]
            label = f"{short_name} λ={lam}"
            vals = sub["best_val_accuracy"].values
            stds = sub.get("acc_std", pd.Series([0] * len(vals))).values
            ax.errorbar(sub["lora_r"], vals, yerr=stds,
                        marker="s", linewidth=2, markersize=8,
                        color=colors[i % len(colors)], linestyle=linestyles[j],
                        capsize=4, label=label)

    ax.set_xlabel("LoRA Rank (r)")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Ablation Study: Rank × SupCon Interaction")
    ax.set_xticks([4, 8, 16])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Rank×SupCon 交互图已保存: {save_path}")
