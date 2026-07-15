"""
v3 评估与可视化模块：跨架构（decoder vs encoder）对比
==========================================================
基于 code2/evaluate.py，新增：
  - architecture 字段处理
  - 跨架构对比图：decoder vs encoder × CE vs SupCon
  - 统计检验提示
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from code3.config import RESULTS_DIR

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
            "architecture": data.get("architecture", "decoder"),
            "few_shot_size": data.get("few_shot_size", -1),
            "lora_r": data.get("lora_r", 8),
            "supcon_lambda": data.get("supcon_lambda", 0.0),
            "pooling_strategy": data.get("pooling_strategy", "mean"),
            "seed": data.get("seed", 42),
            "best_val_accuracy": data.get("best_val_accuracy", None),
            "best_f1_macro": data.get("best_val_metrics", {}).get("f1_macro", None),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("best_val_accuracy", ascending=False)
    return df


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "seed" not in df.columns:
        return df
    group_cols = ["model_id", "architecture", "few_shot_size", "lora_r", "supcon_lambda"]
    if "pooling_strategy" in df.columns:
        group_cols.append("pooling_strategy")
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
        print("[evaluate] 没有实验结果。请先运行 run_v3.py。")
        return

    print("\n" + "=" * 90)
    print("          实验结果汇总（v3：epoch=11 + 跨架构对比）")
    print("=" * 90)
    print(f"{'模型':<30} {'架构':>8} {'λ':>6} {'最佳 Acc':>10} {'最终 F1':>10}")
    print("-" * 90)
    for _, row in df.iterrows():
        name = row["model_id"].split("/")[-1][:28]
        arch = row.get("architecture", "decoder")[:8]
        lam = f"{row['supcon_lambda']:.1f}"
        acc = f"{row['best_val_accuracy']:.4f}" if row["best_val_accuracy"] is not None else "N/A"
        f1 = f"{row['best_f1_macro']:.4f}" if row["best_f1_macro"] is not None else "N/A"
        print(f"{name:<30} {arch:>8} {lam:>6} {acc:>10} {f1:>10}")
    print("=" * 90)

    if "seed" in df.columns and df["seed"].nunique() > 1:
        agg = aggregate_seeds(df)
        multi = agg[agg["n_seeds"] > 1]
        if not multi.empty:
            print("\n--- 多种子均值 ± 标准差（n_seeds>1）---")
            print(f"{'配置':<55} {'n':>3} {'Acc均值':>10} {'Acc ±std':>12} {'F1均值':>10} {'F1 ±std':>12}")
            print("-" * 105)
            for _, row in multi.iterrows():
                model_short = row["model_id"].split("/")[-1]
                arch = row.get("architecture", "decoder")
                cfg = f"[{arch}] {model_short} fs={row['few_shot_size']} r={row['lora_r']} λ={row['supcon_lambda']}"
                n = int(row["n_seeds"])
                acc_m = f"{row['best_val_accuracy']:.4f}" if pd.notna(row["best_val_accuracy"]) else "N/A"
                acc_s = f"{row['acc_std']:.4f}" if pd.notna(row["acc_std"]) else "N/A"
                f1_m = f"{row['best_f1_macro']:.4f}" if pd.notna(row["best_f1_macro"]) else "N/A"
                f1_s = f"{row['f1_std']:.4f}" if pd.notna(row["f1_std"]) else "N/A"
                print(f"{cfg:<55} {n:>3} {acc_m:>10} {acc_s:>12} {f1_m:>10} {f1_s:>12}")
            print("=" * 105)


def _select_best_pooling(sc_data: pd.DataFrame) -> str:
    """
    根据 pooling 消融结果确定 decoder 使用哪种 pooling 用于跨架构图。

    策略：优先选较优者，差异不显著时固定 mean（最广泛使用的 baseline）。
    返回 "mean" 或 "last_token"。
    """
    if "pooling_strategy" not in sc_data.columns:
        return "mean"
    decoder_sc = sc_data[sc_data["architecture"] == "decoder"]
    mean_rows = decoder_sc[decoder_sc["pooling_strategy"] == "mean"]
    last_rows = decoder_sc[decoder_sc["pooling_strategy"] == "last_token"]
    if len(mean_rows) == 0 or len(last_rows) == 0:
        return "mean"

    mean_acc = mean_rows["best_val_accuracy"].values[0]
    last_acc = last_rows["best_val_accuracy"].values[0]
    mean_std = mean_rows.get("acc_std", pd.Series([0])).values[0]
    last_std = last_rows.get("acc_std", pd.Series([0])).values[0]
    diff = last_acc - mean_acc
    pooled_std = np.sqrt((mean_std**2 + last_std**2) / 2) if (pd.notna(mean_std) and pd.notna(last_std)) else 1e-8
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    if abs(cohens_d) > 0.5:
        best = "last_token" if diff > 0 else "mean"
        print(f"[evaluate] Decoder pooling 消融: mean={mean_acc:.4f}, last_token={last_acc:.4f}, "
              f"d={cohens_d:.2f} → 选用 {best}")
    else:
        best = "mean"
        print(f"[evaluate] Decoder pooling 消融: mean={mean_acc:.4f}, last_token={last_acc:.4f}, "
              f"d={cohens_d:.2f} (|d|<0.5, 差异不显著) → 固定使用 mean")

    return best


def plot_cross_architecture(df: pd.DataFrame, save_path: str = None):
    """
    跨架构对比图：decoder vs encoder × CE vs SupCon。
    核心论文图表。

    SupCon 柱的选择策略：
    - 先检查 pooling 消融结果：若 mean vs last_token 差异显著（Cohen's d > 0.5），
      固定使用较优者；否则使用 mean pooling（最广泛使用的 baseline）。
    - 图注写明选择依据，不做事后 max 选择（避免系统性高估）。
    """
    if df.empty:
        return

    df_agg = aggregate_seeds(df)
    df_agg = df_agg[df_agg["few_shot_size"] == 500]

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "cross_architecture.png")

    # 筛选 CE 和 SupCon(λ=1.0) 数据
    ce_data = df_agg[df_agg["supcon_lambda"] == 0.0]
    sc_data = df_agg[df_agg["supcon_lambda"] == 1.0]

    if ce_data.empty:
        print("[evaluate] 无 CE baseline 数据")
        return

    # ---- 确定 decoder 下使用哪种 pooling ----
    # 根据 pooling 消融结果选择：优先选较优者，差异不显著时固定 mean
    decoder_pooling = _select_best_pooling(sc_data)

    architectures = sorted(ce_data["architecture"].unique())
    fig, axes = plt.subplots(1, len(architectures), figsize=(6 * len(architectures), 5), squeeze=False)
    axes = axes[0]

    colors = {"CE": "#4C72B0", "SupCon": "#55A868"}

    for ax_i, arch in enumerate(architectures):
        ax = axes[ax_i]

        ce_arch = ce_data[ce_data["architecture"] == arch]
        sc_arch = sc_data[sc_data["architecture"] == arch]

        models = ce_arch["model_id"].unique()
        x = np.arange(len(models))
        width = 0.35

        ce_vals, ce_errs = [], []
        sc_vals, sc_errs = [], []
        for m in models:
            row = ce_arch[ce_arch["model_id"] == m]
            ce_vals.append(row["best_val_accuracy"].values[0] if len(row) > 0 else np.nan)
            ce_errs.append(row.get("acc_std", pd.Series([0])).values[0] if len(row) > 0 and pd.notna(row.get("acc_std", pd.Series([np.nan])).values[0]) else 0)

            # 显式选择 SupCon pooling：decoder 用消融选定的 pooling，encoder 用 cls_token
            if arch == "decoder" and "pooling_strategy" in sc_arch.columns:
                row = sc_arch[(sc_arch["model_id"] == m) & (sc_arch["pooling_strategy"] == decoder_pooling)]
            else:
                row = sc_arch[sc_arch["model_id"] == m]
            sc_vals.append(row["best_val_accuracy"].values[0] if len(row) > 0 else np.nan)
            sc_errs.append(row.get("acc_std", pd.Series([0])).values[0] if len(row) > 0 and pd.notna(row.get("acc_std", pd.Series([np.nan])).values[0]) else 0)

        ax.bar(x - width/2, ce_vals, width, yerr=ce_errs, label="CE Only", color=colors["CE"], capsize=4)
        ax.bar(x + width/2, sc_vals, width, yerr=sc_errs, label="CE + SupCon", color=colors["SupCon"], capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels([m.split("/")[-1][:20] for m in models], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Best Validation Accuracy")
        ax.set_title(f"{arch.upper()} Architecture")
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

        # 标注 Δ
        for i in range(len(models)):
            if pd.notna(ce_vals[i]) and pd.notna(sc_vals[i]):
                delta = sc_vals[i] - ce_vals[i]
                color = "#2ca02c" if delta > 0 else "#d62728"
                ax.annotate(f"Δ={delta:+.1%}", (x[i] + width/2, sc_vals[i]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=color)

    # 图注说明 decoder pooling 选择依据
    if decoder_pooling:
        note = f"Decoder SupCon uses {decoder_pooling} pooling"
        fig.text(0.5, 0.01, note, ha="center", fontsize=9, fontstyle="italic", color="gray")

    fig.suptitle("Cross-Architecture Comparison: Decoder vs Encoder", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] 跨架构对比图已保存: {save_path} (decoder pooling: {decoder_pooling})")


def plot_pooling_ablation(df: pd.DataFrame, save_path: str = None):
    """
    Decoder Pooling 消融柱状图：CE baseline vs SupCon(mean) vs SupCon(last_token)。
    限定 decoder-only 架构（encoder 的 cls_token vs mean 已有大量文献研究）。
    若 encoder 后续补了 mean pooling 对照实验，可扩展为子图。
    """
    if df.empty:
        return

    df_agg = aggregate_seeds(df)
    # 限定 decoder + fs=500
    df_agg = df_agg[(df_agg["few_shot_size"] == 500) & (df_agg["architecture"] == "decoder")]

    if df_agg.empty:
        print("[evaluate] 无 decoder pooling 消融数据，跳过 pooling_ablation.png")
        return

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "pooling_ablation.png")

    ce_data = df_agg[df_agg["supcon_lambda"] == 0.0]
    supcon_mean = df_agg[(df_agg["supcon_lambda"] == 1.0) & (df_agg["pooling_strategy"] == "mean")]
    supcon_last = df_agg[(df_agg["supcon_lambda"] == 1.0) & (df_agg["pooling_strategy"] == "last_token")]

    if ce_data.empty:
        print("[evaluate] 无 CE baseline 数据")
        return

    models = ce_data["model_id"].unique()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    def _get(sub_df, model_order):
        vals, errs = [], []
        for m in model_order:
            row = sub_df[sub_df["model_id"] == m]
            if len(row) > 0:
                vals.append(row["best_val_accuracy"].values[0])
                e = row.get("acc_std", pd.Series([0])).values[0]
                errs.append(e if pd.notna(e) else 0)
            else:
                vals.append(np.nan)
                errs.append(0)
        return vals, errs

    ce_vals, ce_errs = _get(ce_data, models)
    mean_vals, mean_errs = _get(supcon_mean, models)
    last_vals, last_errs = _get(supcon_last, models)

    ax.bar(x - width, ce_vals, width, yerr=ce_errs, label="CE Baseline", color="#4C72B0", capsize=4)
    ax.bar(x, mean_vals, width, yerr=mean_errs, label="SupCon (mean)", color="#55A868", capsize=4)
    ax.bar(x + width, last_vals, width, yerr=last_errs, label="SupCon (last_token)", color="#C44E52", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=25, ha="right")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Decoder Pooling Strategy Ablation (Qwen2.5-1.5B)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    for i in range(len(models)):
        if pd.notna(ce_vals[i]):
            for pos, vals in [(0, mean_vals), (width, last_vals)]:
                v = vals[i]
                if pd.notna(v):
                    delta = v - ce_vals[i]
                    c = "#55A868" if delta > 0 else "#C44E52"
                    ax.annotate(f"{delta:+.1%}", (x[i] + pos, v),
                                textcoords="offset points", xytext=(0, 10),
                                ha="center", fontsize=7, color=c)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Decoder Pooling 消融图已保存: {save_path}")


def print_statistical_note(df: pd.DataFrame):
    """输出统计检验提示：帮助判断 SupCon 增益是否显著。"""
    if df.empty or "seed" not in df.columns or df["seed"].nunique() < 3:
        return

    print("\n" + "=" * 60)
    print("  统计检验提示（仅信息参考，非正式检验）")
    print("=" * 60)

    for arch in df["architecture"].unique():
        arch_df = df[df["architecture"] == arch]
        for model in arch_df["model_id"].unique():
            mdf = arch_df[arch_df["model_id"] == model]
            ce = mdf[mdf["supcon_lambda"] == 0.0]["best_val_accuracy"]
            # decoder 下 SupCon 可能有多个 pooling，需选定单一策略避免混池
            sc_all = mdf[mdf["supcon_lambda"] == 1.0]
            if "pooling_strategy" in sc_all.columns and sc_all["pooling_strategy"].nunique() > 1:
                best_pool = _select_best_pooling(sc_all)
                sc = sc_all[sc_all["pooling_strategy"] == best_pool]["best_val_accuracy"]
            else:
                sc = sc_all["best_val_accuracy"]
            if len(ce) < 3 or len(sc) < 3:
                continue
            delta_mean = sc.mean() - ce.mean()
            # 简易 effect size：Cohen's d（pooled std）
            pooled_std = np.sqrt((ce.std()**2 + sc.std()**2) / 2)
            cohens_d = delta_mean / pooled_std if pooled_std > 0 else 0
            name = model.split("/")[-1]
            print(f"\n[{arch}] {name}:")
            print(f"  CE     = {ce.mean():.4f} ± {ce.std():.4f} (n={len(ce)})")
            print(f"  SupCon = {sc.mean():.4f} ± {sc.std():.4f} (n={len(sc)})")
            print(f"  Δ      = {delta_mean:+.4f} (Cohen's d = {cohens_d:.2f})")
            if abs(cohens_d) < 0.2:
                note = "可忽略（|d| < 0.2）"
            elif abs(cohens_d) < 0.5:
                note = "弱效应（0.2 ≤ |d| < 0.5）"
            elif abs(cohens_d) < 0.8:
                note = "中等效应（0.5 ≤ |d| < 0.8）"
            else:
                note = "强效应（|d| ≥ 0.8）"
            print(f"  Effect size 解释: {note}")

    print("\n注：Cohen's d 为简易近似。正式论文建议做 paired t-test 或 Wilcoxon。")
    print("=" * 60)
