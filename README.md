# LoRA + SupCon 微调实验 — 开发与部署记录

## 一、项目概述

- **方向**：LLM LoRA 微调 + 监督对比损失（SupCon）少样本文本分类
- **论文定位**：**实证/系统分析型论文**，非新方法论文。核心贡献为"系统刻画监督对比损失在 LoRA 微调 decoder-only 中文 LLM 上的增益规律"
- **数据集**：THUCNews 中文新闻分类，14 类，训练 668,860 条，验证 167,215 条
- **模型**：Qwen/Qwen2.5-0.5B（decoder-only，无 \[CLS] token，SupCon 使用 mean pooling）
- **方法**：CE Loss + λ×SupCon 联合损失，LoRA r=8 注入 q\_proj/v\_proj（可训练参数 553,216 / 494,598,528 = 0.11%）
- **SupCon embedding**：最后一层 masked mean pooling（`pool_hidden_states`，使用 attention\_mask 排除 padding）

***

## 二、代码问题发现与修复

### 第一轮修复（6 项，阻塞性/运行错误）

1. **`device`** **变量未定义（阻塞性 Bug）**
   - **文件**：`code/train.py` 第 60 行

- **问题**：`build_model_and_peft()` 函数内使用了 `model.to(device)`，但 `device` 未作为参数传入，也未在函数内定义
- **修复**：函数签名增加 `device: torch.device` 参数，调用处传入

### 2. 相对导入无法被外部脚本加载

- **文件**：`code/train.py`，`code/evaluate.py`
- **问题**：内部使用 `from .config import ...` 相对导入。从根目录 `run_experiment.py` 启动时，Python 模块解析失败
- **修复**：改为绝对导入 `from code.config import ...`，配合 `sys.path` 设置

### 3. SupCon 模式下 `hidden_states` 为 None

- **文件**：`code/train.py` 第 94-95 行
- **问题**：PEFT 模型在 forward 时不自动返回 hidden\_states，即使 config 中设置了 `output_hidden_states=True`
- **修复**：在 forward 调用时显式传入 `output_hidden_states=True`

### 4. `torch_dtype` 参数废弃

- **文件**：`code/train.py` 第 36 行
- **问题**：新版 transformers 要求使用 `dtype` 代替 `torch_dtype`
- **修复**：将 `torch_dtype=torch.float32` 改为 `dtype=torch.float32`

### 5. 可复现性缺失

- **问题1**：`run_all.py` 和 `run_experiment.py` 均未设置 `torch.manual_seed()`
- **问题2**：`create_dataloaders()` 每次调用 `torch.randperm()` 生成不同随机子集，导致同 `few_shot_size` 实验对比不公平
- **修复**：两处入口添加全局种子；`create_dataloaders` 使用 `torch.Generator` 以 `config.seed + few_shot_size` 为种子固定子集

### 6. `total_mem` 属性名错误

- **文件**：`run_all.py` 第 30 行
- **问题**：PyTorch 2.5.1 中属性名为 `total_memory` 而非 `total_mem`
- **修复**：改为 `total_memory`

***

### 第二轮修复（12 项，逻辑漏洞/数据污染，2026-07-11）

对全部 7 个 `.py` 文件逐行审查，发现并修复 12 个漏洞。

#### 高危（会污染结论）

1. **SupCon 未使用 masked mean pooling → padding 污染 embedding**
   - 影响：`supcon_loss.py` `extract_cls_embedding` → 重命名为 `pool_hidden_states`，增加 `attention_mask` 参数做掩码均值
   - `train.py` 调用处传入 `attention_mask`
   - 后果：旧代码用 `last_hidden.mean(dim=1)` 把 padding（占 256 序列大半）平等平均，不同样本被 padding 稀释程度不同，对比特征不可比
2. **绘图函数未适配多种子 → cherry-pick 最优跑**
   - 影响：`evaluate.py` 三个绘图函数直接传原始 df，同 λ 下有 3 个种子点叠成锯齿
   - `plot_supcon_vs_baseline` 的 `idxmax()` 在全局选最优行，系统性高估 SupCon
   - 修复：绘图前先调 `aggregate_seeds()`，用均值画线/柱，用 std 画误差棒
3. **λ 敏感性图和柱状图未固定 fs/r → 混入不同数据量的行**
   - 影响：同一 model 在 λ=0 处有 fs=100/500/full 和 Stage2 r=4/16 多行，x=0 叠多个点
   - 修复：`plot_lambda_sensitivity` 和 `plot_supcon_vs_baseline` 聚合后过滤 `few_shot_size==500, lora_r==8`
4. **旧格式结果文件残留 → load\_all\_results 仍读取**
   - 影响：`results/result_Qwen_Qwen2.5-0.5B_fs100_r8_CEonly.json`（无 `_seed` 后缀）被 `load_all_results` glob 读取，其 `final_val_metrics` 字段与新 `best_val_metrics` 不匹配，F1=None
   - 修复：删除旧文件，删除 `_result_exists` 中对无 `_seed` 旧格式的兼容分支
5. **best\_val\_accuracy 与 F1 来自不同 epoch**
   - 影响：`train.py` 仅更新标量 `best_val_acc`，`final_val_metrics` 是最后 epoch 的指标
   - 修复：最优 epoch 时同步保存完整 `best_val_metrics` 字典，写入 json 时配对各指标
6. **Stage 2 rank 消融用次优 λ=0.5**
   - 影响：主线已确认 0.5B 上 λ=0.5 掉点、λ=1.0 最佳，Stage2 却用 0.5 做 rank×SupCon 交互
   - 修复：`run_experiment.py` L106 和 `evaluate.py` plot\_rank\_ablation 均改为 λ=1.0

#### 中危（口径/鲁棒性）

1. **`torch.amp.autocast("cuda")`** **硬编码 → CPU 崩溃**
   - 修复：改为 `autocast(device.type, ...)`，`build_model_and_peft` 的 bf16 dtype 增加 `device=="cuda" and is_bf16_supported()` 判断
2. **bf16 标志不一致**：`build_model_and_peft` 用 `use_bf16_safe`，`run_single_experiment` 传 `config.use_bf16`
   - 修复：统一计算 `use_bf16_safe` 后传入
3. **tokenizer 绑定首个模型，多模型实验无警告**
   - 修复：`data_loader.py` 增加多模型提示
4. **分层采样不足量时静默缩水**
   - 修复：增加样本量不足告警

#### 低危（维护/测试）

1. **`config`** **全局单例被循环突变**
   - 修复：`run_experiment.py` Stage2 前后保存/恢复 `lora_r` 和 `lora_alpha`
2. **`smoke_test.py`** **SupConLoss 实例复用**
   - 修复：测试2/3 各自独立创建 `supcon2`/`supcon3`

***

## 三、性能优化

### 问题：验证集 tokenize 重复执行

- 本地 RTX 4060 验证 167K 样本需 7 小时（10s/batch），其中 tokenize 占用大量 CPU 时间
- 每个子实验（18 个）都重新 tokenize 相同验证集

### 解决方案

- 验证集改为启动时一次性预 tokenize，存入内存，后续评估直接读取
- 训练集保持懒加载（预 tokenize 79 万条内存过高）

***

## 四、服务器选型决策

### 需求分析

| 参数     | 最低要求       | 说明                 |
| ------ | ---------- | ------------------ |
| GPU 算力 | >40 TFLOPS | 0.5B 模型 LoRA 微调    |
| 显存     | >8GB       | fp32 batch\_size=8 |
| 价格     | <3 元/h     | 18 实验预估 30-40h     |

### 候选对比

| GPU           | 显存       | 算力            | 单价           | 结论               |
| ------------- | -------- | ------------- | ------------ | ---------------- |
| RTX 4090D     | 24GB     | 74 TFLOPS     | 1.88 元/h     | 算力低 11%，价格便宜 14% |
| **RTX 4090**  | **24GB** | **83 TFLOPS** | **2.18 元/h** | **最优，总差价仅 3 元**  |
| RTX 4090 48GB | 48GB     | 83 TFLOPS     | —            | 0.5B 模型用不到 48G   |
| A100 40GB     | 40GB     | 19 TFLOPS     | 更贵           | 算力不如 4090，适合大模型  |

### 最终选择：RTX 4090

- 83 TFLOPS 是候选中最高的单卡算力
- 24GB 显存绰绰有余
- 总花费预估 78 元 vs 4090D 的 75 元，多 3 元换 4 小时提速

### 平台：AutoDL

- 国内访问快，无需 VPN
- 按小时计费，方便弹性使用
- PyTorch 2.5.1 / CUDA 12.4 / Python 3.12 / Ubuntu 22.04

***

## 五、数据集问题

### 本地压缩失败

- THUCNews 包含 198 万个小文件
- `tar` 命令行打包因输出截断产生损坏
- `Compress-Archive` 同样因文件数过多失败

### 解决方案

- 本地使用 robocopy 快速删除 data/THUCNews 目录
- 本地仅上传代码 + 模型（950MB）
- 服务器端通过 `wget` 重新下载 THUCNews.zip 解压

***

## 六、速度对比

| 指标          | 本地 RTX 4060 | 服务器 RTX 4090 | 提速    |
| ----------- | ----------- | ------------ | ----- |
| Tokenize 速度 | 97 it/s     | 405 it/s     | 4.2x  |
| 评估速度        | 0.47 it/s   | 10.08 it/s   | 21.4x |
| 单次验证        | \~7 小时      | \~17 分钟      | 24x   |
| 全实验预估       | \~16 天      | \~22 小时      | 17x   |

***

## 七、实验矩阵与当前进度

### 2026-07-13 最终结果 — 全部完成（32/32 组）

> **进度**：全部实验已完成。阶段1（24组）+ 阶段2 LoRA rank 消融（8组）。

### 0.5B 实际结果（干净数据）

#### fs=-1 全量（均 seed=42）

| λ        | Acc    | F1     |
| -------- | ------ | ------ |
| 0.0 (CE) | 97.39% | 97.11% |
| 0.1      | 97.40% | 97.09% |
| 0.5      | 97.43% | 97.16% |
| 1.0      | 97.29% | 96.96% |

> 全量已饱和。λ=1.0 时 SupCon 轻微掉点（-0.1%），符合"CE 已达上限、对比损失无增益空间"预期。

#### fs=100（均 seed=42）

| λ        | Acc    | F1    |
| -------- | ------ | ----- |
| 0.0 (CE) | 14.13% | 8.93% |
| 0.1      | 14.12% | 8.95% |
| 0.5      | 14.11% | 8.94% |
| 1.0      | 14.12% | 8.92% |

> SupCon 完全无效。100 样本下模型连 CE 都没学好（14 分类随机基线 7.1%），正负对比缺乏基础。

#### fs=500 多种子

| λ   | seed=42         | seed=123        | seed=456        |
| --- | --------------- | --------------- | --------------- |
| 0.0 | 28.21% / 24.08% | 37.21% / 32.14% | 34.52% / 29.55% |
| 0.1 | 28.31% / 24.19% | —               | —               |
| 0.5 | 28.37% / 24.24% | —               | —               |
| 1.0 | 28.62% / 24.42% | 38.57% / 33.03% | 35.03% / 30.06% |

> **seed 差异 \~10% acc，SupCon λ 差异 \~0.4% acc**。子集选择质量是主导因素，SupCon 信号微弱。

### 1.5B 实际结果（新增）

#### fs=100（seed=42）

| λ            | Acc   | F1    |
| ------------ | ----- | ----- |
| 0.0 (CE)     | 9.67% | 7.45% |
| 1.0 (SupCon) | 9.67% | 7.45% |

> 两个异常信号：(1) 1.5B 比 0.5B 的 14.13% **低了 4.5 个百分点**，大模型在极端少样本下过拟合更严重。(2) SupCon 零增益。

#### fs=500 多种子

| λ            | seed=42         | seed=123        | seed=456        |
| ------------ | --------------- | --------------- | --------------- |
| 0.0 (CE)     | 35.27% / 30.31% | 32.91% / 28.18% | 39.53% / 34.53% |
| 1.0 (SupCon) | 36.48% / 31.52% | 33.46% / 28.70% | 40.92% / 35.96% |
| Δ            | +1.21 / +1.21   | +0.55 / +0.52   | +1.39 / +1.43   |

> **1.5B 的 SupCon 增益比 0.5B 更一致**：三个种子均在 +0.5%\~+1.4% 区间，均值约 +1.05%（vs 0.5B 的 +0.76%）。可能因为 1.5B hidden dim=1536 比 0.5B 的 896 提供了更丰富的对比特征空间。但种子间 CE 基线波动（32.9\~39.5%，范围 6.6%）仍远超 SupCon 收益，batch=8 的瓶颈依然存在。

### Stage 2：LoRA Rank × SupCon 交叉消融（fs=500，seed=42）

#### 0.5B

| r  | CE              | SupCon λ=1.0    | Δ             |
| -- | --------------- | --------------- | ------------- |
| 4  | 27.03% / 22.76% | 27.24% / 22.91% | +0.21 / +0.15 |
| 8  | 28.21% / 24.08% | 28.62% / 24.42% | +0.41 / +0.34 |
| 16 | 30.72% / 26.10% | 31.58% / 26.74% | +0.86 / +0.64 |

> rank 越大 CE 基线越高（27.03%→30.72%），SupCon 增益也越大（+0.21%→+0.86%）。但总体增益仍微弱。

#### 1.5B

| r  | CE              | SupCon λ=1.0        | Δ                 |
| -- | --------------- | ------------------- | ----------------- |
| 4  | 32.21% / 27.23% | 32.98% / 27.94%     | +0.77 / +0.71     |
| 8  | 35.27% / 30.31% | 36.48% / 31.52%     | +1.21 / +1.21     |
| 16 | 43.93% / 38.11% | **46.09% / 40.37%** | **+2.16 / +2.26** |

> **1.5B r=16 SupCon 是全实验中 SupCon 唯一突破 2% 的配置**（+2.16%），也是首次 SupCon 增益超过 2 个百分点。但需注意这是单种子结果（seed=42），未做多种子验证，存在种子偏差风险——seed=42 在阶段1 1.5B 中 CE 基线 35.27%，处于三个种子中偏中间的位置，非极端值。

### 核心发现：batch size=8 是 SupCon 失效的根因

- 14 分类 × batch=8 → 每 batch 平均每类 0.57 个样本
- SupCon 正样本 mask 几乎全零 → 对比损失退化
- 这是实验设计层面的瓶颈，不是代码 bug 也不是"SupCon 本身无效"
- **1.5B 新发现**：SupCon 增益在 1.5B 上比 0.5B 更一致（均值 +1.05% vs +0.76%），提示大模型在更大 hidden dim 下可能从对比学习中获益更多。但 batch=8 的限制使得 seed 噪声仍主导结果。
- **Stage 2 新发现**：SupCon 增益随模型规模 AND rank 递增。1.5B r=16 实现 +2.16%，首次突破 2%。这提示 SupCon 的性能上限可能由 hidden\_dim × rank 的联合容量决定——对比学习需要足够大的特征空间才能生效。但单种子限制使得该结论需谨慎对待。

### 后续方向（2026-07-13 更新，全实验完成后）

| 优先级   | 内容                               | 说明                                                         |
| ----- | -------------------------------- | ---------------------------------------------------------- |
| 🔴 P0 | 增大 batch size（≥28）               | SupCon 有效的必要条件：每 batch 各类 ≥2 样本。0.5B+bf16 在 24GB 可到 32-48  |
| 🔴 P0 | label-aware batch sampling       | 主动构造每 batch 各类至少 2 样本，不等 shuffle 随机                        |
| 🔴 P0 | **聚焦 1.5B r=16** 重跑核心对比点         | 当前最优配置，种种子验证（3种子，λ=0 vs 1.0）。若多种子确认 +2.16% 增益，这是论文最有说服力的结果 |
| 🟡 P1 | Pooling 策略消融（mean vs last-token） | 在增大 batch 后加一组对比，验证 causal mean pooling 是否为第二瓶颈            |
| 🟡 P1 | 若 batch增大后 SupCon 仍无明显增益         | 接受"SupCon 在 decoder-only LoRA 小样本下无效"为负面结论，改叙事方向           |

***

## 八、最终项目结构

```
lunwen/
├── run_experiment.py              # 唯一入口
├── models/
│   └── Qwen_Qwen2.5-0.5B/        # 本地模型
├── project-a-llm-lora/
│   ├── code/
│   │   ├── config.py              # 全局配置 + 本地模型路径解析
│   │   ├── data_loader.py         # TokenizedDataset（验证集预缓存）
│   │   ├── train.py               # LoRA 训练 + SupCon
│   │   ├── supcon_loss.py         # 监督对比损失
│   │   └── evaluate.py            # 结果汇总 + 可视化
│   ├── data/THUCNews/             # 服务器端下载
│   └── results/                   # 实验结果
└── docs/
    └── 08-experiment-log.md       # 本文档
```

***

## 九、论文定位与期刊策略

### 9.1 论文叙事（2026-07-13 更新，全实验完成）

全部 32 组实验的核心发现：

> batch=8 × 14类限制下，Stage 1（λ 消融）SupCon 增益微弱（0.5B 均值 +0.76%，1.5B 均值 +1.05%），均被 seed 间波动（6-10%）淹没。Stage 2（rank 消融）揭示 SupCon 增益随模型规模和 rank 递增：1.5B r=16 实现 +2.16%，首次突破 2 个百分点。这提示 SupCon 的性能上限可能由 hidden\_dim × rank 的联合容量决定——对比学习需要足够大的特征空间才能生效。但该配置仅单种子，需多种子验证排除种子偏差。下一步：增大 batch size（≥28）+ label-aware sampling，聚焦 1.5B r=16 做种种子验证。若确认增益，论文叙事可为"SupCon 在足够大特征空间下有效，batch size 和模型容量是关键前提"；若仍无效，转向"SupCon 在 decoder-only LoRA 上失效的实证分析"。

当前结论：不能断言"SupCon 无效"，只能说"当前 batch size 配置下 SupCon 无增益"。需补 batch 消融实验后方可定论。

### 9.2 decoder-only 池化策略（2026-07-13 更新，经独立核验后修正）

> 关键问题：Qwen2.5 是 decoder-only 架构，causal hidden states 经 mean pooling 后是否能携带足够的类别区分信息来支撑对比学习？
> 结论：**能支撑，但在保留 causal mask 的设定下，last-token pooling 理论上更合理。Pooling 策略是 SupCon 效果弱的次要原因（主因仍是 batch size=8）。**

#### 一、Mean Pooling 在 causal decoder 下的本质缺陷

因果掩码下每个 token 只能看到自身及左侧 token：

- 序列前部 token 几乎无上下文信息
- 序列中部 token 只有左半部分上下文
- 仅最后一个 token（EOS）见过完整序列

Mean pooling 把大量"信息不完整"的表征等权平均，整体语义被稀释。这是 causal mask 的结构性后果，不是实现层面的问题。

#### 二、相关研究（按实验条件区分）

**这是最容易出错的地方：不同工作的 pooling 结论高度依赖是否修改了 attention mask，不能跨条件引用。**

| 实验条件                      | 代表工作                         | pooling 结论                                      |
| ------------------------- | ---------------------------- | ----------------------------------------------- |
| **去掉 causal mask（双向改造后）** | LLM2Vec (2024)               | mean pooling 优于 EOS/last-token，因缓解 recency bias |
| **保留 causal mask**（本实验设定） | Causal2Vec, KV-Embedding     | last-token 更优，早期 token 信息不完整是固有缺陷               |
| 阿里官方 embedding 模型         | GTE-Qwen2, Qwen2.5-Embedding | 默认采用 last-token pooling                         |

**关键区分**：LLM2Vec 的"mean pooling 更优"结论是在改为双向 attention + 继续预训练后得出的，不能直接迁移到保留 causal mask 的纯 decoder 设定。在本实验当前设定下，**last-token pooling 很可能优于 mean pooling**，而非"各有优劣"。

#### 三、对本实验的影响

- Mean pooling 不是 SupCon 无效的主因（主因 batch size=8 导致正样本 mask 全零）
- 但 mean pooling 可能是"第二瓶颈"：在保留 causal mask 的条件下，信息不完整的前部 token 稀释了对比特征
- LoRA 低秩适配可微调注意力模式，但无法从根本上改变 causal mask 的单向信息流

#### 四、计划消融方案

在增大 batch size 后的核心对比实验中加入 pooling 策略对比：

| Pooling 策略               | 实现方式                             | 定位                  |
| ------------------------ | -------------------------------- | ------------------- |
| Masked mean pooling（当前）  | `attention_mask` 加权均值            | baseline            |
| Last token / EOS pooling | 取每个样本最后一个有效 token 的 hidden state | 理论上有优势，需实测验证        |
| Weighted mean pooling    | 位置加权（后面 token 权重更高）              | 探索性候选，无先验文献支持，效果需实测 |

改动量极小，仅需在 `supcon_loss.py` 增加 `pooling_strategy` 参数。

#### 五、探针实验（快速验证 pooling 差距）

冻结 backbone + LoRA，两种 pooling 分别训线性分类器，看准确率差距。

**判断标准**：不以绝对百分点定阈值，而看差距相对于 CE 基线的比例。若差距不到基线 acc 的 1/10，说明 pooling 不是瓶颈；若达 1/5 以上，pooling 策略有显著优化空间。具体阈值结合实测数据判断。

### 9.3 期刊目标

| 档位  | 期刊                         | 定位              |
| --- | -------------------------- | --------------- |
| 冲刺档 | 《中文信息学报》（CSCD+北大核心）        | 先投试水，被拒转投下一档    |
| 稳妥档 | 《计算机应用》《计算机工程与应用》《计算机与现代化》 | 以本科生+单卡体量，大概率能中 |
| 保底档 | 各高校学报（自然科学版）或普通普刊          | 十拿九稳            |

策略：补齐 P0+P1 后先投《中文信息学报》，被拒不重做实验直接转投计算机应用类。

### 9.4 自检问题（来自行为准则第0条，2026-07-13 全实验完成）

1. **就现在的情况，我最大的遗漏是什么？**
   - batch size=8 是 SupCon 失效的根因，14 分类下 batch<14 意味着每 class 期望样本数 <1，SupCon 的正负对比几乎不可能形成。
   - decoder-only 架构的 causal mean pooling 可能成为第二瓶颈。
   - **新增**：Stage 2 仅单种子（seed=42），1.5B r=16 的 +2.16% 增益未排除种子偏差。若多种子验证后增益消失，论文最有力的证据点将不成立。
2. **当前我最没把握的事是什么？**
   - 增大 batch size 后 1.5B r=16 SupCon 增益能否在多种子下复现——这是决定论文叙事方向的关键实验。
   - 1.5B r=16 在增大 batch 后的显存是否够用——r=16 可训练参数 2.2M，比 r=8 翻倍，batch=28 可能爆显存。

***

## 十、实验状态

- **首次启动**：2026-07-08
- **代码修复**：2026-07-11（12 项逻辑漏洞修复）
- **服务器重启**：2026-07-13（解决 Python 标准库 `code.py` 冲突，`PYTHONPATH` 启动）
- **全部完成**：2026-07-13 20:19（32/32 组全部完成，含阶段1 λ消融 + 阶段2 rank消融）
- **关键发现**：
  - batch=8 导致 SupCon 退化，种子噪声（6-10%）淹没对比信号
  - SupCon 增益随模型规模 AND rank 递增：1.5B r=16 实现 +2.16%，全实验唯一突破 2%
  - SupCon 的性能上限可能由 hidden\_dim × rank 联合容量决定

***

## 十一、v2 实验：增大 batch + label-aware sampling（9组，已失败）

### 11.1 实验设计

- **代码目录**：`code2/`
- **服务器**：AutoDL vGPU-48GB-350W / RTX 3090 50.9GB
- **启动时间**：2026-07-13 22:40
- **启动命令**：`PYTHONPATH="/root/autodl-tmp/lunwen/project-a-llm-lora" nohup python -m code2.run_v2 > experiment_v2.log 2>&1 &`

### 11.2 实验矩阵（9组）

| 阶段                 | 模型   | r  | fs  | lambda | pooling     | 种子           |
| ------------------ | ---- | -- | --- | ------ | ----------- | ------------ |
| CE baseline        | 1.5B | 16 | 500 | 0.0    | mean        | 42, 123, 456 |
| SupCon mean        | 1.5B | 16 | 500 | 1.0    | mean        | 42, 123, 456 |
| SupCon last\_token | 1.5B | 16 | 500 | 1.0    | last\_token | 42, 123, 456 |

### 11.3 v2 结果（准确率）

| 配置                   | seed=42 | seed=123 | seed=456 | 均值 ± std       |
| -------------------- | ------- | -------- | -------- | -------------- |
| CE (baseline)        | 21.41%  | 18.97%   | 27.98%   | 22.79% ± 3.79% |
| SupCon + mean        | 22.28%  | 18.97%   | 28.83%   | 23.36% ± 4.07% |
| SupCon + last\_token | 21.80%  | 20.45%   | 28.45%   | 23.57% ± 3.50% |

### 11.4 准确率腰斩根因分析

**对比 v1 同配置（1.5B r=16 seed=42）**：

| 指标          | v1     | v2     | 跌幅      |
| ----------- | ------ | ------ | ------- |
| CE baseline | 43.93% | 21.41% | -22.52% |
| SupCon      | 46.09% | 22.28% | -23.81% |

**根因：batch\_size 8→28 导致有效优化步数从 189→54，LR 未同步调整。**

| 维度                    | v1         | v2          |
| --------------------- | ---------- | ----------- |
| batch\_size           | 8          | 28          |
| steps/epoch           | 500/8 ≈ 63 | 500/28 ≈ 18 |
| total steps (3 epoch) | 189        | 54          |
| learning\_rate        | 2e-4       | 2e-4        |

v1 日志确认：epoch 2 即达到 40.36%，远超 v2 epoch 3 的 22.28%，模型远未收敛。

### 11.5 v2 教训

1. batch\_size 增大时，必须同步调整 num\_epochs 以保持总优化步数
2. SupCon 在未收敛的 baseline 上无法被正确评估——CE 自身都未学会
3. v2 数据不能用于 SupCon 有效性结论

***

## 十二、v3 实验设计：epoch 补偿 + 跨架构对比（code3/）

### 12.1 论文定位修正

论文从"SupCon 是否有效"升级为跨架构效应边界刻画：

> "Decoder vs Encoder：监督对比损失在 LoRA 少样本文本分类中的跨架构效应分析"

核心贡献：

1. 系统刻画 batch\_size × rank × pooling × architecture 的交互效应
2. 在 decoder-only (Qwen) 和 encoder-only (BERT) 上做同条件对比
3. 揭示 seed 方差作为少样本评估中最大的混淆变量

### 12.2 v3 代码改进点（相对 v2）

| 维度              | v2                 | v3                               |
| --------------- | ------------------ | -------------------------------- |
| num\_epochs     | 3                  | 11（≈198 steps，对齐 v1 的 189）       |
| 种子数             | 3                  | 5（42, 123, 456, 789, 1011）       |
| 模型              | 仅 Qwen2.5-1.5B     | Qwen2.5-1.5B + bert-base-chinese |
| 架构支持            | 仅 decoder          | decoder + encoder 自动检测           |
| pooling         | mean / last\_token | mean / last\_token / cls\_token  |
| target\_modules | 写死                 | 自动适配 decoder/encoder             |
| 统计检验            | 无                  | Cohen's d effect size            |
| 可视化             | 单架构                | 跨架构对比图                           |

### 12.3 v3 实验矩阵（25组）

**阶段 1：Decoder（Qwen2.5-1.5B, batch=28, label\_aware, epoch=11）**

| 配置          | lambda | pooling     | 种子数 |
| ----------- | ------ | ----------- | --- |
| CE baseline | 0.0    | mean        | 5   |
| SupCon      | 1.0    | mean        | 5   |
| SupCon      | 1.0    | last\_token | 5   |

**阶段 2：Encoder（bert-base-chinese, batch=28, shuffle, epoch=11）**

| 配置          | lambda | pooling    | 种子数 |
| ----------- | ------ | ---------- | --- |
| CE baseline | 0.0    | cls\_token | 5   |
| SupCon      | 1.0    | cls\_token | 5   |

### 12.4 服务器部署

```bash
PYTHONPATH="/root/autodl-tmp/lunwen/project-a-llm-lora" \
  nohup python -m code3.run_v3 > experiment_v3.log 2>&1 &
```

预估时间：25 组 × 198 steps/组，\~4-5 小时（RTX 3090）。

***

## 十三、项目状态总结（2026-07-14 更新）

| 版本          | 实验数 | 状态          | 结论                            |
| ----------- | --- | ----------- | ----------------------------- |
| v1 (code/)  | 32  | 已完成         | batch=8 限制 SupCon，种子噪声主导      |
| v2 (code2/) | 9   | 已完成（无效）     | 步数不足导致欠拟合，数据不可用               |
| v3 (code3/) | 25  | 代码就绪，待服务器运行 | epoch=11 + cross-architecture |

- **论文方向**：从单一 SupCon 有效性验证 → 跨架构效应边界刻画
- **当前阻塞**：v3 实验等待在服务器上运行

