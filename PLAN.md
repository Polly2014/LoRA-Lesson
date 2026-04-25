# LoRA 训练原理与实操 · 课程规划

> **对象**：对外经贸大学学生（非 CS 专业为主）
> **形式**：两次课，每次约 3 小时
> **目标**：从数学原理 → 手撕代码 → 框架实战，打通 LoRA 全链路
> **规划版本**：v0.3（Polly + 小龙虾 Final Review 通过，Batch 1 开工）

---

## 🦞 v0.2 变更摘要（vs v0.1）

| 决策点 | v0.1 | v0.2（final） |
|--------|------|--------------|
| 第二课部署环节 | 30min 现场实操 | **15min 纯 demo**（部署移课后选做） |
| 第二课任务 | 各组自选领域 | **统一指定：外贸术语 QA（Incoterms + HS 编码）**，Polly 备 500 训练 + 50 测试 |
| 模型选型 | Qwen2.5-1.5B / Phi-3-mini / DeepSeek-MoE 待定 | **Qwen2.5-1.5B-Instruct（定）** |
| 第一课 base 模型 | GPT-2 small (124M) | **Qwen2.5-0.5B（统一家族）** |
| 第一课 LoRA target | 未定 | **主线挂 `q_proj` + `v_proj`**（按原论文）；ablation 对比 QV / QKVO / FFN / 全部 |
| Checkpoint 策略 | 模糊 | **每阶段存 Kaggle Dataset 快照**，掉队可从任意断点接上 |
| 可解释性评分 | "30%" 空描述 | **明确 checklist**（SVD 图 / rank 表 / target_modules 表 / 文字分析） |

---

## 0. 学情与算力诊断

| 小组 | 可用算力 | 实际能跑什么 | 课堂定位 |
|------|---------|-------------|---------|
| 组 1 | 个人笔记本 / Kaggle 免费 GPU | Kaggle T4 x2 (16GB×2)，每周 30h | **Kaggle 基线** |
| 组 2 | 5060 laptop (8GB) / 5070 Ti (16GB) / M4 Pro 48GB | 本地可跑 1.5B–7B QLoRA；Mac 走 MLX | **本地加餐** |
| 组 3 | 个人笔记本 + 微信云函数 | 云函数无 GPU，实际靠 Kaggle | **Kaggle 基线** |

**统一基线**：**Kaggle Notebook（T4）**——零配置、免费、可复现，组 2 本地算力仅作加餐。

---

## 1. 课程总设计

```
第一课（3h）                           第二课（3h）
━━━━━━━━━━━━━━━━━━                   ━━━━━━━━━━━━━━━━━━
原理 + 手撕代码                        框架实操（外贸术语 QA）
（理解"LoRA 为什么 work"）              （聚焦训练→合并→推理主线）

 ┌──────────────────┐                   ┌──────────────────┐
 │ 动机 & 数学 40min │                   │ PEFT+QLoRA 15min │
 │ 手撕 LoRA  90min  │   ───作业一周───► │ 数据/训练  75min  │
 │ 可视化洞察 30min  │                   │ 合并+推理  30min  │
 │ 作业布置   20min  │                   │ 部署 demo  15min  │
 │                  │                   │ 组间评比   30min  │
 └──────────────────┘                   └──────────────────┘

   Qwen2.5-0.5B + 纯 PyTorch             Qwen2.5-1.5B-Instruct + QLoRA
   (不用 peft，手写 LoRALinear)           (peft + trl + bitsandbytes)
```

**模型家族统一为 Qwen2.5**——学生两节课不用切换认知上下文。

---

## 2. 第一次课：原理 + 手撕 LoRA

### 2.1 教学目标

- 理解**为什么全量微调不可行**（显存账）
- 理解**低秩假设**（权重更新的内在维度远小于参数维度）
- 能用 **~30 行 PyTorch** 实现 `LoRALinear` 并挂到 `q_proj` / `v_proj`
- 能**打印/对比可训练参数量**，感受 0.5%–2% 的压缩
- 能做 **rank ablation + target_modules ablation**，看到配置对效果的影响

### 2.2 模块安排

| 模块 | 时长 | 内容要点 |
|------|------|---------|
| ① 动机 | 20min | 全量微调的显存账（7B × 4 ≈ 112GB）；PEFT 家族全景；LoRA 的位置 |
| ② 数学 | 20min | $W_0 + \Delta W = W_0 + BA$；$\alpha/r$ 缩放；$B=0$ 初始化的含义；参数量对比 |
| ③ 手撕代码 | 90min | 纯 `torch` 实现 `LoRALinear`；**挂到 Qwen2.5-0.5B 的 `q_proj` / `v_proj`**；中文情感分类；可训练参数打印；**两个 ablation（rank + target_modules）** |
| ④ 可视化洞察 | 30min | 对 $BA$ 做 SVD，画奇异值谱；对比 QV / QKVO / FFN / 全部的效果差异；讨论层间差异 |
| ⑤ 作业布置 | 20min | 跑通手写 LoRA + 提交 ablation 报告 |

### 2.3 关键代码骨架（课件核心）

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base_linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base_linear.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def inject_lora(model, target_names=("q_proj", "v_proj"), r=8, alpha=16):
    """遍历 model，把符合 target_names 的 nn.Linear 替换成 LoRALinear"""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
    return model
```

**灵魂环节**：让学生打印
```
Trainable: 786,432 / 494,032,768 (0.159%)
```

### 2.4 演示任务

- **Base**：**Qwen2.5-0.5B**（统一家族，中文效果比 GPT-2 好很多）
- **任务**：ChnSentiCorp 中文情感二分类（小样本 2000 条）
- **环境**：Kaggle T4，零配置
- **Checkpoint**：每个 ablation 阶段保存 `.pt` 到 Kaggle Dataset，掉队学生可从任意断点接上

### 2.5 Ablation 设计（Section ③ + ④ 联动）

| 实验 | 变量 | 固定 |
|------|------|------|
| Rank ablation | r ∈ {1, 4, 8, 32} | target = `q_proj`, `v_proj`, α=16 |
| Target ablation | QV / QKVO / FFN / 全部 | r=8, α=16 |

每组实验要求产出**一张对比表 + 一张 loss 曲线图**。

### 2.6 作业（一周）

1. **必做**：跑通手写 LoRA，提交 `.ipynb`，含两个 ablation 表
2. **必做**：SVD 奇异值谱可视化 + 一段文字分析（200 字）
3. **必做**：阅读第二课的统一数据集（外贸术语 QA），理解任务
4. **选做（组 2）**：用 M4 Pro 的 MLX 或 5070 Ti 本地跑同样实验，对比速度

---

## 3. 第二次课：框架实操（外贸术语 QA）

### 3.1 教学目标

- 掌握 🤗 `peft` 的 `LoraConfig` + `get_peft_model` + `merge_and_unload` 三板斧
- 理解 **QLoRA**（4bit 量化 + LoRA）为什么能在 16GB T4 上跑 1.5B 模型
- 走完**完整主线**：数据加载 → SFT 训练 → 合并权重 → 推理对比
- 在**同数据、同测试集**条件下横向对比不同 LoRA 配置的效果

### 3.2 统一任务（定）

**外贸术语问答（Incoterms 2020 + HS 编码）**

- **Polly 提供**：
  - `train.jsonl`：500 条清洗好的 QA（Alpaca 格式）
  - `test.jsonl`：50 条盲测集（课堂评分用）
  - `build_dataset.py`：数据生成脚本（GPT-4 合成 + 规则校验，供学生阅读）

**为什么统一指定**：
- 非 CS 学生一周内从零构造 500 条有质量的数据集不现实
- 同数据集 → 评分有横向可比性
- 学生精力聚焦 LoRA 本身，不是在当数据标注员
- 各组差异化来自 **LoRA 配置**（rank / target_modules / alpha / lr），而非数据

**评分锚点**：盲测集上的 BLEU / ROUGE-L + 组间人工盲评。

### 3.3 工具栈选型（照顾异质算力）

| 场景 | 工具栈 | 模型 | 适配算力 |
|------|-------|------|---------|
| **主线（全员）** | `peft` + `transformers` + `trl` + `bitsandbytes` (QLoRA 4bit) | **Qwen2.5-1.5B-Instruct** | Kaggle T4 够用 |
| **加餐 A** | 同上 + `unsloth`（快 2x，省 70% 显存） | **Qwen2.5-7B-Instruct** | 5070 Ti / 16GB+ |
| **加餐 B（Mac）** | `mlx-lm` + `mlx-lm.lora` | Qwen2.5-7B (MLX 量化版) | M4 Pro 48GB |
| 微信云函数 | 不用于训练，仅做**推理部署**演示（课后选做） | 合并后的 adapter | 组 3 的差异化亮点 |

### 3.4 模块安排（v0.2 时间重分配）

| 模块 | 时长 | 内容要点 |
|------|------|---------|
| ① PEFT + QLoRA 原理 | 15min | `peft` 三板斧；`target_modules` 选择；QLoRA 一页纸 |
| ② 数据加载 + `SFTTrainer` 配置 | 20min | 加载 `train.jsonl`；prompt 模板；`SFTConfig` 要点 |
| ③ 训练（含 buffer） | 55min | 跑起来；看 loss；处理常见报错；**checkpoint 可接力** |
| ④ 合并 + 推理对比 | 30min | `merge_and_unload`；微调前后在测试集上的对比；BLEU/ROUGE 打分 |
| ⑤ 部署 demo（老师演示） | 15min | M4 Pro 现场 `ollama` / 云函数推理 / adapter 上传 HF Hub；**学生不现场跑** |
| ⑥ 组间展示评比 | 30min | 每组 5min 展示 LoRA 配置 + 盲测分 + 失败案例；盲评 |

**总计 165min，留 15min 机动**（处理现场环境问题）。

### 3.5 评分（明确可操作）

| 维度 | 权重 | 可操作 Checklist |
|------|------|-----------------|
| **参数量压缩** | 30% | ☐ 可训练参数占比打印 ☐ adapter 文件大小 ☐ 与全量 FT 的显存对比估算 |
| **任务效果** | 40% | ☐ 盲测集 BLEU ☐ 盲测集 ROUGE-L ☐ 组间人工盲评得分 |
| **可解释性** | 30% | ☐ rank ablation 表 ☐ target_modules ablation 表 ☐ SVD 奇异值图 ☐ 200 字失败案例分析 |

**强制做 ablation** + **checklist 明示给分点** → 非 CS 学生也能清楚怎么拿分。

---

## 4. 交付物清单

### 4.1 第一次课

- [ ] `lesson1/slides.md`：课件（Markdown，可用 Marp/Slidev 转 PDF）
- [ ] `lesson1/notebook_kaggle.ipynb`：主线 notebook（自动适配 Kaggle CUDA / Mac MPS / CPU，Qwen2.5-0.5B 手撕 LoRA）
- [ ] `lesson1/checkpoints/`：各阶段 `.pt` 快照（上传 Kaggle Dataset）
- [ ] `lesson1/homework.md`：作业说明 + 提交格式
- [ ] `lesson1/solution.ipynb`：参考答案（课后发）

### 4.2 第二次课

- [ ] `lesson2/slides.md`：课件
- [ ] `lesson2/notebook_qlora_kaggle.ipynb`：Qwen2.5-1.5B QLoRA 主线
- [ ] `lesson2/notebook_mlx_mac.ipynb`：MLX 版本（Mac 加餐）
- [ ] `lesson2/notebook_unsloth_local.ipynb`：unsloth 版本（5070Ti 加餐）
- [ ] `lesson2/data/train.jsonl`：**500 条外贸术语 QA**（Polly 备）
- [ ] `lesson2/data/test.jsonl`：**50 条盲测集**（Polly 备）
- [ ] `lesson2/data/build_dataset.py`：数据生成脚本
- [ ] `lesson2/eval.py`：BLEU/ROUGE + 盲评脚本
- [ ] `lesson2/checkpoints/`：各阶段 adapter 快照
- [ ] `lesson2/homework.md`：最终项目提交要求 + 评分 checklist

### 4.3 公共资源

- [ ] `README.md`：课程总览 + 环境配置
- [ ] `PLAN.md`：本文件
- [ ] `TROUBLESHOOTING.md`：Kaggle/Colab/Mac/Windows 常见坑
- [ ] `RESOURCES.md`：延伸阅读（LoRA/QLoRA/DoRA/AdaLoRA 论文）
- [ ] `LICENSE`：MIT or CC-BY

---

## 5. 文件结构预案

```
X-Workspace/LoRA-Lesson/
├── README.md
├── PLAN.md                     # 本文件
├── TROUBLESHOOTING.md
├── RESOURCES.md
├── LICENSE
├── lesson1/
│   ├── slides.md
│   ├── notebook_kaggle.ipynb   # 自动适配 cuda/mps/cpu
│   ├── checkpoints/
│   ├── homework.md
│   └── solution.ipynb
├── lesson2/
│   ├── slides.md
│   ├── notebook_qlora_kaggle.ipynb
│   ├── notebook_mlx_mac.ipynb
│   ├── notebook_unsloth_local.ipynb
│   ├── data/
│   │   ├── train.jsonl           # 500 条（Polly 备）
│   │   ├── test.jsonl            # 50 条盲测（Polly 备）
│   │   └── build_dataset.py
│   ├── checkpoints/
│   ├── eval.py
│   └── homework.md
└── assets/
    ├── lora_diagram.png
    ├── svd_spectrum.png
    └── memory_budget.png
```

---

## 6. 时间节奏（每次课 3h）

### 第一次课（v0.3：收紧到 3h 整）

```
00:00 ─┬─ ① 动机（全量微调贵在哪里）        20min
00:20 ─┼─ ② 数学（W + BA, α/r, init）       20min
00:40 ─┼─ ③ 手撕 Part A（LoRALinear 实现）  30min
01:10 ─┼─ ③ 手撕 Part B（挂到 Qwen 0.5B）   30min
01:40 ─┼─ ☕ 休息（合并一次）                10min
01:50 ─┼─ ③ 手撕 Part C（rank + target ablation） 30min
02:20 ─┼─ ④ 可视化洞察（SVD + ablation 对比）30min
02:50 ─┴─ ⑤ 作业布置 + Q&A                  10min
```

**注**：作业详情放 `homework.md` 让学生课后看，课堂 10min 只讲提交格式和 deadline。

### 第二次课（v0.2 重分配）

```
00:00 ─┬─ ① PEFT + QLoRA 原理              15min
00:15 ─┼─ ② 数据加载 + SFTTrainer 配置      20min
       │
00:35 ─┼─ ☕ 休息                           10min
       │
00:45 ─┼─ ③ 训练 Part A（跑起来 + debug）   30min
01:15 ─┼─ ③ 训练 Part B（调参观察）         25min
01:40 ─┼─ ☕ 休息                           10min
01:50 ─┼─ ④ 合并权重 + 推理对比 + 盲测打分   30min
       │
02:20 ─┼─ ⑤ 部署 demo（老师演示）           15min
       │
02:35 ─┴─ ⑥ 组间展示评比                    30min
```

---

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| Kaggle 账号验证 / 网络问题 | 课前一周布置注册 + 跑 hello-world notebook |
| 下载 HF 模型超时 | **提前做 Kaggle Dataset 镜像**（Qwen2.5-0.5B + 1.5B）；HF 镜像 `HF_ENDPOINT=https://hf-mirror.com` |
| bitsandbytes 安装失败（Mac/Windows） | Mac 走 MLX 分支；Windows 统一用 Kaggle |
| 数据集质量不齐 | **Polly 统一备 500+50 条**，学生不碰数据 |
| 课堂时间不够 debug | **checkpoint notebook** + 第二课 buffer 15min |
| 环境冲突拖课 | 第二课部署只做 demo，不让学生现场跑部署 |
| 学生数学基础弱 | SVD 可视化做成"看图说话"，不推导证明 |
| Ablation 没人做 | **评分 checklist 明示**每项给分点 |
| **GQA 导致 ablation 参数量反直觉** 🦞 | Qwen2.5-0.5B 用 **GQA**（K/V 头数少于 Q），K/V 维度比 Q 小。notebook 里**加一个 cell 打印各 proj 层 shape** 并科普 GQA，避免学生看到 QKVO 的可训练参数不是 QV 的 2× 而困惑 |

---

## 8. 延伸（可选）

- **LoRA Hub**：一个 base + N 个 adapter 热插拔的架构讨论
- **DoRA / AdaLoRA / VeRA**：LoRA 家族最新进展（5min 概念介绍）
- **Merge 技巧**：TIES / DARE / Model Soups（课后资料）
- **安全议题**：LoRA 被用于越狱攻击的案例（AI 安全讨论题）
- **数据加餐**：学生自己扩充外贸 QA 数据集作为 bonus 分

---

## 9. 开工清单

Review 通过 ✅，下一步分两批交付：

### Batch 1：第一次课全部材料 ✅ **已交付**
- [x] `lesson1/slides.md`
- [x] `lesson1/notebook_kaggle.ipynb`（Qwen2.5-0.5B + QV LoRA + 两个 ablation，自动适配 cuda/mps/cpu）
- [x] `lesson1/homework.md`
- [x] `lesson1/solution.ipynb`

### Batch 2：第二次课全部材料 + 公共资源 ✅ **已交付**
- [x] `lesson2/data/build_dataset.py`（CopilotX API 合成 + seeds.jsonl fallback，已 smoke test）
- [x] `lesson2/data/seeds.jsonl`（20 条人工精写 Incoterms + HS QA）
- [x] `lesson2/data/README.md`（使用说明）
- [x] `lesson2/notebook_qlora_kaggle.ipynb`（MODE 分支：qlora / fp16_lora / fp32_lora）
- [x] `lesson2/eval.py`（BLEU + ROUGE-L CLI 打分）
- [x] `lesson2/homework.md`
- [x] `lesson2/slides.md`
- [x] `lesson2/checkpoints/README.md`
- [x] `TROUBLESHOOTING.md` / `RESOURCES.md`（repo 根）
- ⚠️ `notebook_mlx_mac.ipynb` / `notebook_unsloth_local.ipynb`：**合入主 notebook 的 MODE 分支**，不再单独出文件

### 完成后
- 本地跑通所有 notebook（Polly 自测）
- Polly 推到新建 GitHub Repo
- 发给贸大老师预审

---

> Plan v0.2 locked. 🦞 Ready to ship Batch 1. 🚀
