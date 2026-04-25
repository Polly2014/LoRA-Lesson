---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-family: -apple-system, "PingFang SC", sans-serif; }
  h1, h2 { color: #1a5490; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
  pre { background: #2d2d2d; color: #f8f8f2; border-radius: 6px; }
  .highlight { background: #fffbdd; padding: 2px 6px; border-radius: 3px; }
---

<!-- _class: lead -->

# 第一课 · LoRA 训练原理与手撕实操

### 从全量微调的显存账，到 $W + BA$ 的优雅解法

**对外经贸大学 · 2026 春**
Polly Wang · 3 hours

---

## 今天要回答的三个问题

1. **为什么**全量微调在消费级 GPU 上跑不动？
2. **LoRA 到底改了什么**——用一页数学说清楚
3. **动手**：20 行 PyTorch，让 Qwen2.5-0.5B 跑起来

---

## 课堂时间线（3h）

| 时段 | 模块 | 形式 |
|------|------|------|
| 0:00–0:20 | ① 动机：显存账 + PEFT 全景 | 讲授 |
| 0:20–0:40 | ② 数学：$W + \frac{\alpha}{r}BA$ | 讲授 |
| 0:40–1:10 | ③a 手撕 `LoRALinear` | **敲代码** |
| 1:10–1:40 | ③b 挂到 Qwen2.5-0.5B | **敲代码** |
| 1:40–1:50 | ☕ 休息 |  |
| 1:50–2:20 | ③c Rank + Target Ablation（10 min 讲配置 + 15 min 等训练 + 5 min 讨论） | **敲代码** |
| 2:20–2:50 | ④ SVD 可视化 + 洞察 | 分析 |
| 2:50–3:00 | ⑤ 作业 + Q&A | 收尾 |

> ⏱️ 训练等待时段可用于答疑 / 对比小组参数选择 / 预告作业。

---

# ① 动机：全量微调为什么贵？

---

## 一笔显存账

训练一个 **7B 模型**，fp32 精度，AdamW 优化器：

| 项目 | 占用 | 计算 |
|------|------|------|
| 模型参数 | 28 GB | 7B × 4 bytes |
| 梯度 | 28 GB | 同参数 |
| Adam 一阶动量 | 28 GB | 同参数 |
| Adam 二阶动量 | 28 GB | 同参数 |
| **合计** | **112 GB** | ❌ 消费级 GPU 完全跑不动 |

一张 4090 只有 24GB。这就是为什么必须有 PEFT。

---

## PEFT 家族全景

**PEFT = Parameter-Efficient Fine-Tuning**

| 方法 | 核心思路 | 代表工作 |
|------|---------|---------|
| **Adapter** | 插入小 MLP | Houlsby 2019 |
| **Prefix / Prompt Tuning** | 学一段前缀 token | Li & Liang 2021 |
| **BitFit** | 只训 bias | Zaken 2022 |
| **LoRA** | 低秩分解 $\Delta W$ | **Hu 2021** ⭐ |
| **QLoRA** | 4bit 量化 + LoRA | Dettmers 2023 |

> 今天主角：**LoRA**

---

# ② 数学：LoRA 怎么工作

---

## 核心假设

> **权重更新是低秩的**（Intrinsic Rank Hypothesis）

全量微调得到 $\Delta W \in \mathbb{R}^{d \times k}$，但其"有效信息"其实只占极少维度。

**LoRA 的做法**：强制 $\Delta W$ 分解为两个小矩阵相乘

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k)$$

---

## 前向传播

$$h = W_0 x + \frac{\alpha}{r} \cdot BAx$$

- $W_0$：原始权重，**冻结**（不算梯度）
- $A$：高斯初始化 $\mathcal{N}(0, \sigma^2)$
- $B$：**零初始化** → 训练起点 $\Delta W = 0$，等价于原模型
- $\alpha / r$：缩放因子，防止超参跨 rank 变化

**可训练参数** = $r(d+k)$，vs 原始 $d \times k$

---

## 一个直观例子

Qwen2.5-0.5B 的 `q_proj`：$d_\text{out} \times d_\text{in} = 896 \times 896$

| 方案 | 可训练参数 | 压缩比 |
|------|-----------|-------|
| 全量 | 802,816 | 1× |
| LoRA r=8 | 8 × (896+896) = 14,336 | **56×** |
| LoRA r=1 | 1,792 | **448×** |

**整个模型**：LoRA (QV, r=8) 只训 ~0.16% 参数 ✨

---

## ⚠️ GQA 预警（今天会踩的坑）

Qwen2.5-0.5B 使用 **Grouped Query Attention**：
- `num_attention_heads = 14` （Q 的头）
- `num_key_value_heads = 2`  （K/V 的头，**共享**！）

所以：
- `q_proj.shape = [896, 896]`  → LoRA 参数 = 2r × 896
- `k_proj.shape = [128, 896]`  → LoRA 参数 = r × (128+896)
- `v_proj.shape = [128, 896]`  → 同上

**结论**：**QKVO ≠ QV × 2**。这个现象一会儿你会亲眼看到。

---

# ③ 手撕代码

---

## Part A · 30 行搞定 `LoRALinear`

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False          # 冻结 W0
        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))   # ← B=0 关键
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
```

**➡️ 翻到 Kaggle notebook，Section 4**

---

## Part B · 挂到 Qwen2.5-0.5B

`inject_lora` 递归遍历模型，找到名字匹配的 `nn.Linear` 替换掉：

```python
def inject_lora(model, target_names=("q_proj", "v_proj"), r=8, alpha=16):
    for p in model.parameters(): p.requires_grad = False
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
```

**➡️ Kaggle notebook Section 5–6**

**见证**：`Trainable: 786,432 / 494,032,768 (0.159%)`

---

## Part C · 两个 Ablation

**实验 1 · Rank**（固定 QV, α=16）
`r ∈ {1, 4, 8, 32}` → 画 loss 曲线

**实验 2 · Target**（固定 r=8, α=16）
`QV / QKVO / FFN / ALL` → 对比参数量 + val_loss

➡️ Kaggle notebook Section 12

> 课堂先各跑 2 个配置，剩下作业补齐

---

# ④ 可视化洞察：SVD

---

## 把 $\Delta W$ 打开看看

训练完之后，合并 $\Delta W = \frac{\alpha}{r} BA$，做奇异值分解：

$$BA = U \Sigma V^\top$$

**如果 LoRA 的低秩假设成立**，那么：
- 前几个 $\sigma_i$ 应该远大于其他
- 大部分信息集中在前 ~r 个方向

➡️ Kaggle notebook Section 13

---

## 典型的奇异值谱

```
σᵢ / σ₀  (log scale)
 1.0 ●
      ●
      ⎹ ●
 0.1  ⎹  ●
      ⎹   ● ●
0.01  ⎹       ● ● ●
      ⎹             ● ● ● ● ● ● ● ● ●
      └────┬───────────────────────────────
        前 5–8 个                   后续几乎为零
```

**➡️ 现场跑 Kaggle notebook Section 13 看真图**

**观察**：
- 前 5–8 个奇异值衰减陡峭
- 后面几十个几乎为零
- → $\Delta W$ 的"有效秩" ≈ r，假设成立 ✅

---

# ⑤ 作业 & Q&A

---

## 作业（详见 `homework.md`）

**必做（100 分）**
1. Rank Ablation：`r ∈ {1,4,8,32}` 对比表 + 曲线图（30 分）
2. Target Ablation：QV/QKVO/FFN/ALL 对比（30 分）
3. SVD 分析（≥3 层 + 200 字分析）（30 分）
4. 推理对比（5 条样例）（10 分）

**Bonus（20 分）**：任选跨 rank SVD 对比 / 本地算力挑战 / 第二课数据集预热

**截止**：下节课前一天 23:59

---

## 下节课预告

**第二课：`peft` + QLoRA 实操**

- 模型：Qwen2.5-**1.5B**-Instruct
- 任务：**外贸术语问答**（Incoterms 2020 + HS 编码）
- 数据：Polly 备 500 训练 + 50 盲测
- 工具：`peft` + `trl` + `bitsandbytes`
- 终极目标：**同数据集横向对比各组 LoRA 配置**

---

<!-- _class: lead -->

# Q & A

**一切问题都是好问题。**

> 今日资源：
> - notebook: `lesson1/notebook_kaggle.ipynb`
> - 作业: `lesson1/homework.md`
> - 参考: `RESOURCES.md`
