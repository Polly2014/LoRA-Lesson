# 📎 APPENDIX — LoRA 课程补充知识点

> 课件/notebook 里一笔带过但值得展开的概念，供课后查阅。

---

## A1. 浮点精度（FP32 / FP16 / BF16 / INT8 / INT4）

**FP32 = Floating Point 32-bit**，即 32 位单精度浮点数（IEEE 754 标准：1 位符号 + 8 位指数 + 23 位尾数）。

这是课件里"显存账"的计算基础：每个参数在 FP32 下占 **4 bytes**，所以 7B 模型仅参数就要 7B × 4 = **28 GB**。

| 精度 | 全称 | 位数 | 每参数 | 典型用途 |
|------|------|------|--------|---------|
| FP32 | Float 32 / Single Precision | 32 bit | 4 bytes | 训练默认精度 |
| FP16 | Float 16 / Half Precision | 16 bit | 2 bytes | 混合精度训练 |
| BF16 | Brain Float 16 | 16 bit | 2 bytes | 大模型训练（A100/H100） |
| INT8 | Integer 8-bit | 8 bit | 1 byte | 推理量化 |
| NF4 | 4-bit NormalFloat（QLoRA） | 4 bit | 0.5 byte | QLoRA 训练 |

**FP16 vs BF16**：位数相同但分配不同。BF16 保留了 FP32 的指数范围（8 位），牺牲尾数精度（7 位 vs FP16 的 10 位），因此更不容易溢出，适合大模型训练。

**QLoRA 的核心**：用 NF4（4-bit）量化冻结权重，LoRA 的 A/B 矩阵仍用 FP16/BF16 训练，兼顾显存和精度。

---

## A2. 高斯初始化 vs Kaiming 初始化

LoRA 原论文描述 A 矩阵为"高斯初始化"（Gaussian / 正态分布），但代码实现用的是 **Kaiming uniform**。这不是 bug，是学术界的惯例简化。

**高斯分布 = 正态分布**，同一个东西两个名字：
- "高斯"来自数学家 Carl Friedrich Gauss
- "正态"（Normal）是统计学术语

**Kaiming 初始化**（He et al. 2015）：从均匀分布 $U(-\text{bound}, +\text{bound})$ 采样，其中 $\text{bound} = \sqrt{6 / \text{fan\_in}}$。目的是让每层输出的方差稳定，防止深层网络梯度爆炸/消失。

**教学上说"高斯初始化"没问题**——在 CLT 的意义下，Kaiming 的效果和小方差高斯类似。核心 insight 是 **B=0 保证起点安全**，A 用什么分布是次要的。

---

## A3. GQA（Grouped Query Attention）

Qwen2.5 系列使用 GQA，这会影响 LoRA 参数量计算。

标准 MHA（Multi-Head Attention）：Q/K/V 各有 `n_heads` 个头，维度相同。
GQA：Q 有 `n_heads` 个头，K/V 只有 `n_kv_heads` 个头（多组 Q 共享同一组 K/V）。

Qwen2.5-0.5B 的配置：
- `num_attention_heads = 14`（Q）
- `num_key_value_heads = 2`（K/V，**共享**）

因此：
- `q_proj`: [896, 896] → LoRA 参数 = 2r × 896
- `k_proj`: [**128**, 896] → LoRA 参数 = r × (128 + 896)
- `v_proj`: [**128**, 896] → 同上

**结论**：加上 K/V 的 LoRA 参数比预期少很多，因为输出维度只有 128 不是 896。这就是为什么"QKVO 的参数量 ≠ QV × 2"。

---

## A4. SVD 奇异值分解

SVD（Singular Value Decomposition）将任意矩阵分解为三个矩阵的乘积：

$$M = U \Sigma V^\top$$

其中 $\Sigma$ 是对角矩阵，对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ 叫**奇异值**。

**和 LoRA 的关系**：训练完 LoRA 后，合并得到 $\Delta W = \frac{\alpha}{r} BA$，对它做 SVD 可以观察奇异值的分布。如果前 r 个奇异值远大于其余的，说明 LoRA 的低秩假设（Intrinsic Rank Hypothesis）确实成立——权重更新的"有效维度"很低。

**课件里的实验**：画出 $\sigma_i / \sigma_0$ 的 log-scale 曲线，观察到前 5-8 个奇异值陡降，后续趋近于零。

---

## A5. α/r 缩放因子

LoRA 前向传播：$h = W_0 x + \frac{\alpha}{r} \cdot BA x$

**为什么要除以 r？** 当你改变 rank 时（比如从 r=4 改到 r=8），BA 矩阵的"容量"变了，如果不缩放，学习率需要跟着调。$\alpha/r$ 让你可以固定 $\alpha$（通常等于第一次实验的 r 值），换 rank 时不用重新调学习率。

**实践惯例**：
- 设 $\alpha = r$（缩放因子 = 1），最简单
- 或 $\alpha = 2r$（稍微放大更新幅度）
- HuggingFace PEFT 默认 $\alpha = 8$，和 $r = 8$ 配合时缩放因子 = 1

---

## A6. AdamW 优化器

AdamW（Adam with decoupled Weight decay）是大模型训练的标准优化器。它是 Adam 的改良版，由 Loshchilov & Hutter 2019 提出。

### 核心公式

每一步更新（对每个参数 $\theta$）：

1. **梯度**：$g_t = \nabla L(\theta_{t-1})$
2. **一阶动量**（梯度的指数移动平均）：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3. **二阶动量**（梯度平方的指数移动平均）：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4. **偏差校正**：$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$，$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
5. **参数更新**：$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$

其中：
- $\eta$：学习率（learning rate）
- $\beta_1 = 0.9$：一阶动量衰减系数
- $\beta_2 = 0.999$：二阶动量衰减系数
- $\epsilon = 10^{-8}$：防止除零
- $\lambda$：权重衰减系数（weight decay）

### 为什么占 4 倍显存？

这就是课件"显存账"的来源。对于每个参数，AdamW 需要存储：

| 项目 | 大小 | 说明 |
|------|------|------|
| 参数 $\theta$ | 1× | 模型本身 |
| 梯度 $g$ | 1× | 反向传播计算的梯度 |
| 一阶动量 $m$ | 1× | 梯度的滑动平均 |
| 二阶动量 $v$ | 1× | 梯度平方的滑动平均 |
| **合计** | **4×** | 所以 7B × 4 bytes × 4 = **112 GB** |

### Adam vs AdamW 的区别

**Adam**：权重衰减和梯度更新耦合在一起（$L_2$ 正则化加在 loss 上）
**AdamW**：权重衰减解耦（直接在参数上减，不经过动量）

AdamW 的解耦写法让权重衰减的效果不受 Adam 自适应学习率的干扰，实验证明泛化性能更好。PyTorch 默认的 `torch.optim.AdamW` 就是解耦版本。

---

## A7. 课程用到的模型：Qwen2.5 系列

本课程统一使用阿里通义千问 Qwen2.5 家族，两次课用不同规模：

| | 第一课 | 第二课 |
|---|-------|-------|
| 模型 | Qwen2.5-**0.5B** | Qwen2.5-**1.5B**-Instruct |
| 参数量 | 494M | ~1.5B |
| 类型 | 基座模型（Base） | 指令微调模型（Instruct） |
| 任务 | 情感分类（ChnSentiCorp） | 外贸术语 QA |
| 训练方式 | 手撕 LoRA（FP32） | peft QLoRA（4-bit） |

### Qwen2.5-0.5B 架构参数

| 参数 | 值 |
|------|---|
| 架构 | Transformer Decoder-only |
| 隐藏维度 (d_model) | 896 |
| 层数 | 24 |
| 注意力头数（Q） | 14 |
| KV 头数（GQA） | 2 |
| 每头维度 | 64 |
| 词表大小 | 151,936 |
| 上下文长度 | 128K |
| 模型文件 | ~1 GB（FP32 safetensors） |

### 显存估算

**推理**（仅加载参数）：

| 精度 | 0.5B | 1.5B | 7B |
|------|------|------|-----|
| FP32 | ~2 GB | ~6 GB | ~28 GB |
| FP16 | ~1 GB | ~3 GB | ~14 GB |
| INT4 | ~0.3 GB | ~0.8 GB | ~3.5 GB |

**训练**（参数 + 梯度 + 优化器 + 激活值）：

- 全量微调 FP32：约 **4× 参数** + 激活值（0.5B ≈ 10+ GB，7B ≈ 112+ GB）
- **LoRA 微调**：冻结 99.8%+ 参数，梯度和优化器只需为 LoRA 参数分配，0.5B 在 T4 16GB 上绰绰有余

### Base vs Instruct 的区别

- **Base**（第一课用）：纯预训练模型，只会"接话"（续写），没有对话能力。适合分类等判别任务
- **Instruct**（第二课用）：在 Base 基础上经过指令微调（SFT + RLHF），能理解并遵循指令。适合 QA、翻译等生成任务

### 为什么选 Qwen？

1. **中文原生**：训练数据中文占比高，适合中文任务
2. **家族统一**：0.5B → 1.5B → 7B 架构一致，两次课无认知切换
3. **GQA 架构**：提供了 GQA 这个教学点（QKVO ≠ QV × 2）
4. **免费可用**：Apache 2.0 开源，HuggingFace / ModelScope 均可下载

---

## A8. target_modules 的四种典型配置（QV / QKVO / FFN / ALL）

LoRA 训练中，`target_modules` 决定把 LoRA 加在哪些线性层上。Target Ablation 实验对比这 4 种配置：

| 配置 | 加 LoRA 的层 | 含义 |
|------|-------------|------|
| **QV** | `q_proj`, `v_proj` | 原 LoRA 论文配置，最省参数 |
| **QKVO** | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 整个 Attention 模块全加 |
| **FFN** | `gate_proj`, `up_proj`, `down_proj` | 只加在前馈网络（Feed-Forward Network） |
| **ALL** | Attention 4 个 + FFN 3 个 = 7 个 linear | 全部线性层都加 |

### Transformer Block 结构

每一层 Transformer block 由两大模块组成：

```
输入
 ↓
┌─────────────────────────┐
│  Self-Attention         │  ← QKVO 在这里
│  q_proj, k_proj,        │
│  v_proj, o_proj         │
└─────────────────────────┘
 ↓ + 残差
┌─────────────────────────┐
│  FFN (Feed-Forward)     │  ← FFN 在这里
│  Qwen2.5 用 SwiGLU:     │
│  gate_proj, up_proj,    │
│  down_proj              │
└─────────────────────────┘
 ↓ + 残差
输出
```

### SwiGLU FFN 公式

Qwen2.5 / Llama / Mistral 使用 **SwiGLU** 激活，FFN 由 3 个矩阵组成（不是传统 GPT-2 的 fc1+fc2 两个）：

$$\text{FFN}(x) = \text{down\_proj}\big(\,\text{silu}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x)\,\big)$$

所以加 FFN 类的 LoRA 时，需要同时加 `gate_proj`、`up_proj`、`down_proj` 三个层。

### 为什么 FFN 单独拎出来对比？

**FFN 占模型参数的大头**（通常 60–70%）。Qwen2.5-0.5B 里：

- Attention 部分（QKVO）：约 30%
- FFN 部分（gate / up / down）：约 65%

一个反直觉现象：**只加 FFN（不加 Attention）效果常常比只加 QV 还好**——因为 FFN 才是"知识储存的地方"（参考 Geva et al. 2021，FFN 是 key-value memory）。

### 实验预期

| 配置 | 参数量 | val_loss 预期 |
|------|--------|--------------|
| QV | ~0.05% | 中等 |
| QKVO | ~0.16% | 略好（K/V 便宜，O 加一层） |
| FFN | ~0.5% | 更好（FFN 容量大） |
| ALL | ~0.66% | 最好但性价比最低 |

**教学结论**：加得多 ≠ 性价比高。作业里让学生自己算 `val_loss / 参数量` 比值，找到甜点。

---

## A9. `AutoModelForCausalLM` — HuggingFace 通用加载器

notebook 第 5 节用到 `AutoModelForCausalLM.from_pretrained(...)`，这是 HuggingFace `transformers` 的通用模型加载器。

### 一句话功能

> **给我一个模型名（如 `Qwen/Qwen2.5-0.5B`），我帮你下载权重 + 自动选对应的模型类 + 加上"预测下一个 token"的语言建模头。**

### 拆解三个关键词

**`Auto`** — 自动识别架构。读 `config.json` 里的 `architectures` 字段，自动 import 对应类：

```python
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
# → 内部其实是 Qwen2ForCausalLM

AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# → LlamaForCausalLM
```

好处：换模型只改 `MODEL_NAME`，import 不变。

**`Model`** — Transformer 主干（embedding + 24 层 attention+FFN + LayerNorm）

**`ForCausalLM`** — 主干顶部加一个 **LM Head**（一个 `nn.Linear(hidden_size, vocab_size)`），把 hidden state 投到词表，输出"下一个 token 是哪个词"的 logits。

### `AutoModelXxx` 家族对比

同一个 backbone + 不同输出头适配不同任务：

| 类 | 用途 | 输出头 | 典型任务 |
|----|------|--------|---------|
| `AutoModel` | 只要 backbone | 无 | 抽 hidden states |
| **`AutoModelForCausalLM`** | **生成 / 续写** | **`[hidden → vocab]`** | **GPT、Qwen、Llama** |
| `AutoModelForSeq2SeqLM` | 编码-解码 | encoder + decoder | T5、BART 翻译 |
| `AutoModelForMaskedLM` | 完形填空 | `[hidden → vocab]` | BERT |
| `AutoModelForSequenceClassification` | 分类 | `[hidden → num_labels]` | 情感分类（专用头） |
| `AutoModelForTokenClassification` | 序列标注 | `[hidden → num_labels]` per token | NER |

### 返回的 `model` 对象有 3 个核心方法

| 方法 | 用途 | notebook 里在哪用 |
|------|------|------------------|
| `model(**batch)` | 一次前向，返回 `loss` + `logits` | Section 8/9 训练 |
| `model.generate(...)` | 自回归生成新 token | Section 11 推理对比 |
| `model.parameters()` | 拿所有参数 | Section 6 注入 LoRA、计数 |

### 教学钩子：为什么不用 `AutoModelForSequenceClassification`？

我们做的是**情感二分类**，看起来 `SequenceClassification` 更合适，但课程统一用 `CausalLM`，原因：

1. **统一接口**：第二课要做 QA（生成式），用 CausalLM 一套代码两课都能用
2. **不引入新参数**：分类头会随机初始化一个新的 `[hidden → 2]` linear，等于额外训了个分类器；CausalLM 让模型直接"说出'正面'/'负面'"，复用已有的 LM head
3. **现代 LLM 的范式**：所有任务都转成"续写"，这是 GPT-3 之后的主流做法（in-context learning、instruction tuning 都依赖这个）

**口诀**：分类 vs 生成不重要，重要的是把任务转成 next-token prediction。

### 一行代码看清结构

```python
print(model)
```

打印整个网络——前面是 24 层 `Qwen2DecoderLayer`，最后一行 `lm_head: Linear(in=896, out=151936)` 就是 `ForCausalLM` 加的那个头。

---

## A10. CausalLM 怎么"学会"输出"正面/负面" — Prompt + Target + `-100` Mask

学生最常问的一个问题：**"基座是续写模型，为什么 LoRA 训练完就能输出'正面/负面'两个字？是模型架构改了吗？"**

**答案**：模型架构完全没变，还是续写下一个 token。变的是**训练数据告诉它该续写什么**。

### 一、ChnSentiCorp 数据集本身**只有 0/1**

```python
{"text": "房间又小又脏",     "label": 0}   # 注意 label 是整数
{"text": "服务很好下次还来", "label": 1}
```

**原始数据里没有"正面""负面"这两个汉字**。

### 二、"正面/负面"是 notebook 里**人为定义**的映射

```python
LABEL2TEXT = {0: "负面", 1: "正面"}      # 这是 Polly 写的字典

def __getitem__(self, idx):
    prompt = build_prompt(ex["text"])              # "评论：xxx\n情感："
    target = LABEL2TEXT[ex["label"]] + tok.eos_token   # "负面<eos>"
```

拼出来的训练序列：

```
"评论：房间又小又脏\n情感：负面<eos>"
└──────── prompt ────────┘└─target─┘
```

**"负面"两个字是数据预处理时加上去的，不是数据集自带的。**

### 三、`labels` 用 `-100` 掩盖 prompt — 让 loss 只在 target 上算

这是 **CausalLM 做"指令微调"的核心技巧**：

```python
labels = ([-100] * len(p_ids) + t_ids)[: max_len]
#         └── prompt 段：不算 loss ──┘  └ target ┘
```

| 位置 | input_ids | labels | 说明 |
|------|-----------|--------|------|
| `评`、`论`、`：`... | 正常 token | `-100` | **不算 loss**（prompt 部分） |
| `房`、`间`、`又`、`小`... | 正常 token | `-100` | **不算 loss** |
| `情`、`感`、`：` | 正常 token | `-100` | **不算 loss** |
| **`负`** | 正常 token | **token id of "负"** | ✅ 算 loss |
| **`面`** | 正常 token | **token id of "面"** | ✅ 算 loss |
| **`<eos>`** | 正常 token | token id of eos | ✅ 算 loss |
| pad | pad token | `-100` | 不算 loss |

PyTorch 的 `CrossEntropyLoss` 默认 `ignore_index=-100`，**打 `-100` 的位置梯度为 0**，模型在那些位置无论预测什么都不会被惩罚。

**结果**：1000 条样本 × 2 epoch，每条都在告诉 LoRA：

> "看到 `评论：xxx\n情感：` 这个上下文后，下一个 token 必须是 `正` 或 `负`。"

LoRA 的 A/B 矩阵学会了**在 `情感：` 之后强烈偏好"正面"/"负面"这两组 token**。模型从来没"切换模式"——它一直都在做 next-token prediction，LoRA 改的是 `情感：` 之后的 token 概率分布。

### 四、Prompt + Target 设计的分工

| 部分 | 内容 | 谁定义 |
|------|------|-------|
| 数据集 | text、label（整数） | ChnSentiCorp 原作者 |
| prompt 模板 | `"评论：{text}\n情感："` | 课程作者（在 `build_prompt` 里） |
| target 映射 | `0 → "负面"`、`1 → "正面"` | 课程作者（在 `LABEL2TEXT` 里） |
| 结束符 | `<eos>` | tokenizer 自带 |

四样东西**各管一摊**，组合起来才形成训练序列。

### 五、教学反问 — 揭示 LoRA 学的本质

可以问学生这两个问题，效果极好：

**Q1**: "如果我把 `LABEL2TEXT = {0: '苹果', 1: '香蕉'}` 改成水果，会怎样？"

**A**: 模型一样会乖乖学。看到差评 → 输出"苹果"，好评 → 输出"香蕉"。模型不知道水果跟情感有什么关系，它只学到 prompt 模式 → 输出 token 的纯映射。

**Q2**: "如果我把 `LABEL2TEXT` 标签反了写（`{0: '正面', 1: '负面'}`），会怎样？"

**A**: 模型会**乖乖学反**——给好评输出"负面"，给差评输出"正面"。

**这两个反问让学生彻底明白**：
- 模型对"正面/负面"两个词没有任何先验偏好
- LoRA 学的是 `prompt 模式 → 输出 token` 的纯映射
- **数据告诉它谁对应谁，它就学谁对应谁**

### 六、可换的 target 格式（演示用）

```python
LABEL2TEXT = {0: "负面",   1: "正面"}    # 默认（中文）
LABEL2TEXT = {0: "差评",   1: "好评"}    # 同义中文
LABEL2TEXT = {0: "neg",    1: "pos"}     # 英文
LABEL2TEXT = {0: "👎",     1: "👍"}      # emoji
LABEL2TEXT = {0: "0",      1: "1"}       # 数字字符串
```

全都能 work。给学生改一行重新训，一分钟看到结果。

### 一句话总结

> **数据集只给原料（文本 + 整数标签），怎么把原料拼成 prompt → completion 序列，是你设计的。LoRA 学的是你设计的这个映射。**

这正是 instruction tuning 时代算法工程师的核心活：**Prompt + Completion 的格式设计**。Lesson 2 用 `peft` + 外贸 QA 时会看到更复杂的 prompt 设计（chat template）。

---

## A11. PEFT 家族详解 — Prompt Tuning / Prefix Tuning / LoRA / QLoRA

第二课 slides "2.1 PEFT 家族全景" 那个表格的展开版。学生最容易混淆的是 **Prompt Tuning ≠ Prefix Tuning**，重点讲清这两个的差别。

### PEFT 五大流派一句话总览

| 方法 | 在哪儿动手 | 可训练参数 | 工业界用途 |
|------|----------|-----------|-----------|
| **Prompt Tuning** | Embedding 层（输入最前面加 soft prompt） | ~0.001% | 极小参数 / Edge |
| **Prefix Tuning** | 每层 Attention 的 KV（加可训练 prefix） | ~0.25% | 多任务适配器 |
| **LoRA** | 每个 Linear 的权重更新 $\Delta W = BA$ | ~0.5–1% | **主流（本课）** |
| **QLoRA** | LoRA + 4bit NF4 量化底座 | ~0.5–1% | 大模型微调 |
| **DoRA / AdaLoRA** | LoRA 的改良（幅度/方向解耦、自适应秩） | ~0.5–1% | 前沿研究 |

### Prompt Tuning（最浅）

**做什么**：在输入 token embedding 的最前面，拼上一段可训练的"虚拟 token"向量。

```
标准输入：    [emb(请), emb(解释), emb(FOB)]                          shape: (3, 4096)
Prompt Tuning：[p_0, p_1, p_2, p_3, p_4, emb(请), emb(解释), emb(FOB)]  shape: (8, 4096)
              ↑ 这 5 个可训练，其他全冻结
```

**可训练参数量**：`prefix_len × hidden_size`，例如 5 × 4096 = 20 KB ≈ 0.001%

**损失计算**：soft prompt 段的 label 全设为 `-100`，只对真实输入段算 loss。

```python
labels = torch.cat([
    torch.full((1, 5), -100),     # soft prompt 段：不算 loss
    real_input_ids                  # 真实段：算 loss
], dim=1)
```

**反向传播**：只对 `soft_prompt` 这一个张量求梯度，Transformer 权重完全冻结。

**类比**：给医生贴张"咒语卡片"在胸口，医术本身没变，只是被"提示"了。

**短板**：效果一般。Prompt Tuning 只改"输入上下文"，没动 Transformer 内部，对需要真正改变行为的任务（QA、领域知识）力不从心。小模型上更弱（原论文用 T5-XL 3B 才有效）。

### Prefix Tuning（中等深度）

**做什么**：不是在输入加 prompt，而是在**每一层 Attention 的 K 和 V** 前面加一段可训练的 prefix。

```python
# 每层 Attention 内部
Q = x @ W_q                                                # 正常
K = torch.cat([prefix_k[layer_idx], x @ W_k], dim=1)       # ← 前面加 prefix_k
V = torch.cat([prefix_v[layer_idx], x @ W_v], dim=1)       # ← 前面加 prefix_v
attn = softmax(Q @ K.T / sqrt(d)) @ V
```

**关键**：每层有**独立**的 prefix KV 参数（24 层就有 24 组）。

**可训练参数量**：

$$\text{params} = \text{num\_layers} \times 2 \times \text{prefix\_len} \times \text{hidden\_size}$$

例如 Qwen2.5-1.5B（24 层、4096 维）、prefix_len=20：

$$24 \times 2 \times 20 \times 4096 \approx 3.9\text{M} \approx 0.25\%$$

**比 Prompt Tuning 多 ~250 倍参数，但效果接近全量微调**（SuperGLUE 上只差 0.2 分 vs Prompt Tuning 差 1.7 分，Li & Liang 2021）。

**类比**：医生大脑每个区域（每层 Attention）都被"激活"了，每个脑区有自己的"前缀记忆"——直接影响诊断逻辑，不只是心理暗示。

### LoRA（本课主角）

**做什么**：在每个 Linear 层旁边并联一组低秩矩阵 $A \in \mathbb{R}^{r \times d}$、$B \in \mathbb{R}^{d \times r}$，前向时：

$$h = W_0 x + \frac{\alpha}{r} \cdot B A x$$

$W_0$ 冻结，只训 $A$、$B$。

**可训练参数量**：`2 × r × hidden_size × num_target_modules × num_layers`，r=8 时约 0.5–1%。

**和 Prefix Tuning 的本质区别**：
- Prefix Tuning 改的是 attention 的 **KV 输入**（序列变长）
- LoRA 改的是 **权重本身**（序列不变）

→ LoRA 推理时可以 `merge_and_unload` 把 $\Delta W$ 加回 $W_0$，**零额外开销**；Prefix Tuning 永远要带着 prefix 跑。

### QLoRA（LoRA 的省显存版）

LoRA 的扩展：底座先 4bit NF4 量化（冻结、不参与梯度），LoRA 的 A/B 矩阵仍用 fp16/bf16 训练。

**适用场景**：单卡 GPU 想训大模型（7B / 13B / 70B）。本课在 1.5B 用 QLoRA 主要是**教学统一**（让 Kaggle T4 体验完整流程），1.5B 其实直接 fp16 LoRA 也跑得下。

### 为什么本课选 LoRA/QLoRA 而不是 Prefix Tuning？

| 维度 | Prefix Tuning | LoRA | 选 LoRA 的原因 |
|------|--------------|------|---------------|
| 可解释性 | "prefix 记忆"抽象 | $\Delta W = BA$ 数学清晰 | 第一课已铺垫 |
| 代码 | 要 hack Attention 内部 | `peft` 一行配置 | 工程简单 |
| 推理 | 永远要带 prefix | 可合并回底座 | 部署友好 |
| 社区 | 实现少 | HuggingFace / Unsloth / vLLM 全支持 | 生态成熟 |

### 学生常见追问

**Q: 那 Prompt Tuning / Prefix Tuning 还有用吗？**

A: 有，但场景窄。Prompt Tuning 适合**多任务热切换**——给每个任务存一套 soft prompt（20 KB），推理时根据任务切换 embedding 拼接。LoRA 也能做（adapter 几十 MB），但 Prompt Tuning 更轻。

**Q: Prefix Tuning 既然效果好，为什么没火起来？**

A: 三个原因：(1) 推理时序列变长，KV cache 永远多带一段；(2) 实现要 hack Transformer，没有 LoRA 那种"挂适配器"的优雅；(3) LoRA 后来居上，效果相当 + 工程简单。

---

## A12. NF4 量化深入 — 为什么是"Normal Float"

A1 给了 NF4 在精度家族中的位置。这里展开 **NF4 vs INT4** 的本质差异，以及"Normal"的含义。

### 全称对照

| 缩写 | 全称 | 中文 |
|------|------|------|
| **INT4** | **Integer 4-bit** | 4 位整数（均匀量化） |
| **NF4** | **Normal Float 4-bit** | 4 位"正态浮点"（非均匀量化） |

**"Normal"指正态分布（Normal Distribution）**，不是"普通"。

### 关键观察 — 神经网络权重接近正态分布

Dettmers 2023 的核心 insight：**预训练完的 Transformer 权重高度服从** $\mathcal{N}(0, \sigma^2)$。

```
频率
  |       ╱╲
  |      ╱  ╲       ← 权重主要集中在 0 附近
  |     ╱    ╲
  |    ╱      ╲
  |___╱________╲___
      -σ  0  +σ
```

如果用 INT4 那样**等距分割**，大部分量化等级都浪费在权重很少出现的区域（两端），中间密集区反而粒度太粗 → 精度损失大。

### INT4 vs NF4 量化等级对比

```
INT4（均匀等距 16 等级）：
  -1.00, -0.87, -0.73, -0.60, -0.47, -0.33, -0.20, -0.07,
   0.07,  0.20,  0.33,  0.47,  0.60,  0.73,  0.87,  1.00
  ↑ 等距 0.13，中间和两端粒度一样

NF4（按正态分布分位数 16 等级）：
  -1.00, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
   0.080, 0.161,  0.246,  0.338,  0.441,  0.563,  0.723, 1.0
  ↑ 中间密集（粒度细）、两端稀疏（粒度粗）
```

### NF4 的设计原理（一句话）

> NF4 的 16 个量化等级，是 $\mathcal{N}(0,1)$ 的 **16 个等概率分位点**。

→ 每个量化"桶"里落进去的权重数量**大致相等**，信息利用率达到理论最优。

### 通俗比喻

- **INT4** = 用统一尺寸的箱子装水果（常见水果的箱子堆满了，少见水果的箱子空着）
- **NF4** = 根据水果分布调整箱子大小（常见水果用小箱子分细，少见水果用大箱子粗放）

### 实验效果（Dettmers 2023）

在 LLaMA-65B 上，NF4 量化后的 5-shot MMLU 分数与 fp16 几乎无差距（差距 < 0.5%），而 INT4 会掉 2–3 个点。这就是 QLoRA 论文最重要的实验结论之一。

### 一句话给学生

> "INT4 把直线均匀切 16 段，NF4 按正态分布的'人口密度'切——常见区域切得细、少见区域切得粗，4 个 bit 用得最值。"

---

## A13. 双重量化（Double Quantization）— "量化常数本身也量化"

A1 提了一句"双重量化"，这里展开。这是 QLoRA 三大技术里**最容易跳过、但最体现工程巧思**的一项。

### 问题：量化常数本身占显存

NF4 量化不是单纯把 fp16 砍成 4bit。每**一组**权重（通常 64 个）共享一个 **scaling factor**（缩放常数），用来恢复原始数值范围：

```python
# 伪代码
group_max = max(abs(weights[i*64 : (i+1)*64]))    # 这一组的最大绝对值
scale = group_max / 7                              # 4bit 有符号能表示 -8 ~ +7
quantized = round(weights / scale).clip(-8, 7)     # 量化到 INT4
# 反量化时：dequant = quantized × scale
```

**这个 `scale` 必须存着**，否则反量化时不知道还原成多大。

**显存账（Qwen2.5-1.5B）**：

```
权重总数：         1.5 B
分组大小：         64
组数：             1.5B / 64 = 23.4 M 组
每组 scale 占用：  fp16 = 2 byte
量化常数总占用：   23.4M × 2 byte ≈ 47 MB
```

权重本身才 0.75 GB，光量化常数就要 47 MB —— **占比 6%**，不容忽视。

### 解决方案：对 scale 再做一次 8bit 量化

```
第一层量化：
  权重 (fp16) ─NF4量化─→ 权重_4bit + scale_1 (fp16)
                                     ↓
第二层量化：把这堆 scale_1 再分组（每 256 个一组）
  scale_1 (fp16) ─INT8量化─→ scale_1_int8 + scale_2 (fp32)
```

每 256 个 scale_1 共享一个 scale_2，所以 scale_2 数量极少（23.4M / 256 ≈ 91K 个，占用 < 1 MB）。

### 节省效果

| | 一次量化 | 双重量化 |
|---|---------|---------|
| 权重 | 0.75 GB | 0.75 GB |
| 第一层 scale (fp16) | **47 MB** | 23 MB (压成 INT8) |
| 第二层 scale (fp32) | — | 0.4 MB |
| **合计** | **~800 MB** | **~775 MB** |
| **节省** | — | **~25 MB（≈ 3%）** |

数字看着不大，但放到 70B 模型上能省 1 GB+。"省下的每一 MB 都能让 batch_size 多 1"是大模型工程的金科玉律。

### 通俗比喻

- 一次量化：把大象装进冰箱（权重 → 4bit），但冰箱旁边还堆着一堆说明书（scale）
- 双重量化：连说明书也压缩成卡片（scale → INT8）

### 一句话给学生

> "权重量化了，**记录怎么还原权重的那个 scale 也量化一次**——这就叫双重量化。技巧虽小，但体现了 QLoRA '榨干每一 MB' 的工程哲学。"

---

## A14. Paged Optimizer — 借用 OS 虚拟内存思想

QLoRA 三技术的最后一项。这是 `bitsandbytes` 提供的、`optim="paged_adamw_8bit"` 背后的机制。

### 问题：训练时显存峰值经常爆掉

QLoRA 把权重压到 0.9 GB、LoRA 参数+梯度+optimizer state 也只占 100 MB 左右，但**激活值缓存（activations）** 在长序列时会膨胀：

```
激活值 ≈ batch_size × seq_len × hidden_size × num_layers × bytes_per_value × overhead

例：batch=2, seq_len=2048, hidden=4096, layers=24, fp16
   = 2 × 2048 × 4096 × 24 × 2 byte
   ≈ 1.6 GB（不开 gradient checkpointing）
```

如果某个 batch 的序列特别长，激活值瞬间冲到 5–6 GB，整个显存就到红线了。一旦 OOM —— **训练直接 crash**，前面跑的全白费。

### 解决方案：把 optimizer state 分页，溢出时换到 CPU

灵感来自操作系统的**虚拟内存（virtual memory）**：进程的内存看起来是连续大的，实际由 OS 分页（page）管理，物理内存不够时把不活跃的页换到磁盘（swap）。

Paged Optimizer 把 AdamW 的 `m`（一阶动量）、`v`（二阶动量）分成小页（page），驻留在 GPU 显存。当 GPU 显存吃紧（如激活值突然变大）：

```
正常时：
  GPU: [模型(0.9G) | LoRA+grad(0.1G) | optim_state(0.1G) | activations(4G)] ≈ 5.1G ✓

序列变长瞬间：
  需要 activations 6G，但 GPU 只剩 5G
  ↓
  Paged Optimizer 触发：
  把 optim_state 的 0.05G 换到 CPU 内存
  ↓
  GPU: [模型(0.9G) | LoRA+grad(0.1G) | optim_state(0.05G) | activations(6G)] ≈ 7.05G ✓
  CPU 内存: [optim_state_swapped(0.05G)]
```

需要那部分 optimizer state 时，再从 CPU 读回 GPU。**整个过程对用户透明，靠 NVIDIA Unified Memory 在底层做 page fault 处理。**

### 实际配置（notebook 里就一行）

```python
TrainingArguments(
    ...
    optim="paged_adamw_8bit",     # ← 这就开启了 Paged Optimizer
    ...
)
```

`paged_adamw_8bit` 还顺带做了 optimizer state 的 8bit 量化（m、v 从 fp32 压到 INT8），双管齐下进一步省显存。

### 副作用

- **轻微变慢**：CPU↔GPU 数据传输有延迟，比纯 GPU 慢 5–15%
- **要求 NVIDIA Unified Memory**：旧卡（K80、P40）可能不支持
- **CPU 内存够大**：通常需要 ≥ 32 GB 系统内存

### 通俗比喻

- 标准 AdamW：只用桌子放东西，桌子满了就摔东西（OOM crash）
- Paged Optimizer：桌子 + 旁边柜子（CPU 内存），桌子满了把不常用的塞柜子里，要用再拿回来 —— **不会 crash**

### 一句话给学生

> "训练的核心痛点是'激活值突发膨胀导致 OOM'。Paged Optimizer 借 OS 虚拟内存的思想，把 optimizer state 当成'可换页内存'，显存吃紧时自动溢出到 CPU，平稳渡过峰值。"

---

## A15. QLoRA 显存账 — 6GB 是怎么算出来的

第二课 slides "2.3 为什么能压到这么小" 的表格反推，给学生完整推导。

### 总账对比

| 存储项 | 全量微调 (fp16) | QLoRA |
|-------|----------------|-------|
| 模型参数 | 3 GB (fp16) | 0.9 GB (4bit NF4 + 双重量化) |
| 梯度 | 3 GB | ~35 MB（只对 LoRA） |
| AdamW state (m + v) | 6 GB（2×参数，fp32） | ~70 MB（只对 LoRA） |
| 激活值（seq=512, bs=2） | ~4 GB | ~4 GB |
| **合计** | **~16 GB** | **~5–6 GB** ✓ |

→ Kaggle T4（16 GB）全量 1.5B 都已经勉强（只剩 0 GB 余量，长序列必爆），QLoRA 留出 10 GB 余量，**可以舒舒服服训 7B 模型**。

### QLoRA 5GB 的详细拆解

#### 1. 量化基座（冻结，不训练）

```
Qwen2.5-1.5B = 1.55B 参数
NF4 量化：    1.55B × 4 bit / 8 = 0.775 GB
+ 双重量化后的 scale 常数：~25 MB
────────────────────────────────
合计：         ~0.9 GB
```

#### 2. LoRA 可训练部分（fp16）

```
target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # 7 个
r = 8

每个 Linear 的 LoRA 参数：
  A 矩阵：(r, hidden_in)  = (8, 4096)   → 32 K 参数
  B 矩阵：(hidden_out, r) = (4096, 8)   → 32 K 参数
  小计：~64 K 参数 × 2 byte (fp16) = 128 KB

⚠ 注意 GQA：k_proj/v_proj 的 hidden_out 只有 256（不是 4096），所以参数更少。
   这里只算粗估。

全部 LoRA 总参数：
  ≈ 24 层 × 7 modules × 平均 50K params ≈ 8.4 M 参数
  
显存占用：8.4 M × 2 byte (fp16) ≈ 17 MB
（PEFT 实际打印通常显示 8.8M / 1.55B ≈ 0.566%）
```

#### 3. 梯度（只对 LoRA）

```
LoRA 参数有梯度，量级和参数本身相同：
  ≈ 17 MB
```

#### 4. AdamW Optimizer State

```
AdamW 对每个可训练参数存 m（一阶动量）+ v（二阶动量）：
  paged_adamw_8bit：m + v 压成 INT8 → 8.4M × 1 byte × 2 = ~17 MB
  普通 adamw_fp32： m + v 用 fp32 → 8.4M × 4 byte × 2 = ~67 MB

我们用 paged_adamw_8bit：~17 MB
```

#### 5. 激活值（**显存大头**）

这是不容易省的部分，公式：

```
activations ≈ batch × seq_len × hidden × num_layers × dtype × overhead
            ≈ 2 × 512 × 4096 × 24 × 2 byte × 1.5
            ≈ 600 MB（开 gradient checkpointing）

不开 gradient checkpointing 会膨胀到 ~4 GB。
QLoRA 必开 gradient_checkpointing=True。
```

#### 总账

```
量化基座            ≈ 0.9 GB
LoRA 参数            ≈ 17 MB
LoRA 梯度            ≈ 17 MB
AdamW state (8bit)   ≈ 17 MB
激活值 (with ckpt)   ≈ 0.6–1.5 GB（动态）
临时 buffer / fragment ≈ 0.5 GB
─────────────────────────────────
合计                 ≈ 2–4 GB（小 batch）
                       4–6 GB（大 batch / 长序列）
```

### Slides 表格里"4 GB 激活值"的口径

Slides 上写的是 **不开 gradient checkpointing** 的口径（让对比更明显）。实际跑 notebook 会开 `gradient_checkpointing=True`，激活值会进一步降到 1 GB 量级。

### 学生常见追问

**Q: 那为什么不全量微调 1.5B？反正只占 16 GB。**

A: (1) 16 GB 完全占满 = 0 余量，任何长序列、batch 微调都会 OOM；(2) 全量微调改了模型的所有权重，**很容易破坏底座原有能力**（catastrophic forgetting），1500 条数据训完可能连基础对话都不会了；(3) LoRA 只动 0.5% 参数，**有强力的正则化效果**，泛化更好。

**Q: QLoRA 能训多大模型？**

A: Kaggle T4 (16 GB) → 舒适训 7B、勉强训 13B；A100 (80 GB) → 舒适训 70B（QLoRA 原论文就是这么干的）。

### 一句话给学生

> "全量微调 = 模型 1× + 梯度 1× + AdamW 2× = **4× 参数显存**；QLoRA = 基座 0.3× + LoRA 部分忽略不计 + 激活值 = **0.3–0.5× 参数显存**。这就是为什么 T4 能跑 7B、A100 能跑 70B。"
