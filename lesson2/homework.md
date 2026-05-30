# 第二课作业 · 外贸术语 QA LoRA 微调

> **截止时间**：下次课前一天 23:59（按组间展示时间倒推）
> **提交方式**：Kaggle Notebook 链接 + 本地 `.zip`（含 notebook + adapter + result.json + report.md）
> **总分**：100 + 20（加分）

---

## 🎯 任务目标

在 Polly 提供的 500 条 Incoterms + HS 编码 QA 数据上，用 `peft + QLoRA` 微调 Qwen2.5-1.5B-Instruct，并在 50 条盲测集上打分。

你可以改 rank / target_modules / learning_rate / epochs 等超参，目标是**在保证 BLEU / ROUGE-L 有提升的前提下，把可训练参数占比压到 1% 以内**。

---

## 📋 必做项（100 分）

### 1. 跑通完整主线（30 分）

- [ ] **15 分** — 成功训练并保存 adapter（≤ 50MB）
- [ ] **15 分** — 在 `test.jsonl` 全量 50 条上跑出 `base` + `lora` 的 BLEU / ROUGE-L

提交：`result.json`（由 `eval.py` 生成，包含逐条对比）。

### 2. 参数量报告（20 分）

- [ ] **10 分** — `model.print_trainable_parameters()` 的截图
- [ ] **5 分**  — adapter 文件大小（MB）
- [ ] **5 分**  — 与「全量微调 1.5B fp16」的显存需求估算对比

格式示例（写入 `report.md`）：

```
| 项目 | 全量 FT (fp16) | QLoRA (r=8 QKVO+FFN) |
| ---- | -------------- | -------------------- |
| 可训练参数 | 1,552,743,424 | 8,798,208 (0.566%) |
| 训练显存估算 | ~24 GB | ~7 GB |
| adapter 大小 | 3 GB | 35 MB |
```

### 3. 至少 1 个 Ablation（20 分）

从下面**任选一个维度**做 2-3 组对比：

| 变量 | 选项 |
|------|------|
| **A. Rank** | `r ∈ {4, 8, 16}`，其他固定 |
| **B. Target modules** | `QV` vs `QKVO` vs `QKVO + FFN`，其他固定 |
| **C. Learning rate** | `lr ∈ {5e-5, 1e-4, 2e-4, 5e-4}`，其他固定 |

每组产出一行表格（BLEU / ROUGE-L / 训练时长 / 可训练参数量）+ 一段 100 字分析，写入 `report.md`。

### 4. 推理对比与失败案例分析（20 分）

- [ ] **10 分** — 至少挑 5 条测试样例，放「基座 vs LoRA 后」的答案对比
- [ ] **10 分** — 选 **1-2 个 LoRA 答错或答歪**的样本，分析可能原因（100-200 字）

失败案例分析是**看学生有没有真的在看模型输出**的关键指标——不要只看分数。

### 5. 组间展示（10 分，课堂打分）

5 分钟 PPT 或 Notebook 展示：
- LoRA 配置（r / alpha / target / lr）
- 盲测分
- 1 个你觉得最有意思的失败案例 + 你的解释

---

## ✨ 加分项（+20 分，选做）

| 项目 | 分数 | 说明 |
|------|------|------|
| **A. 合并权重 + 导出完整模型** | +5 | `merge_and_unload` → `save_pretrained`，可直接分发 |
| **B. 本地部署 demo** | +5 | 用 `ollama` 或 `llama.cpp` 跑合并后的模型，录 30s 视频 |
| **C. 上传 adapter 到 HuggingFace Hub** | +5 | `push_to_hub`，分享链接 |
| **D. 用 `unsloth` 重跑对比** | +5 | 5070 Ti / 8GB+ GPU 同学可选，对比 `unsloth` 的加速比 |

⚠️ 做 B/C/D 必须**先完成所有必做项**，否则加分不计。

---

## 📦 交付物清单

打包一个 zip（命名：`学号_姓名_LoRA第二课.zip`），含：

```
学号_姓名_LoRA第二课/
├── notebook.ipynb             # 你的训练 notebook（能重新 Run All）
├── result.json                # eval.py 输出
├── report.md                  # 参数量报告 + ablation 表 + 失败案例分析
├── adapter_link.txt           # adapter 下载链接（见下方说明）
└── (optional) demo.mp4        # 加分项的演示视频
```

### Adapter 提交方式（二选一）

**方式 A（推荐）**：Kaggle Notebook 右上角 **Output** → 确认 adapter 在 Output 里 → 把 Notebook 链接贴到 `adapter_link.txt`

**方式 B**：`push_to_hub` 上传到 HuggingFace Hub → 把 repo 链接贴到 `adapter_link.txt`

⚠️ **不要**把 adapter 打进 zip（30-40 MB 邮件发不出去）。也**不要**上传完整的 base 模型或训练数据——两者都有官方来源，助教用原样跑你的 adapter。

---

## 🚨 常见问题

> **Q: Kaggle 跑不完 3 epoch？**
> A: 把 `per_device_train_batch_size` 降到 1，`gradient_accumulation_steps` 升到 8；或者把 `num_train_epochs` 改成 2。

> **Q: Mac 上 OOM 了？**
> A: M1/M2 8GB 跑不动 1.5B fp16，建议换到 Kaggle。如果必须本地跑，把 `MAX_LEN` 从 512 降到 256，或改用 Qwen2.5-0.5B-Instruct。

> **Q: BLEU / ROUGE-L 分都很低，正常吗？**
> A: 中文 QA 的 BLEU 绝对值不高（10-30 属正常区间）。看**相对提升**更重要——LoRA 后比基座高 ≥5 分就算成功。

> **Q: 可以改 prompt 模板吗？**
> A: 可以，但 base 和 LoRA 必须用**同一个模板**评测，否则对比不公平。

---

## 📚 推荐阅读

- LoRA: [Hu et al., 2021 《LoRA: Low-Rank Adaptation of Large Language Models》](https://arxiv.org/abs/2106.09685)
- QLoRA: [Dettmers et al., 2023 《QLoRA: Efficient Finetuning of Quantized LLMs》](https://arxiv.org/abs/2305.14314)
- PEFT 官方文档：https://huggingface.co/docs/peft
- Qwen2.5 技术报告：https://arxiv.org/abs/2412.15115

祝你玩得开心，外贸小助手炼成后记得秀给老师看 🦞
