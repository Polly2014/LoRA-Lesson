# 第一课作业 · 手撕 LoRA 微调 Qwen2.5-0.5B

> **截止时间**：下次上课前一天 23:59
> **提交方式**：Kaggle Notebook 分享链接 + 导出的 `.ipynb` 文件（邮件）
> **总分**：100 分 + 20 分 bonus

---

## 一、作业目标

1. 跑通 `notebook_kaggle.ipynb` 的全部 cell，理解每一步在做什么
2. 完成两个 **Ablation 实验**，产出对比表 + loss 曲线图
3. 对训练后的 $\Delta W$ 做 **SVD 可视化**，并写一段分析
4. 阅读并理解第二课的外贸术语 QA 数据集（不需要跑）

---

## 二、必做项（共 100 分）

### 任务 1 · Rank Ablation（30 分）

固定 `target_modules = ("q_proj", "v_proj")`, `alpha = 16`，对比 **r ∈ {1, 4, 8, 32}** 四种配置：

- 每个配置训练 2 epochs（与 notebook 主线一致）
- 记录：可训练参数量、最终 `val_loss`、训练 loss 曲线
- 产出：**一张对比表** + **一张四条曲线叠在同一图的 loss 曲线图**

**评分点**：
- [ ] 4 个配置全部跑完（10 分）
- [ ] 对比表含"配置 / 可训练参数 / 占比 / val_loss"四列（10 分）
- [ ] Loss 曲线图清晰、含图例（10 分）

### 任务 2 · Target Modules Ablation（30 分）

固定 `r = 8`, `alpha = 16`，对比 **QV / QKVO / FFN / ALL** 四种配置：

- 其中 `QV` 和 `FFN` notebook 已给出，需要**补齐 QKVO 和 ALL**
- 同样输出对比表 + loss 曲线图

**评分点**：
- [ ] 4 个配置全部跑完（10 分）
- [ ] 对比表格完整（10 分）
- [ ] **文字说明 QKVO 的参数量为什么不是 QV 的 2×**（10 分，提示：GQA）

### 任务 3 · SVD 可视化 + 分析（30 分）

对**主线配置（QV, r=8）训练后**的 LoRA：

- 至少画 **3 个代表性层**（如第 0、中间、最后）的 q_proj 和 v_proj 的 $\Delta W$ 的 SVD 奇异值谱
- Y 轴用 log scale，画出归一化奇异值 $\sigma_i / \sigma_0$
- 在图上标出 `r=8` 截断线

> 📌 **边界澄清**：notebook Section 13 已经画好 3 层 × 2 proj 的 SVD 谱，**图可以直接复用**（把自己训练出的权重代入即可）。但**200 字分析必须自己写**，抄 `solution.ipynb` 里的范文会判 0 分（我们会对比重复率）。

**评分点**：
- [ ] SVD 谱图 ≥ 3 层（15 分）
- [ ] **一段 200 字以上的文字分析（原创）**，回答：（15 分）
  - 为什么前几个奇异值远大于其他？
  - 这验证了 LoRA 原论文的哪个假设？
  - 如果我们把 r 从 8 降到 2，是否仍能捕获大部分信息？

### 任务 4 · 推理对比展示（10 分）

- 选 5 条 ChnSentiCorp **验证集未见过**的评论
- 用 `disable_lora` context manager 对比**基座 vs LoRA 后**的预测
- 做成表格粘到 notebook 末尾

**评分点**：
- [ ] 5 条样例齐全（5 分）
- [ ] 表格清晰，含原文、基座预测、LoRA 预测三列（5 分）

---

## 三、Bonus（共 20 分）

任选其一：

### Bonus A · 跨 rank 的 SVD 对比（10 分）

画一张图，对比 **r=4, r=8, r=32** 训练后的同一层（如第 0 层 q_proj）的 SVD 谱，看"可用秩"是否真的由 r 决定。

### Bonus B · 本地算力挑战（10 分，仅组 2 可做）

用 M4 Pro (MLX) 或 5070 Ti 本地跑一遍主线配置，报告：
- 训练速度对比（Kaggle T4 vs 本地）
- 显存占用
- 遇到的坑 + 解决方案

### Bonus C · 第二课数据集预热（10 分）

阅读 `lesson2/data/train.jsonl` 的前 50 条，回答：
- 数据格式是什么？
- 任务定义是什么（给什么输入、期望什么输出）？
- 如果让你设计 prompt 模板，你会怎么写？

---

## 四、提交格式

请**把所有内容合并到一个 `.ipynb`** 里提交：

```
notebook_homework_<你的学号>.ipynb
├── [原 notebook 主线] 保留
├── Section 14: Rank Ablation          ← 任务 1
├── Section 15: Target Ablation        ← 任务 2
├── Section 16: SVD Deep Dive          ← 任务 3
├── Section 17: 推理对比               ← 任务 4
└── Section 18 (可选): Bonus           ← Bonus
```

Markdown cell 里用二级标题清晰标注每个任务。

---

## 五、常见问题

**Q：Kaggle 每周 30h GPU 够用吗？**
A：四个 r 配置 + 四个 target 配置 = 8 次训练，每次 5min，加上 SVD 分析共约 1 小时。完全够用。

**Q：HF 模型下载卡住怎么办？**
A：在 notebook 第一个 cell 加：
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

**Q：`val_loss` 多低算好？**
A：主线 (QV, r=8, 2 epoch) 参考值约 **0.5–0.8**。r=1 会明显高一些，r=32 会略低但提升不大（这正是作业要观察的现象）。

**Q：可以用 `peft` 库吗？**
A：**不可以**。第一课的核心目的是理解 LoRA 内部机制，必须手撕。第二课才用框架。

---

## 六、预习下次课

第二课我们用 **HuggingFace `peft` + QLoRA** 训练 **Qwen2.5-1.5B** 做**外贸术语问答**。请：
1. 注册 HuggingFace 账号（如果还没有）
2. 浏览一下 `peft` 文档：https://huggingface.co/docs/peft
3. 看一眼 `lesson2/data/train.jsonl`（会提前发到群里）

---

> 加油！💪 有问题随时群里问。
