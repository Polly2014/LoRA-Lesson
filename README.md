# LoRA 训练原理与实操 · 对外经贸大学

> 两次课，从**数学原理 → 手撕代码 → 框架实战**，用 LoRA 微调中文大模型
> 适合非 CS 专业学生，Kaggle 零配置即可上手

---

## 🗺️ 目录

```
LoRA-Lesson/
├── PLAN.md                     # 📋 完整课程规划（v0.3）
├── README.md                   # 本文件
├── TROUBLESHOOTING.md          # 🔥 常见报错与解法
├── RESOURCES.md                # 📚 论文 / 文档 / 延伸阅读
├── lesson1/                    # 🎓 第一课：原理 + 手撕 LoRA
│   ├── slides.md               # 课件（Marp 格式）
│   ├── notebook_kaggle.ipynb   # 主线 notebook（自动适配 Kaggle CUDA / Mac MPS / CPU）
│   ├── homework.md             # 作业要求
│   └── solution.ipynb          # 参考答案（课后发）
├── lesson2/                    # 🛠️ 第二课：框架实操（外贸术语 QA）
│   ├── slides.md
│   ├── notebook_qlora_kaggle.ipynb   # MODE 分支：qlora / fp16_lora / fp32_lora
│   ├── eval.py                 # BLEU + ROUGE-L 盲测打分
│   ├── homework.md
│   ├── data/
│   │   ├── seeds.jsonl         # 20 条人工精写种子
│   │   ├── build_dataset.py    # CopilotX 合成脚本
│   │   └── README.md
│   └── checkpoints/README.md
└── assets/                     # 图片资源
```

---

## ⚡ 课前准备（必做，请在上课前一周完成）

1. **注册 Kaggle 账号**：https://www.kaggle.com/account/login?phase=startRegisterTab
2. **手机号验证**：头像 → Settings → Phone Verification → 输入手机号（支持 +86）
   - ⚠️ **不验证手机号 = 无法使用 GPU、无法联网**，课堂上会完全卡住
3. **验证成功后测试**：新建 Notebook → Settings → Accelerator 选 **GPU T4 x2** → Internet 选 **On**
4. 跑一个 cell 确认环境：
   ```python
   import torch; print(torch.cuda.get_device_name(0))  # 应输出 "Tesla T4"
   ```
5. **注册 HuggingFace 账号**（第二课需要）：https://huggingface.co/join

> 🔴 如果 GPU 选项是灰色的，说明手机号验证还没完成。

---

## 🚀 快速开始（Kaggle）

1. 打开 Kaggle，新建 Notebook
2. Upload → 选 [`lesson1/notebook_kaggle.ipynb`](lesson1/notebook_kaggle.ipynb)
3. 右侧 Settings → Accelerator 选 **GPU T4 x2**，Internet 选 **On**
4. `Run All` → 全程约 15 分钟

> 首次运行会下载 Qwen2.5-0.5B（约 1 GB）。如遇网络问题，查 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

---

## 💻 快速开始（本地 GPU / Mac）

```bash
# 用 conda（推荐 Mac M1/M2/M3/M4）
conda create -n lora python=3.11 -y
conda activate lora
pip install "torch>=2.1" "transformers>=4.40" "datasets>=2.18" \
            "accelerate>=0.27" matplotlib tqdm ipykernel
jupyter lab
```

直接打开 `lesson1/notebook_kaggle.ipynb` 即可——notebook 里的 `DEVICE` 会自动检测 `cuda` / `mps` / `cpu`，无需改代码。

> 💡 Mac 用 MPS 跑 Qwen2.5-0.5B fp32 比 T4 慢约 3–5 倍，但完全可用（单 epoch ~5min）。

---

## 📚 第一课 · 原理 + 手撕 LoRA（3h）

| Part | 内容 |
|------|------|
| Part 1 | 动机（显存账）+ 数学（$W + \frac{\alpha}{r}BA$） |
| Part 2 | 手撕 `LoRALinear` + 挂到 Qwen2.5-0.5B + Rank/Target Ablation |
| Part 3 | SVD 可视化 + 作业 + Q&A |

**核心产出**：用 ~30 行 PyTorch 实现 LoRA，在 T4 上 5 分钟跑完中文情感分类微调，可训练参数仅占 0.16%。

详见 [`lesson1/slides.md`](lesson1/slides.md) 和 [`lesson1/homework.md`](lesson1/homework.md)。

---

## 🛠️ 第二课 · 框架实操：外贸术语 QA（3h）

| Part | 内容 |
|------|------|
| Part 1 | PEFT + QLoRA 原理 + 数据/SFTTrainer 配置 |
| Part 2 | 训练（跑起来 + 调参观察）|
| Part 3 | 合并权重 + 推理对比 + 盲测打分 + 部署 demo + 组间展示 |

**核心产出**：用 `peft` + QLoRA 微调 Qwen2.5-1.5B-Instruct 到外贸 QA 场景，盲测集 BLEU/ROUGE-L 相对基座提升 ≥ 5 分，adapter 仅 30-40 MB。

**数据**：Polly 用 CopilotX 合成的 500 训练 + 50 盲测（Incoterms 2020 + HS 编码）。
**算力**：Kaggle T4 → 4bit QLoRA；Mac M 系列 → fp16 LoRA；同一个 notebook 自动分支。

详见 [`lesson2/slides.md`](lesson2/slides.md) 和 [`lesson2/homework.md`](lesson2/homework.md)。

---

## 🎯 学习目标

学完本课程，你能够：

1. **理解** LoRA 为什么 work（低秩假设 + SVD 验证）
2. **手写** LoRA 实现，不依赖任何框架
3. **使用** HuggingFace `peft` 做工业级 QLoRA 微调
4. **设计** LoRA 配置的 ablation 实验
5. **部署** 合并后的模型到本地 / 云函数 / Ollama

---

## 💡 算力要求

| 你有什么 | 能跑什么 |
|---------|---------|
| 任何笔记本 | Kaggle T4（免费）✅ 推荐 |
| M4 Pro 48GB | 本地 MPS + bf16 加餐 |
| 5070 Ti / 4090 (16GB+) | 本地 CUDA + 第二课的 7B 加餐 |
| 仅 CPU | 只能看 notebook，不跑实验 |

**不需要**付费 GPU、Colab Pro、本地显卡。

---

## 🧑‍🏫 讲师

**Polly Wang (王保利)**
- 博客：https://polly.wang
- 本课程为对外经贸大学 2026 春学期客座课程

---

## 📜 License

MIT License。欢迎 fork、改编、用于自己的教学。

---

## 🙏 致谢

- **小龙虾** 🦞——课程规划的 reviewer，帮我砍掉了一堆过度设计
- HuggingFace `peft` / `transformers` / `datasets` 团队
- Qwen 团队（阿里）提供的开源中文基座
- LoRA 原作者 Hu et al.（2021）
