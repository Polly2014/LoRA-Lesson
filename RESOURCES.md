# RESOURCES · 扩展阅读

按「从易到难」排列。课堂上用不到，但你想继续深挖可以看。

---

## 🎓 论文

### 必读（第一二课的直接背景）

- **LoRA**: Hu et al., 2021
  《LoRA: Low-Rank Adaptation of Large Language Models》
  https://arxiv.org/abs/2106.09685

- **QLoRA**: Dettmers et al., 2023
  《QLoRA: Efficient Finetuning of Quantized LLMs》
  https://arxiv.org/abs/2305.14314

### 进阶（PEFT 家族）

- **Prefix Tuning**: Li & Liang, 2021 — https://arxiv.org/abs/2101.00190
- **Prompt Tuning**: Lester et al., 2021 — https://arxiv.org/abs/2104.08691
- **AdaLoRA**: Zhang et al., 2023 — https://arxiv.org/abs/2303.10512
- **DoRA**: Liu et al., 2024 — https://arxiv.org/abs/2402.09353
- **ReFT**: Wu et al., 2024 — https://arxiv.org/abs/2404.03592（非参数化 PEFT）

### 量化技术

- **GPTQ**: Frantar et al., 2023 — https://arxiv.org/abs/2210.17323
- **AWQ**: Lin et al., 2023 — https://arxiv.org/abs/2306.00978
- **bitsandbytes 8bit**: Dettmers et al., 2022 — https://arxiv.org/abs/2208.07339

---

## 📖 官方文档

- **HuggingFace PEFT**: https://huggingface.co/docs/peft
  → 教程 + API reference + 所有 PEFT 方法对比表
- **TRL (Transformers RL)**: https://huggingface.co/docs/trl
  → `SFTTrainer` / `DPOTrainer`（做 RLHF 的必看）
- **bitsandbytes**: https://huggingface.co/docs/bitsandbytes
- **Qwen2.5**: https://qwenlm.github.io/blog/qwen2.5/

---

## 📺 视频 / 博客（中文友好）

### 入门

- Sebastian Raschka《Finetuning LLMs with LoRA and QLoRA》
  https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- HuggingFace 官方 PEFT 教程（英文，有代码）
  https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu

### 进阶

- **Unsloth 加速原理**（1.5-2× 训练速度，30% 显存节省）
  https://github.com/unslothai/unsloth
- **LLaMA-Factory**（各家 PEFT 方法的集大成者）
  https://github.com/hiyouga/LLaMA-Factory

---

## 🛠️ 实用仓库

| 仓库 | 做什么用 |
|------|---------|
| [huggingface/peft](https://github.com/huggingface/peft) | 本课的 PEFT 官方实现 |
| [huggingface/trl](https://github.com/huggingface/trl) | SFT / DPO / PPO 训练器 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | LoRA 训练加速器 |
| [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 一站式 LLM 微调平台 |
| [OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | 工业级训练配置框架 |

---

## 🌐 数据集 / Benchmark

### 通用指令数据

- Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- Alpaca-GPT4 中文: https://huggingface.co/datasets/shibing624/alpaca-zh
- ShareGPT: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- COIG-CQIA（中文精选）: https://huggingface.co/datasets/m-a-p/COIG-CQIA

### 领域数据

- 法律: https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers
- 医疗: https://github.com/Toyhom/Chinese-medical-dialogue-data
- 金融: https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa

### 评测

- MMLU / C-Eval / CMMLU（通用能力）
- Helm / BigBench（多任务）
- AlpacaEval / MT-Bench（对齐质量）

---

## 💡 下一步学什么

本课讲了 **SFT + LoRA**。想继续深入的路线：

1. **RLHF / DPO**（对齐）→ 看 TRL 的 `DPOTrainer`
2. **更大模型**（7B/13B/70B）→ 多 GPU 训练，看 `accelerate` + `deepspeed`
3. **Agent / Tool Use** → 看 langchain / llama-index / 自研 agent 框架
4. **推理优化** → vLLM / TGI / llama.cpp / TensorRT-LLM
5. **预训练**（from scratch）→ Megatron / nanoGPT

---

## 🦞 问 Polly

- **博客**: https://polly.wang
- **Master-Translator MCP**（开源翻译工具）: https://github.com/polly-wang/master-translator
- **对外经贸大学 AI Spring School 2026** 持续更新

有问题课堂提 / 群里 @ 我 / 博客评论都行。
