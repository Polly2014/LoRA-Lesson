# Checkpoints（课堂接力快照）

本目录用于存放**各 ablation 阶段的 LoRA 权重快照**（`.pt` 文件），课堂上有同学跑不通可以从任意断点接力。

Polly 会在开课前把跑好的 checkpoint 传到对应的 Kaggle Dataset，链接会在课堂上发布。

**包含**：
- `lora_qv_r1.pt` — Rank Ablation 各配置
- `lora_qv_r4.pt`
- `lora_qv_r8.pt`（主线）
- `lora_qv_r32.pt`
- `lora_qkvo_r8.pt` — Target Ablation 各配置
- `lora_ffn_r8.pt`
- `lora_all_r8.pt`

**用法**（在 notebook 里）：
```python
load_lora_state(model, "checkpoints/lora_qv_r8.pt")
```

> ⚠️ 本目录为空是正常的——checkpoint 是课堂 artifact，不进 git 仓库。
