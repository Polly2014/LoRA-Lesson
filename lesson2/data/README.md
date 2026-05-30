# 外贸术语 QA 数据集（Lesson 2 教学专用）

## 文件清单

| 文件 | 说明 |
|------|------|
| `seeds.jsonl` | 人工撰写的 20 条高质量种子（few-shot 示例 + 兜底数据） |
| `build_dataset.py` | 合成脚本（调用 CopilotX / OpenAI 兼容 API） |
| `train.jsonl` | 500 条训练数据（`python build_dataset.py` 后生成） |
| `test.jsonl` | 50 条盲测数据（同上） |

## 生成全量数据集

```bash
cd lesson2/data

# 1) 设置 API key（CopilotX 走 api.polly.wang）
export CPX_API_KEY=sk-xxxxx
# 可选 base_url，默认即 https://api.polly.wang/v1
# export CPX_BASE_URL=https://api.polly.wang/v1

# 2) 生成 500 训练 + 50 测试
python build_dataset.py --n-train 500 --n-test 50 --model gpt-4o

# 快速试跑（小样本）
python build_dataset.py --n-train 20 --n-test 5

# 完全不调 API，仅用 seeds
python build_dataset.py --no-api --n-train 20 --n-test 5
```

预计耗时：gpt-4o + 8 并发 ≈ 10-15 分钟。

## 数据格式

Alpaca-style 单轮指令：

```json
{"instruction": "请解释 Incoterms 2020 中 FOB 术语的含义和买卖双方责任。", "input": "", "output": "FOB（Free On Board……"}
```

## 话题覆盖

- **Incoterms 2020 全 11 个术语**：EXW / FCA / FAS / FOB / CFR / CIF / CPT / CIP / DAP / DPU / DDP
- **HS 编码实务**：归类总规则 GRI、6 位国际 vs 本国细分、原产地规则、常见章节（85 机电、61 针织服装、22 酒类等）
- **衍生主题**：保险条款 ICC-A/B/C、信用证下术语选择、预归类裁定

## 质量保证

1. **20 条 seeds 人工撰写**，确保 baseline 质量
2. **few-shot 提示**：每次生成时随机抽 3 条 seeds 作为示例
3. **字面相似度去重**（SequenceMatcher > 0.75 视为重复）
4. **失败重试 3 次**（指数退避）
5. **seed=2026 固定洗牌**，train/test 可复现

## 盲测集使用

`test.jsonl` 的 50 条**不要**让学生看到——课堂评分用。学生提交微调后的模型，用 `eval.py` 在 `test.jsonl` 上跑 BLEU + ROUGE-L。
