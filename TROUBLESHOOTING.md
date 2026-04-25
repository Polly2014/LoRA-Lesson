# TROUBLESHOOTING · 常见翻车现场

Kaggle / Mac / Colab 上跑 LoRA 踩过的坑汇总。按 **报错信息** 搜索关键字。

---

## 🔥 环境 & 依赖

### `ImportError: cannot import name ...` from `peft` / `transformers`

**原因**：`peft` 和 `transformers` 版本不匹配，尤其 Kaggle 自带版本老。

**解决**：notebook 第一个 install cell 已经写死：
```python
!pip -q install -U "transformers>=4.45" "peft>=0.11" "trl>=0.9" "accelerate>=0.33"
```
装完**务必重启 runtime**（Kaggle 右上角 Run → Restart kernel）。

### `bitsandbytes` 在 Mac 上装不上

**正常现象**。`bitsandbytes` 只支持 CUDA。notebook 用 `MODE` 检测，Mac 跳过安装，走 fp16 LoRA 分支。

如果你强行 `pip install bitsandbytes` 会失败或装上 CPU-only 假包，后面 `load_in_4bit` 会报 `ValueError: No GPU found`。

### `ModuleNotFoundError: No module named 'jieba'`

eval cell 缺了依赖：
```python
!pip -q install jieba sacrebleu rouge-score
```

---

## 💾 显存 / OOM

### `CUDA out of memory` (T4 16GB)

排查顺序：
1. `per_device_train_batch_size = 1`
2. `gradient_accumulation_steps = 8`（保持有效 bs）
3. `MAX_LEN = 256`（默认 512）
4. 确认 `gradient_checkpointing=True`
5. 确认 `optim="paged_adamw_8bit"` 而不是 `adamw_torch`

如果还 OOM，换小模型：`Qwen/Qwen2.5-0.5B-Instruct`。

### `MPS backend out of memory` (Mac 8GB)

Mac 内存统一架构，fp16 1.5B 模型 + 激活 ≥ 6GB，8GB 机器很勉强。方案：
- 降 `MAX_LEN` 到 256
- 改用 Qwen2.5-0.5B-Instruct
- 或者**不在本地训**，去 Kaggle

### `RuntimeError: Placeholder storage has not been allocated on MPS`

Mac 上某些 tensor 没显式 `.to(device)`。检查自定义 `Dataset` 返回的 tensor 是不是在 CPU 上。

---

## 🧠 训练 / Loss 异常

### Loss 一直是 `nan`

- **fp16 溢出**：降 lr (2e-4 → 1e-4)，或启 `bf16=True`（Ampere+ 或 Mac M2+）
- **标签错了**：打印一条样本看 `labels` 是不是全 -100（那就 loss=0 不更新）
- **gradient checkpointing + kbit**：必须先 `prepare_model_for_kbit_training`，顺序不能错

### Loss 下降但推理质量反而变差

典型"**过拟合到训练集 prompt 格式**"。检查：
- 推理时的 chat template 是否和训练一致？
- 训练数据多样性够不够？（500 条 synthesized 数据可能同质化严重）
- 是不是 epoch 太多了？3 epoch → 2 epoch 试试

### `print_trainable_parameters` 显示 0%

`get_peft_model(model, config)` 返回值**必须赋给原变量**：
```python
model = get_peft_model(model, lora_config)   # ✅
get_peft_model(model, lora_config)            # ❌ 返回值丢了
```

---

## 🎯 推理 / 评测

### LoRA 后生成一模一样的答案

```python
# ❌ 没切换到推理模式
outputs = model(...)

# ✅
model.eval()
with torch.no_grad():
    outputs = model.generate(..., do_sample=False)
```

### BLEU = 0 / ROUGE-L = 0

- 没用 jieba 分词，英文 tokenizer 把中文当一个 token
- 预测和参考**长度都 < 4 个 token**：BLEU n-gram 算不出来，属正常

### 基座 (base) 分数 > LoRA 分数

- 训练时间不够（1 epoch 用 100 步就停了）
- lr 太大把底座打坏了
- target_modules 选错了（只挂 `o_proj` 之类）

把训练 loss 曲线贴出来看——正常应该 `2.5 → 1.0` 左右。

---

## 📦 合并 / 导出

### `merge_and_unload` 报错："can't merge with 4bit base"

QLoRA 的底座是 4bit 量化的，**不能直接 merge**。两种解法：

**方案 A**：重新加载 fp16 底座 + 挂 adapter + merge
```python
base_fp16 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16
)
merged = PeftModel.from_pretrained(base_fp16, "./adapter").merge_and_unload()
merged.save_pretrained("./merged")
```

**方案 B**：不 merge，推理时永远带 adapter。

---

## 🐛 Kaggle 专属

### Dataset 挂载不上

- Notebook 右边栏 → Input → Add Dataset → 搜你上传的数据集名
- 挂载后路径永远是 `/kaggle/input/<dataset-name>/`
- notebook 代码里是 `DATA_DIR = "/kaggle/input/lora-lesson2-data"`，**和数据集 slug 必须匹配**

### 训练卡在 "0%" 不动

- GPU 额度用完了（每周 30h）
- 数据集太大，第一轮在 tokenize
- 打开 Kaggle 右下角的 GPU/CPU 监控看是不是真的闲着

### Session 断了 adapter 怎么找

Kaggle 左下角 → **Versions** → 每次保存都有快照
→ 最新一版 Output 区能下到 `/kaggle/working/` 所有内容

---

## 💬 还是搞不定？

在班级群里发：
- **报错完整 traceback**（截图或贴文字都行）
- **你跑的环境**（Kaggle / Mac / Colab / 哪个 Python 版本）
- **改了哪些代码**（和原始 notebook 的 diff）

这三样齐了，90% 的问题 10 分钟能搞定。
