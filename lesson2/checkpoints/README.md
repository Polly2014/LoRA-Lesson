# Checkpoints

训练产物默认写到这里。**不要**把这个目录 commit 到 git（已加到 `.gitignore`）。

## 目录结构

```
checkpoints/
├── lora_qlora_out/              # notebook 默认 OUTPUT_DIR
│   ├── adapter/                 # 最终 adapter（提交作业的就是这个）
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors   # ~30-40 MB
│   │   └── README.md
│   ├── checkpoint-50/           # 训练中的中间 ckpt
│   ├── checkpoint-100/
│   └── trainer_state.json
└── merged/                      # （可选）merge_and_unload 后的完整模型
    ├── config.json
    ├── model.safetensors        # ~3 GB
    └── tokenizer.json
```

## Kaggle 提交规则

Kaggle notebook 训练完后，路径是 `/kaggle/working/lora_qlora_out/adapter/`。

想把 adapter 下回本地/交作业：

1. **Notebook 右上角** → Output tab → 下载 zip
2. 或者在 notebook 最后一个 cell 打包：
   ```python
   !cd /kaggle/working && zip -r adapter.zip lora_qlora_out/adapter
   ```

## 本地加载 adapter 做推理

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto", device_map="auto",
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "./checkpoints/lora_qlora_out/adapter")
model.eval()
```

## 磁盘占用参考

| 内容 | 大小 |
|------|------|
| 单个 adapter | 30-40 MB |
| 3 个中间 ckpt (save_total_limit=3) | ~120 MB |
| merged 完整模型 | ~3 GB (fp16) |

训练过程中峰值 ~500 MB，完成后清理中间 ckpt 可降到 50 MB 以内。
