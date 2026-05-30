"""
eval.py — 盲测集打分脚本（BLEU + ROUGE-L）

用法：
    # 训练好的 adapter 在 ./checkpoints/lora_qlora_out/adapter
    python eval.py --adapter ./checkpoints/lora_qlora_out/adapter \
                   --test data/test.jsonl \
                   --out result.json

    # 对比基座（无 adapter）
    python eval.py --base-only --test data/test.jsonl

    # 指定底座模型（默认 Qwen2.5-1.5B-Instruct）
    python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter ...

输出：
    - 终端表格（BLEU / ROUGE-L）
    - result.json 含逐条 pred/ref + 整体分数
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def pick_device_and_mode():
    if torch.cuda.is_available():
        return "cuda", "qlora"
    if torch.backends.mps.is_available():
        return "mps", "fp16_lora"
    return "cpu", "fp32_lora"


def load_base_model(model_name: str, mode: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if mode == "qlora":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
    else:
        dtype = torch.float16 if mode == "fp16_lora" else torch.float32
        device, _ = pick_device_and_mode()
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
    return tok, model


def attach_adapter(model, adapter_path: str):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter_path)


def chat(model, tok, prompt: str, max_new_tokens: int = 300) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.pad_token_id)
    new = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def score(preds: list[str], refs: list[str]) -> dict:
    """jieba 分词后计算 BLEU + ROUGE-L"""
    import jieba
    from sacrebleu import corpus_bleu
    from rouge_score import rouge_scorer

    def tok(s): return " ".join(jieba.lcut(s))
    preds_tok = [tok(p) for p in preds]
    refs_tok = [tok(r) for r in refs]

    bleu = corpus_bleu(preds_tok, [refs_tok], tokenize="none").score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rougels = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs_tok, preds_tok)]
    return {"BLEU": bleu, "ROUGE-L": float(np.mean(rougels)) * 100, "n": len(preds)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="LoRA 盲测评分")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录；不传即评测基座")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--base-only", action="store_true", help="只评基座")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"❌ 测试集不存在：{test_path}", file=sys.stderr)
        sys.exit(1)
    test = load_jsonl(test_path)
    print(f"[load] test = {len(test)} 条")

    device, mode = pick_device_and_mode()
    print(f"[env] device={device}  mode={mode}")

    tok, model = load_base_model(args.model, mode)

    results = {"n": len(test), "model": args.model, "test": str(test_path)}

    def run(label: str):
        preds, refs = [], []
        for ex in tqdm(test, desc=label):
            preds.append(chat(model, tok, ex["instruction"], args.max_new_tokens))
            refs.append(ex["output"])
        metrics = score(preds, refs)
        print(f"\n{label:<12} | BLEU {metrics['BLEU']:.2f}  ROUGE-L {metrics['ROUGE-L']:.2f}")
        return metrics, [
            {"instruction": ex["instruction"], "reference": r, "prediction": p}
            for ex, p, r in zip(test, preds, refs)
        ]

    # 1) 基座
    base_metrics, base_items = run("base")
    results["base"] = base_metrics

    # 2) 带 adapter
    if not args.base_only and args.adapter:
        model = attach_adapter(model, args.adapter)
        model.eval()
        lora_metrics, lora_items = run("lora")
        results["lora"] = lora_metrics
        results["delta"] = {k: lora_metrics[k] - base_metrics[k]
                            for k in ["BLEU", "ROUGE-L"]}
        results["items"] = [
            {**b, "prediction_lora": l["prediction"]}
            for b, l in zip(base_items, lora_items)
        ]
    else:
        results["items"] = base_items

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已写入：{args.out}")

    if "lora" in results:
        print(f"\n--- 汇总 ---")
        print(f"{'':<8} | {'BLEU':>8} | {'ROUGE-L':>8}")
        print(f"{'base':<8} | {results['base']['BLEU']:>8.2f} | {results['base']['ROUGE-L']:>8.2f}")
        print(f"{'lora':<8} | {results['lora']['BLEU']:>8.2f} | {results['lora']['ROUGE-L']:>8.2f}")
        d = results["delta"]
        print(f"{'Δ':<8} | {d['BLEU']:>+8.2f} | {d['ROUGE-L']:>+8.2f}")


if __name__ == "__main__":
    main()
