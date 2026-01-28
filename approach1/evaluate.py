#!/usr/bin/env python3
"""
evaluate.py
Load adapter & evaluate on a held-out JSONL (same schema).
Computes Exact Match and token-overlap F1 (simple).
"""
import re
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# CONFIG
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # base checkpoint to load
ADAPTER_DIR = Path("./qa_lora_out")
EVAL_FILE = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/approch1_pagewise/qa_dataset.jsonl")  # ideally a val split

PROMPT_TEMPLATE = """### Instruction:
Answer the question concisely and authoritatively using the Income Tax Act.

Question:
{question}

Answer:
"""

def normalize(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def f1_score(pred: str, gold: str) -> float:
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gold).split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 1.0 if p_tokens == g_tokens else 0.0
    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    if match == 0:
        return 0.0
    prec = match / len(p_tokens)
    rec = match / len(g_tokens)
    return 2 * prec * rec / (prec + rec)

def load_eval_samples(path: Path, n=None):
    out = []
    with path.open() as f:
        for i,line in enumerate(f):
            if n and i>=n: break
            obj = json.loads(line)
            q = obj.get("question") or obj.get("q") or obj.get("prompt")
            a = obj.get("answer") or obj.get("ans") or obj.get("response")
            if q is None or a is None:
                continue
            out.append((q, a))
    return out

def main():
    print("Loading tokenizer + base model")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map="auto", trust_remote_code=True)

    samples = load_eval_samples(EVAL_FILE)
    print("Evaluation samples:", len(samples))

    ems = []
    f1s = []
    import torch
    model.eval()
    for q, gold in samples:
        prompt = PROMPT_TEMPLATE.format(question=q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        em = 1 if normalize(gen) == normalize(gold) else 0
        f1 = f1_score(gen, gold)
        ems.append(em)
        f1s.append(f1)

    print(f"Exact Match: {sum(ems)/len(ems):.4f}  ({sum(ems)}/{len(ems)})")
    print(f"Avg F1:       {sum(f1s)/len(f1s):.4f}")

if __name__ == "__main__":
    main()
