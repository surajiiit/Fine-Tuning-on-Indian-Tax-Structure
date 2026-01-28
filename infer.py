
import os
import re
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
from peft import PeftModel

# -------------------------
# Paths (adjust to your layout)
# -------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]              # .../FineTuning/approach1_pagewise
FINETUNE_DIR = PROJECT_ROOT / "finetune"

BASE_MODEL_DIR = FINETUNE_DIR / "llama3"
ADAPTER_DIR = FINETUNE_DIR / "qa_lora_out"

# Device reported (uses whatever CUDA_VISIBLE_DEVICES exposes)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Diagnostics
print("Using DEVICE:", DEVICE)
print("ENV CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())

if torch.cuda.is_available():
    try:
        print("torch visible device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  torch device {i}: name={p.name}, total_mem={p.total_memory//1024//1024} MB, major={p.major}, minor={p.minor}")
    except Exception:
        # Some environments may error on get_device_properties in odd cases; continue anyway
        pass

# -------------------------
# Tokenizer loader
# -------------------------
def load_tokenizer() -> PreTrainedTokenizerFast:
    tok_path = BASE_MODEL_DIR / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tok_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))

    # Llama 3 special tokens (as used during training)
    tokenizer.bos_token_id = 128000
    tokenizer.eos_token_id = 128001
    tokenizer.padding_side = "right"

    # Use EOS as PAD (Llama-style)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

# -------------------------
# Model loader
# -------------------------
def load_model():
    # Use bfloat16 on CUDA for Ampere/A100; if CUDA not available, float32 will be used by device_map
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    print(f"Loading base model from: {BASE_MODEL_DIR} with dtype={dtype}, device_map='auto'")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), device_map="auto")
    model.eval()
    return model

# -------------------------
# Prompt builder (strict)
# -------------------------
def build_prompt(question: str) -> str:
    return f"""### Instruction:
You are a helpful assistant. Answer the question concisely using the Income Tax Act.
Do NOT output any of the following under any circumstance:
- disclaimers (e.g. "Disclaimer: ...")
- identification tokens or IDs (e.g. "(id=...)" or "id: ...")
- metadata fields such as "Difficulty:", "Tags:", "Category:", or similar
- legal advice phrasing; give factual answers only

Only output the plain answer text (no trailing metadata).

Question:
{question}

Answer:
"""

# -------------------------
# Post-generation cleaning
# -------------------------
def clean_output(text: str) -> str:
    # Normalize whitespace
    text = text.strip()

    # If the model included "Answer:" at other positions, remove the earlier portion
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()

    # Remove common disclaimers (catch lots of variants, case-insensitive)
    text = re.sub(r"Disclaimer:.*?$", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove ID patterns like (id=xxxxx-xxxx-...), [id: ...], id=..., id: ...
    text = re.sub(r"\(id=[^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[id:[^\]]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(id|ID|Id)\s*[:=]\s*[0-9A-Za-z\-]+\b", "", text, flags=re.IGNORECASE)

    # Remove metadata lines like Difficulty: Medium, Tags: a, b, c, Difficulty - Medium etc.
    text = re.sub(r"^\s*(Difficulty|Tags?)\s*[:\-].*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove trailing parentheses or stray separators left over
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)          # collapse multiple spaces
    text = re.sub(r"\n{2,}", "\n", text)         # collapse multiple newlines

    return text.strip()

# -------------------------
# Generation
# -------------------------
@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Remove token_type_ids if present
    inputs.pop("token_type_ids", None)

    # Move inputs to device (cuda or cpu)
    device_for_inputs = DEVICE if DEVICE == "cuda" else "cpu"
    inputs = {k: v.to(device_for_inputs) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_output(raw)
    return cleaned

# -------------------------
# CLI / REPL
# -------------------------
def main():
    try:
        tokenizer = load_tokenizer()
    except FileNotFoundError as e:
        print("Tokenizer error:", e, file=sys.stderr)
        sys.exit(2)

    try:
        model = load_model()
    except Exception as e:
        print("Model load error:", e, file=sys.stderr)
        sys.exit(3)

    # One-shot: accept question as CLI args
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print("Q:", question)
        ans = generate_answer(model, tokenizer, question)
        print("A:", ans)
        return

    # Interactive mode
    print("Loaded model. Ask tax questions (type 'exit' to quit).")
    try:
        while True:
            q = input("\nQ: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                print("Bye.")
                break
            ans = generate_answer(model, tokenizer, q)
            print("A:", ans)
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")

if __name__ == "__main__":
    main()