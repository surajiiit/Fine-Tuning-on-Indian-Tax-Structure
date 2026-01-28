# Approach 1 — Page-wise fine-tuning & inference

This folder contains scripts and data for a page-wise fine-tuning approach and associated inference utilities. The layout is intended for preparing QA-style JSONL datasets, fine-tuning a LoRA adapter, and running inference/evaluation.

Directory layout
- `batch_ollama.py` — helper for batching requests to an Ollama server/CLI (if used).
- `qa_dataset*.jsonl` — various dataset versions in JSONL format (cleaned, filtered, fixed, and raw variants).
- `finetune/` — scripts and model artifacts for fine-tuning:
  - `clean_qa_jsonl.py` — clean and normalize QA JSONL files before training.
  - `convert_to_jsonl.py` — conversion utilities (e.g., from other formats to JSONL).
  - `lora.py` — training script that applies LoRA-style fine-tuning (adapter training).
  - `llama3/` — a local model snapshot (safetensors parts, tokenizer, and config). May be used as the base model for LoRA training or inference.

- `qa_lora_out/` — LoRA adapter outputs and checkpoints. Example contents:
  - `adapter_model.safetensors` — trained LoRA weights.
  - `adapter_config.json` — adapter configuration.
  - `checkpoint-601/` — training checkpoint directory with optimizer/scheduler state and README.

- `inference/` — inference and evaluation utilities:
  - `infer.py` — interactive or scripted inference using the base model + adapter.
  - `evaluate.py` — evaluation script to compute metrics on a dataset (e.g., accuracy, exact match).

- `models/` — a place for additional model files (may be empty or contain local checkpoints).

Quick start
1. Create and activate a Python virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

2. Install likely dependencies. Inspect the scripts for exact requirements. A typical set includes:

```
pip install torch transformers accelerate peft datasets safetensors sentencepiece
```

If you use 8-bit training or `bitsandbytes` for memory efficiency add:

```
pip install bitsandbytes
```

3. Prepare / clean the dataset and convert if needed:

```
python finetune/clean_qa_jsonl.py --input qa_dataset.jsonl --output qa_dataset_clean.jsonl
python finetune/convert_to_jsonl.py --src some_source --dst qa_dataset_clean.jsonl
```

4. Fine-tune a LoRA adapter (example):

```
python finetune/lora.py --dataset qa_dataset_clean.jsonl --output_dir qa_lora_out --model_path finetune/llama3
```

5. Run inference / evaluate (examples):

```
python inference/infer.py --model_path qa_lora_out --prompt "<your prompt>"
python inference/evaluate.py --model_path qa_lora_out --dataset qa_dataset_clean.jsonl
```

Notes & recommendations
- Inspect `finetune/lora.py` for required CLI args (batch size, learning rate, device settings). Adjust for your GPU/CPU availability.
- Add a `requirements.txt` in this directory to pin dependencies for reproducible runs.
- If you plan to deploy or reproduce experiments, include exact `torch` and `transformers` versions and whether `bitsandbytes` is used.
- If `batch_ollama.py` is present, confirm whether you use Ollama (local server) and ensure credentials or host settings are configured.

Outputs to expect
- `qa_lora_out/` — trained adapter artifacts and checkpoints.
- `finetune/llama3/` — base model files (if present locally) used for training or inference.

Next steps I can help with
- Add a `requirements.txt` with pinned versions.
- Create a small example `examples/` with a single sample JSONL and a simple run script.
- Produce a Dockerfile or Makefile to make training and inference reproducible.

If you want any of the next steps, tell me which one and I'll add it.
