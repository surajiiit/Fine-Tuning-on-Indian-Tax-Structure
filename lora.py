#!/usr/bin/env python3
"""
finetune_qa_lora.py - fine-tune local Llama 3 with LoRA on QA JSONL.

- Uses your local snapshot:
  /fab3/btech/2022/suraj.yadav22b/FineTuning/approch1_pagewise/finetune/llama3
- Uses bitsandbytes 4-bit quant (optional)
- Trains LoRA adapters (PEFT) and saves only adapter files
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ================= CONFIG ===================
@dataclass
class Cfg:
    # Point to your LOCAL model folder (downloaded snapshot)
    model_name: str = field(
        default="/fab3/btech/2022/suraj.yadav22b/FineTuning/approch1_pagewise/finetune/llama3"
    )
    # Use the cleaned file without "example" column
    data_path: str = field(
        default="/fab3/btech/2022/suraj.yadav22b/FineTuning/approch1_pagewise/qa_dataset_clean_noexample.jsonl"
    )
    output_dir: str = field(default="./qa_lora_out")

    # training hyperparams
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)  # keep low for large models
    gradient_accumulation_steps: int = field(default=8)  # to simulate larger batch
    learning_rate: float = field(default=2e-4)
    max_seq_length: int = field(default=1024)
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.0)

    # LoRA hyperparams
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[List[str]] = field(default=None)

    # quantization / device
    use_4bit: bool = field(default=False)     # requires bitsandbytes
    bnb_compute_dtype: str = field(default="bfloat16")  # or "float16" if needed
    bf16: bool = field(default=True)         # set True if GPU + torch supports bf16
    seed: int = field(default=42)


cfg = Cfg()

# ================ PROMPT TEMPLATE ================
PROMPT_TEMPLATE = """### Instruction:
You are a helpful assistant. Answer the question below concisely and authoritatively using the Income Tax Act.

(Example metadata: id={id}, difficulty={difficulty}, tags={tags})

Question:
{question}

Answer:
{answer}"""

# ================ HELPERS =======================
def format_example(example):
    q = example.get("question", "")
    a = example.get("answer", "")
    _id = example.get("id", "")
    difficulty = example.get("difficulty", "")
    tags = example.get("tags", "")
    if isinstance(tags, (list, tuple, set)):
        tags = ", ".join(map(str, tags))
    else:
        tags = str(tags) if tags is not None else ""
    example["text"] = PROMPT_TEMPLATE.format(
        id=_id, difficulty=difficulty, tags=tags, question=q, answer=a
    )
    return example


def autodetect_device_map():
    """
    Return device_map="auto" for HF to shard across GPUs/CPU.
    """
    if torch.cuda.is_available():
        return "auto"
    # fallback to cpu
    return None


def load_jsonl_as_dataset(path: str) -> Dataset:
    """
    Manually read JSONL with Python's json, skipping bad lines,
    and normalize types so Arrow doesn't complain.

    We only keep:
      - id (str)
      - question (str)
      - answer (str)
      - difficulty (str)
      - tags (list[str])
    """
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] Skipping bad JSON line {lineno}: {e}")
                bad += 1
                continue

            # Normalize fields
            _id = str(obj.get("id", ""))
            question = str(obj.get("question", ""))
            answer = str(obj.get("answer", ""))
            difficulty = str(obj.get("difficulty", ""))

            raw_tags = obj.get("tags", [])
            if raw_tags is None:
                tags_list: List[str] = []
            elif isinstance(raw_tags, (list, tuple, set)):
                tags_list = [str(t) for t in raw_tags]
            else:
                # single value -> wrap into list
                tags_list = [str(raw_tags)]

            rows.append(
                {
                    "id": _id,
                    "question": question,
                    "answer": answer,
                    "difficulty": difficulty,
                    "tags": tags_list,
                }
            )

    print(f"Loaded {len(rows)} good lines from {path}, skipped {bad} bad lines.")
    if not rows:
        raise RuntimeError("No valid rows found in dataset!")
    return Dataset.from_list(rows)


# ================ MAIN ==========================
def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"Config: model={cfg.model_name}, data={cfg.data_path}, output={cfg.output_dir}")
    print("Checking CUDA availability:", torch.cuda.is_available())

    # === load dataset (manual JSONL loader, normalized types) ===
    ds = load_jsonl_as_dataset(cfg.data_path)
    print("Total examples:", len(ds))
    ds = ds.map(format_example, remove_columns=ds.column_names)
    print("Mapped dataset columns:", ds.column_names)

    # === tokenizer (load directly from tokenizer.json) ===
         # === tokenizer (load directly from tokenizer.json) ===
    print("Loading tokenizer from:", cfg.model_name)
    tokenizer_path = os.path.join(cfg.model_name, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
    )

    # Set special tokens explicitly (Llama 3 style)
    tokenizer.bos_token_id = 128000
    tokenizer.eos_token_id = 128001
    tokenizer.padding_side = "right"

    # Use EOS as PAD (common for causal LM)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    # === tokenize dataset for Trainer ===
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenized dataset columns:", tokenized_ds.column_names)

    # === bitsandbytes config (optional) ===
    bnb_config = None
    if cfg.use_4bit:
        compute_dtype = (
            torch.bfloat16 if cfg.bnb_compute_dtype == "bfloat16" else torch.float16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        print("Using 4-bit quantization (bitsandbytes) with compute dtype:", cfg.bnb_compute_dtype)

    # === load base model ===
    device_map = autodetect_device_map()
    model_load_kwargs = {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if device_map is None:
        model_load_kwargs.pop("device_map", None)

    print("Loading base model (this may take a while) with kwargs:", model_load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_load_kwargs)

    # === PEFT / LoRA setup ===
    if cfg.lora_target_modules is None:
        cfg.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === Training arguments ===
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        bf16=cfg.bf16 and torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        report_to="none",
        seed=cfg.seed,
    )

    # === Data collator for causal LM ===
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # === start training ===
    print("Starting training...")
    trainer.train()

    # === save adapter & tokenizer ===
    print("Saving adapter and tokenizer to:", cfg.output_dir)
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Training complete. Adapter saved at:", cfg.output_dir)


if __name__ == "__main__":
    main()
