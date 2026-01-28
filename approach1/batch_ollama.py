#!/usr/bin/env python3
import os
import json
import uuid
import subprocess
from pathlib import Path
from time import sleep

# ====== CONFIG ======
MODEL = "llama3.2:3b"

INPUT_FOLDER = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/income_taxt_act")
SYSTEM_PROMPT_FILE = INPUT_FOLDER / "system_prompt.txt"
OUTPUT_FILE = INPUT_FOLDER / "qa_dataset.jsonl"

TEMPERATURE = 0.1
MAX_TOKENS = 1200
SLEEP_BETWEEN = 0.6

if not SYSTEM_PROMPT_FILE.exists():
    raise SystemExit(f"Missing system_prompt.txt in: {SYSTEM_PROMPT_FILE}")

system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def call_ollama(prompt: str) -> str:
    """
    Works with old Ollama CLI: `ollama run model`
    Provides prompt via stdin.
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=240,
        )
        if proc.returncode != 0:
            print("OLLAMA ERROR:", proc.stderr[:300])
            return ""
        return proc.stdout.strip()
    except Exception as e:
        print("Exception calling ollama:", e)
        return ""


def process_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        user_block = f"source_file: {file_path.name}\n\n<BLANK PAGE>"
    else:
        user_block = f"source_file: {file_path.name}\n\n{text}"

    prompt = (
        "SYSTEM INSTRUCTIONS:\n"
        + system_prompt
        + "\n\n---\nUSER INPUT:\n"
        + user_block
        + "\n\nPlease output ONLY a JSON array as described."
    )

    print(f"Calling Ollama for {file_path.name} ...")
    raw = call_ollama(prompt)
    if not raw:
        print("Empty response.")
        return []

    # Try JSON parse
    try:
        return json.loads(raw)
    except:
        # Try extracting array
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                print("JSON parse failed for:", file_path.name)
                print("RAW RESPONSE:", raw[:600])
                return []
        print("Could not extract JSON for:", file_path.name)
        return []


def main():
    txt_files = sorted([p for p in INPUT_FOLDER.iterdir() if p.suffix == ".txt"])

    if not txt_files:
        print("No .txt files found!")
        return

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for f in txt_files:
            results = process_file(f)
            for obj in results:
                if "id" not in obj:
                    obj["id"] = f"{f.stem}_{uuid.uuid4().hex[:8]}"
                if "source_file" not in obj:
                    obj["source_file"] = f.name
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sleep(SLEEP_BETWEEN)

    print("\nDONE! Output saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
