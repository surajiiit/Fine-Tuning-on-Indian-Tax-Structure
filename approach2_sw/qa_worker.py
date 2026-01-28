#!/usr/bin/env python3
# qa_worker.py
import os, subprocess, time, json
from typing import List, Dict,Any

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
WORKER_TIMEOUT = 300
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

def call_ollama_raw(prompt: str, timeout:int = WORKER_TIMEOUT, cuda_device: str = None) -> str:
    env = os.environ.copy()
    if cuda_device:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    try:
        proc = subprocess.run(["ollama", "run", OLLAMA_MODEL], input=prompt, text=True, capture_output=True, timeout=timeout, env=env)
        if proc.returncode != 0:
            return "__OLLAMA_ERR__:" + proc.stderr.strip()
        return proc.stdout.strip()
    except Exception as e:
        return "__OLLAMA_EXCEPTION__:" + repr(e)

def parse_json_array_from_raw(raw: str) -> List[Dict[str,Any]]:
    if not raw:
        return []
    if raw.startswith("__OLLAMA_ERR__") or raw.startswith("__OLLAMA_EXCEPTION__"):
        return [{"__error__": raw}]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # try extraction
    start = raw.find("["); end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except Exception:
            return [{"__error__":"json_extract_failed","raw_head":raw[:1000]}]
    return [{"__error__":"no_json_array_found","raw_head":raw[:1000]}]

def generate_for_chunk(system_prompt: str, source_file: str, chunk_id: str, chunk_text: str, cuda_device: str = None) -> List[Dict[str,Any]]:
    user_block = (
        f"source_file: {source_file}\n"
        f"source_chunk_id: {chunk_id}\n\n"
        f"{chunk_text}\n\n"
        "ADDITIONAL: Output EXACTLY a JSON array and include 'source_chunk_id'. Do NOT include any commentary."
    )
    prompt = "SYSTEM INSTRUCTIONS:\n" + system_prompt + "\n\n---\nUSER INPUT:\n" + user_block + "\n\nOutput EXACTLY a JSON array."
    raw = ""
    delay = 1.0
    for attempt in range(1, MAX_RETRIES+1):
        raw = call_ollama_raw(prompt, timeout=WORKER_TIMEOUT, cuda_device=cuda_device)
        if raw and not raw.startswith("__OLLAMA_ERR__") and not raw.startswith("__OLLAMA_EXCEPTION__"):
            break
        time.sleep(delay)
        delay *= RETRY_BACKOFF
    objs = parse_json_array_from_raw(raw)
    # ensure provenance fields present
    for o in objs:
        if isinstance(o, dict):
            o.setdefault("source_chunk_id", chunk_id)
            o.setdefault("source_file", source_file)
    return objs
