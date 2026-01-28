#!/usr/bin/env python3
# chunks_io.py
import json
from pathlib import Path
import os

def make_chunk_id(source_file: str, chunk_index: int) -> str:
    base = Path(source_file).stem
    return f"{base}__chunk_{chunk_index:04d}"

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
