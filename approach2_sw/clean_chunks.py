#!/usr/bin/env python3
# clean_chunks.py — filters trivial chunks from chunks.jsonl

import json, re
from pathlib import Path

IN = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/chunks.jsonl")
OUT = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/chunks.cleaned.jsonl")
MIN_NONSPACE_CHARS = 20  # same threshold as chunker. tune if needed.

def is_trivial_text(t: str) -> bool:
    if not t: return True
    s = t.strip()
    if re.match(r'^\[?PAGE\s+\d+\]?$', s, flags=re.IGNORECASE):
        return True
    # skip if too short (few non-space chars)
    if len(re.sub(r'\s+','', s)) < MIN_NONSPACE_CHARS:
        return True
    return False

def main():
    if not IN.exists():
        print("Input not found:", IN); return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with IN.open("r", encoding="utf-8") as fh_in, OUT.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                # keep malformed lines for inspection
                fh_out.write(line + "\n")
                kept += 1
                continue
            text = obj.get("text","")
            if is_trivial_text(text):
                skipped += 1
                continue
            fh_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Done. kept={kept} skipped={skipped}. cleaned file: {OUT}")

if __name__ == "__main__":
    main()
