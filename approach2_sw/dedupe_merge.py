#!/usr/bin/env python3
# dedupe_merge.py
import re, hashlib
from typing import List, Dict,Any

def cheap_signature(question: str, answer: str) -> str:
    s = (question or "") + "||" + (answer or "")
    s_norm = re.sub(r'\d+', '<NUM>', s).lower()
    s_norm = re.sub(r'\s+', ' ', s_norm).strip()
    return hashlib.sha256(s_norm.encode('utf-8')).hexdigest()

def cheap_dedupe(candidates: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    seen = set()
    for obj in candidates:
        q = obj.get("question","")
        a = obj.get("answer","")
        sig = cheap_signature(q,a)
        if sig in seen:
            # if duplicate, merge provenance if available
            prev = next((x for x in out if cheap_signature(x.get("question",""), x.get("answer",""))==sig), None)
            if prev:
                prev.setdefault("source_chunk_ids", [])
                scid = obj.get("source_chunk_id")
                if scid and scid not in prev["source_chunk_ids"]:
                    prev["source_chunk_ids"].append(scid)
            continue
        seen.add(sig)
        obj.setdefault("source_chunk_ids", [obj.get("source_chunk_id")])
        out.append(obj)
    return out

# page-level supplement placeholder (you can implement full supplement that calls generate_for_chunk with concatenated text)
def page_level_supplement(concat_text: str, need:int, system_prompt:str, source_file:str):
    # Implement as needed; placeholder returns empty list for now
    return []
