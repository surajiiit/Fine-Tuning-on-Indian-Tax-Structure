#!/usr/bin/env python3
# cleanup.py
import re
from collections import Counter
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join([ln.strip() for ln in text.splitlines()]).strip()

def detect_repeated_lines_across_pages(pages: List[Dict[str,Any]], min_occurrence: int = 3, top_k:int=50) -> List[str]:
    """
    pages: list of {'page_num','text'}
    Return common lines (candidate headers/footers).
    """
    counter = Counter()
    for p in pages:
        lines = [ln.strip() for ln in clean_text(p['text']).splitlines() if ln.strip()]
        unique = set(lines)
        for ln in unique:
            counter[ln] += 1
    candidates = [ln for ln,c in counter.items() if c >= min_occurrence]
    # prefer longer repeated lines (likely headers/footers)
    candidates = sorted(candidates, key=lambda s: (-counter[s], -len(s)))[:top_k]
    return candidates

def strip_headers_footers(text: str, hf_patterns: List[str]) -> str:
    if not hf_patterns:
        return text or ""
    hf_set = set([p.strip() for p in hf_patterns if p and p.strip()])
    lines = (text or "").splitlines()
    new_lines = [ln for ln in lines if ln.strip() not in hf_set]
    return "\n".join(new_lines)
