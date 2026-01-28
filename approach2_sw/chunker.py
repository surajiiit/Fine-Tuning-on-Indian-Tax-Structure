#!/usr/bin/env python3
# chunker.py
import re
from typing import List, Dict, Any

HEADING_PATTERNS = [
    r'^\s*(#{1,6})\s+(.*)$',
    r'^\s*(\d+(\.\d+)*)\s+[-–—]?\s*(.+)$',
    r'^[A-Z][A-Z0-9 \-]{10,}$',
    r'^(.*):\s*$'
]
HEADING_REGEX = re.compile('|'.join(f'({p})' for p in HEADING_PATTERNS), flags=re.MULTILINE)

def find_headings(text: str):
    matches = []
    for m in HEADING_REGEX.finditer(text):
        s,e = m.start(), m.end()
        heading_line = text[s:e].strip()
        heading_line = re.sub(r'^[#\d\.\s:-]+', '', heading_line).strip()
        if heading_line:
            matches.append((s,e,heading_line))
    return matches

def chunk_text_heading_aware(text: str, target_chars:int=3600, overlap_chars:int=800, min_chars:int=600) -> List[Dict[str,Any]]:
    """
    Returns list of chunks: { chunk_text, char_start, char_end, chunk_index, heading }
    """
    T = (text or "").strip()
    if not T:
        return []
    headings = find_headings(T)
    boundaries = [0] + [s for s,_,_ in headings] + [len(T)]
    boundaries = sorted(set(boundaries))

    segments = []
    for i in range(len(boundaries)-1):
        seg_s, seg_e = boundaries[i], boundaries[i+1]
        seg_text = T[seg_s:seg_e].strip()
        if not seg_text:
            continue
        heading = None
        for hs,he,ht in reversed(headings):
            if hs <= seg_s:
                heading = ht
                break
        segments.append((seg_s, seg_e, heading, seg_text))

    chunks=[]
    idx=0
    for seg_s, seg_e, heading, seg_text in segments:
        seg_len = len(seg_text)
        if seg_len <= target_chars:
            chunks.append({'chunk_text': seg_text, 'char_start': seg_s, 'char_end': seg_e, 'chunk_index': idx, 'heading': heading})
            idx += 1
            continue
        start = 0
        while start < seg_len:
            end = min(seg_len, start + target_chars)
            body = seg_text[start:end].strip()
            if len(body) < min_chars and chunks:
                prev = chunks[-1]
                prev['chunk_text'] = prev['chunk_text'] + "\n\n" + body
                prev['char_end'] = seg_s + end
                break
            chunks.append({'chunk_text': body, 'char_start': seg_s + start, 'char_end': seg_s + end, 'chunk_index': idx, 'heading': heading})
            idx += 1
            if end >= seg_len:
                break
            start = end - overlap_chars
            if start < 0:
                start = 0
    return chunks
