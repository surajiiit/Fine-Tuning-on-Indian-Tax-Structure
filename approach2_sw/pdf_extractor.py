#!/usr/bin/env python3
# pdf_extractor.py
from pathlib import Path
import fitz  # PyMuPDF

def extract_text_per_page(pdf_path: Path):
    """
    Return list of dicts: { 'page_num': int (1-indexed), 'text': str }
    """
    pages = []
    doc = fitz.open(pdf_path.as_posix())
    for i in range(len(doc)):
        txt = doc[i].get_text("text") or ""
        pages.append({"page_num": i + 1, "text": txt})
    doc.close()
    return pages
