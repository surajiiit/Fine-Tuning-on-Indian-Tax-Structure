#!/usr/bin/env python3
# run_pipeline.py
"""
Run PDF chunking and QA pipeline (modular).
Defaults set to your FineTuning paths.
"""

import argparse
import json
import time
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any

# local modules (must be in same directory or on PYTHONPATH)
from pdf_extractor import extract_text_per_page
from cleanup import clean_text, detect_repeated_lines_across_pages, strip_headers_footers
from chunker import chunk_text_heading_aware
from chunks_io import append_jsonl, make_chunk_id
from qa_worker import generate_for_chunk
from dedupe_merge import cheap_dedupe, page_level_supplement

# Defaults (update if you want different locations)
DEFAULT_INPUT_DIR = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/data")
DEFAULT_CHUNKS = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/chunks.jsonl")
DEFAULT_QA = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/qa_dataset.jsonl")
DEFAULT_SYSTEM_PROMPT = Path("/fab3/btech/2022/suraj.yadav22b/FineTuning/system_prompt.txt")


def emit_chunks_for_pdf(pdf_path: Path, out_chunks: Path, target_chars=3600, overlap_chars=800, min_chars=600) -> int:
    pages = extract_text_per_page(pdf_path)
    hf_cands = detect_repeated_lines_across_pages(pages)
    cleaned_pages = []
    for p in pages:
        txt = strip_headers_footers(p['text'], hf_cands)
        cleaned_pages.append({'page_num': p['page_num'], 'text': clean_text(txt)})
    combined_parts: List[str] = []
    for p in cleaned_pages:
        combined_parts.append(f"\n\n[PAGE {p['page_num']}]\n\n")
        combined_parts.append(p['text'])
    combined_text = "".join(combined_parts)
    # chunk
    chunks = chunk_text_heading_aware(combined_text, target_chars=target_chars, overlap_chars=overlap_chars, min_chars=min_chars)
    # compute page spans
    page_markers = []
    for m in re.finditer(r'\[PAGE (\d+)\]', combined_text):
        page_markers.append((int(m.group(1)), m.start()))
    page_spans = []
    for idx, (pnum, start_idx) in enumerate(page_markers):
        s = start_idx
        e = page_markers[idx+1][1] if idx+1 < len(page_markers) else len(combined_text)
        page_spans.append((pnum, s, e))
    # write chunks
    for c in chunks:
        cs = int(c['char_start']); ce = int(c['char_end'])
        pages_overlap = [pnum for pnum, s, e in page_spans if not (ce <= s or cs >= e)]
        chunk_id = make_chunk_id(pdf_path.name, c['chunk_index'])
        record = {
            "chunk_id": chunk_id,
            "source_file": pdf_path.name,
            "section_title": c.get('heading') or None,
            "page_range": pages_overlap if pages_overlap else None,
            "char_range": [cs, ce],
            "text": c['chunk_text']
        }
        append_jsonl(out_chunks, record)
    return len(chunks)


def generate_qa_from_chunks(chunks_jsonl: Path, out_qa_jsonl: Path, system_prompt_path: Path, cuda_device: str = None):
    """
    Generate Q/A for all chunks in chunks_jsonl -> append to out_qa_jsonl.
    Prints progress and ETA to stdout (useful with nohup).
    """
    # load system prompt
    if not system_prompt_path.exists():
        raise SystemExit(f"Missing system prompt at {system_prompt_path}")
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    # group chunks by file
    file_chunks: Dict[str, List[Dict[str, Any]]] = {}
    with chunks_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            file_chunks.setdefault(rec['source_file'], []).append(rec)

    # pre-count for ETA
    total_files = len(file_chunks)
    total_chunks = sum(len(v) for v in file_chunks.values())
    processed_chunks = 0

    print(f"[INIT] Total Files: {total_files}, Total Chunks: {total_chunks}")

    start_time = time.time()

    # for each file: generate qas per chunk then dedupe and write final lines
    for idx_file, (source_file, chunks) in enumerate(file_chunks.items(), 1):

        print(f"[FILE {idx_file}/{total_files}] {source_file} → {len(chunks)} chunks")

        all_qas: List[Dict[str, Any]] = []
        chunks_sorted = sorted(chunks, key=lambda r: r['chunk_id'])

        for c in chunks_sorted:
            t0 = time.time()

            # ----- generate
            qas = generate_for_chunk(
                system_prompt,
                source_file,
                c['chunk_id'],
                c['text'],
                cuda_device=cuda_device
            )

            # ensure provenance
            for q in qas:
                if isinstance(q, dict):
                    q.setdefault("source_chunk_id", c['chunk_id'])
                    q.setdefault("source_file", source_file)
            all_qas.extend([q for q in qas if isinstance(q, dict)])

            processed_chunks += 1
            elapsed = time.time() - start_time
            speed = processed_chunks / elapsed if elapsed > 0 else 0
            remaining = (total_chunks - processed_chunks) / speed if speed > 0 else 0

            # ETA print every chunk (stdout -> captured by nohup)
            print(
                f"[CHUNK {processed_chunks}/{total_chunks}] "
                f"Time/Chunk: {time.time()-t0:.1f}s  "
                f"Elapsed: {elapsed/60:.1f}m  "
                f"ETA: {remaining/60:.1f}m"
            )

        # cheap dedupe
        deduped = cheap_dedupe(all_qas)

        # fallback supplement
        if len(deduped) < 6:
            concat_text = "\n\n".join([c['text'] for c in chunks_sorted])
            need = 6 - len(deduped)
            extras = page_level_supplement(concat_text, need, system_prompt, source_file)
            for e in extras:
                e.setdefault("source_chunk_id", "page_level_supplement")
                e.setdefault("source_file", source_file)
            deduped.extend(extras)

        # light validation and disclaimers
        for obj in deduped:
            obj.setdefault("id", obj.get("id") or str(uuid.uuid4()))
            obj.setdefault("difficulty", obj.get("difficulty", "low"))
            obj.setdefault("tags", obj.get("tags", []))
            obj.setdefault("inferred", bool(obj.get("inferred", False)))
            obj.setdefault("example", bool(obj.get("example", False)))
            if obj.get("inferred", False) or obj.get("difficulty") == "high":
                if not obj.get("answer", "").strip().endswith(
                        "Disclaimer: For personalized tax advice, consult a qualified tax professional."):
                    obj["answer"] = obj.get("answer", "").strip() + \
                                    "\n\nDisclaimer: For personalized tax advice, consult a qualified tax professional."

        # write
        out_qa_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_qa_jsonl.open("a", encoding="utf-8") as fh:
            for obj in deduped:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"[DONE] All QA generated in {elapsed/60:.1f} minutes.")


def main():
    parser = argparse.ArgumentParser(description="Run PDF chunking and QA pipeline (modular).")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Input PDF directory")
    parser.add_argument("--chunks-out", type=str, default=str(DEFAULT_CHUNKS), help="Output chunks.jsonl")
    parser.add_argument("--qa-out", type=str, default=str(DEFAULT_QA), help="Output qa_dataset.jsonl")
    parser.add_argument("--system-prompt", type=str, default=str(DEFAULT_SYSTEM_PROMPT), help="System prompt file")
    parser.add_argument("--emit-chunks-only", action="store_true", help="Only emit chunks.jsonl")
    parser.add_argument("--generate-qa", action="store_true", help="Generate Q/A from existing chunks.jsonl")
    parser.add_argument("--target-chars", type=int, default=3600, help="Target chunk length in chars (approx)")
    parser.add_argument("--overlap-chars", type=int, default=800, help="Overlap length in chars")
    parser.add_argument("--min-chars", type=int, default=600, help="Minimum chunk length in chars")
    parser.add_argument("--cuda-device", type=str, default=None, help="CUDA_VISIBLE_DEVICES for Ollama (e.g. '0')")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_chunks = Path(args.chunks_out)
    out_qa = Path(args.qa_out)
    system_prompt = Path(args.system_prompt)

    if args.emit_chucks_only if False else args.emit_chunks_only:  # backward-safe guard
        pdfs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"])
        if not pdfs:
            print("No PDFs found in", input_dir)
            return
        if out_chunks.exists():
            print("Appending to existing chunks file:", out_chunks)
        for pdf in pdfs:
            print("Processing:", pdf.name)
            try:
                n = emit_chunks_for_pdf(pdf, out_chunks, target_chars=args.target_chars, overlap_chars=args.overlap_chars, min_chars=args.min_chars)
                print(f"Wrote {n} chunks for {pdf.name}")
            except Exception as e:
                print(f"Skipping {pdf.name} due to error: {e}")
        print("Chunk emission complete.")
        return

    if args.generate_qa:
        if not out_chunks.exists():
            raise SystemExit(f"chunks.jsonl not found at {out_chunks}. Run with --emit-chunks-only first.")
        generate_qa_from_chunks(out_chunks, out_qa, system_prompt, cuda_device=args.cuda_device)
        print("QA generation complete.")
        return

    # default: run both steps sequentially
    pdfs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print("No PDFs found in", input_dir)
        return
    # emit chunks
    if out_chunks.exists():
        print("Appending to existing chunks file:", out_chunks)
    for pdf in pdfs:
        print("Processing:", pdf.name)
        try:
            emit_chunks_for_pdf(pdf, out_chunks, target_chars=args.target_chars, overlap_chars=args.overlap_chars, min_chars=args.min_chars)
        except Exception as e:
            print(f"Skipping {pdf.name} due to error during chunk emission: {e}")
            continue
    print("Chunk emission complete.")
    # generate QA
    generate_qa_from_chunks(out_chunks, out_qa, system_prompt, cuda_device=args.cuda_device)
    print("Full pipeline complete. QA at:", out_qa)


if __name__ == "__main__":
    main()
