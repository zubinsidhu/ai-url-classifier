# src/chunker.py
"""
Simple, robust chunker for long text.
Breaks by characters with overlap (fast and deterministic).
Designed to avoid adding tokenizers as a dependency in Step 2.
"""

from typing import List, Tuple

def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> List[Tuple[int,int,str]]:
    """
    Break `text` into chunks up to max_chars characters with `overlap` chars overlap.
    Returns list of tuples (start_idx, end_idx, chunk_text).
    """
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        overlap = 0
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = start + max_chars
        if end >= n:
            end = n
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
