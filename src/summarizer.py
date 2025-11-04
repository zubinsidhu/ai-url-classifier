# src/summarizer.py
"""
Summarizer pipeline:
- chunk input text
- summarize each chunk using a HF summarization pipeline (local model)
- aggregate chunk summaries into a single final summary (concatenate then summarize)
- extract simple keywords (frequency-based after stopword filtering)
"""

from typing import List, Dict, Any
from transformers import pipeline, Pipeline
import math
import re
from collections import Counter
from src.chunker import chunk_text

# Lazy-init summarizer pipeline to avoid heavy import when not used
_SUMMARIZER: Pipeline | None = None

# get_summarizer function returns a transformers summarization pipeline.
# We use a small distilBART model by default.
# We return the pipeline.
def get_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6", device: int = -1):
    """
    Return a transformers summarization pipeline. Defaults to a small distilBART model - swap to larger model for higher quality
    device: -1 (cpu) or 0 (gpu)
    """
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline("summarization", model=model_name, device=device, truncation=True)
    return _SUMMARIZER

# clean_text_for_keywords function cleans the text for keywords.
# We lowercase the text, remove punctuation, and split the text into tokens.
# We return the tokens.
def clean_text_for_keywords(text: str) -> List[str]:
    # simple cleaning: lowercase, remove punctuation, split
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2]
    # micro stopword list
    STOP = {
        "the","and","for","that","with","this","from","they","have","were","their","them",
        "what","when","where","which","there","would","could","should","about","these",
        "those","after","before","more","over","such","also","other","some","your","you","are","was","but","not","you","all","any"
    }
    return [t for t in tokens if t not in STOP]

# top_keywords function returns the top keywords from the text.
# We clean the text for keywords, count the frequency of each token, and return the top keywords.
def top_keywords(text: str, top_k: int = 10) -> List[str]:
    tokens = clean_text_for_keywords(text)
    if not tokens:
        return []
    counts = Counter(tokens)
    most = counts.most_common(top_k)
    return [w for w, _ in most]

# summarize_text function summarizes the text.
# We chunk the text, summarize each chunk, concatenate the summaries, and summarize again to get the final summary.
# We return the final summary, the chunk summaries, the keywords, and the number of chunks.
def summarize_text(text: str,
                   summarizer_model: str = "sshleifer/distilbart-cnn-12-6",
                   max_chars_per_chunk: int = 4000,
                   overlap: int = 200,
                   device: int = -1) -> Dict[str, Any]:
    """
    Summarize `text` by:
      1) chunking
      2) summarizing each chunk
      3) concatenating chunk summaries and summarizing again (to get compact output)
    Returns:
      {
        "final_summary": "....",
        "chunk_summaries": [ "...", ... ],
        "keywords": [...],
        "num_chunks": N
      }
    """
    if not text:
        return {"final_summary": "", "chunk_summaries": [], "keywords": [], "num_chunks": 0}

    chunks = chunk_text(text, max_chars=max_chars_per_chunk, overlap=overlap)
    summarizer = get_summarizer(model_name=summarizer_model, device=device)

    chunk_summaries = []
    # Summarize each chunk (use small max_length/min_length to keep output concise)
    for idx, (start, end, chunk) in enumerate(chunks):
        # some models expect shorter inputs or will truncate; we still send each chunk
        try:
            out = summarizer(chunk, max_length=120, min_length=20, do_sample=False)[0]["summary_text"]
            chunk_summaries.append(out.strip())
        except Exception:
            # fallback: take the first 200 chars as a trivial "summary"
            fallback = (chunk[:200].strip() + "...") if len(chunk) > 200 else chunk.strip()
            chunk_summaries.append(fallback)

    # Aggregate: concatenate chunk summaries and summarize again to get final compact summary
    aggregated = "\n\n".join(chunk_summaries)
    try:
        final = summarizer(aggregated, max_length=180, min_length=30, do_sample=False)[0]["summary_text"].strip()
    except Exception:
        # if aggregation summary fails, join top N chunk summaries
        final = " ".join(chunk_summaries[:3]).strip()

    # keywords using the aggregated summary (fast heuristic)
    keywords = top_keywords(aggregated, top_k=15)

    return {
        "final_summary": final,
        "chunk_summaries": chunk_summaries,
        "keywords": keywords,
        "num_chunks": len(chunks),
    }
