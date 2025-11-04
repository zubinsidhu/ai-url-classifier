# tests/test_chunker.py
from src.chunker import chunk_text

def test_chunker_basic():
    text = "a" * 10000
    chunks = chunk_text(text, max_chars=3000, overlap=100)
    assert len(chunks) >= 3
    # ensure overlaps
    first_start, first_end, _ = chunks[0]
    second_start, second_end, _ = chunks[1]
    assert second_start < first_end
