# src/embeddings.py
"""
Embeddings generator using sentence-transformers.
Stores embeddings as JSON array strings in the DB (for SQLite MVP).
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List

_MODEL: SentenceTransformer | None = None

# get_embedding_model function returns a sentence-transformers model.
# We use a small all-MiniLM-L6-v2 model by default.
# We return the model.
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

# embed_text function returns a list of embeddings for each text in `texts`.
# We use the sentence-transformers model to encode the text.
# We return the embeddings as a list of lists.
def embed_text(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Returns list of embeddings (as python lists) for each text in `texts`.
    """
    if not texts:
        return []
    model = get_embedding_model(model_name=model_name)
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # convert to plain Python lists for JSON storage
    return [emb.astype(float).tolist() for emb in embs]

# embed_single function returns a single embedding for a single text.
# We use the sentence-transformers model to encode the text.
# We return the embedding as a list.
def embed_single(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Returns single embedding (as python list) for `text`.
    """
    out = embed_text([text], model_name=model_name)
    return out[0] if out else []
