# src/search.py
"""
Semantic search module (SQLite JSON-stored embeddings -> cosine search).
- Uses sentence-transformers embeddings already stored in Page.embedding (list of floats).
- Falls back to computing embedding for the query via src.embeddings.embed_single.
- Returns top_k results sorted by descending similarity score.

Design notes:
- For performance at scale, replace this module with a vector DB (Qdrant, Milvus).
- The similarity metric is cosine similarity.
- For numeric stability we normalize vectors before dot product.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
from src.embeddings import embed_single
from src.models import init_db, Page
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SessionLocal = init_db()

def _to_numpy(vec):
    """Convert list-like to numpy float vector, return None for invalid."""
    if vec is None:
        return None
    try:
        arr = np.array(vec, dtype=float)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two numpy vectors.
    Returns value in [-1, 1]. If either is zero-vector, returns -1.
    """
    if a is None or b is None:
        return -1.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def _fetch_pages_with_embeddings(session: Session) -> List[Page]:
    """
    Return all Page rows that have a non-null embedding (not empty).
    For big datasets, replace with batched query or a vector DB.
    """
    # SQLAlchemy JSON column may be stored as text in sqlite; still query for non-null
    return session.query(Page).filter(Page.embedding != None).all()

def search_pages(query: str, top_k: int = 5, embedding_model: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """
    Semantic search entrypoint.
    - compute query embedding
    - compute cosine similarity against stored embeddings
    - return list of dicts: [{ 'page': Page, 'score': float }, ...] sorted desc by score
    """
    if not query or not query.strip():
        return []

    # compute query embedding
    q_emb_list = embed_single(query, model_name=embedding_model)
    q_emb = _to_numpy(q_emb_list)
    if q_emb is None:
        logger.error("Failed to compute query embedding")
        return []

    session = SessionLocal()
    try:
        pages = _fetch_pages_with_embeddings(session)
        results: List[Tuple[float, Page]] = []
        for p in pages:
            p_emb_np = _to_numpy(p.embedding)
            if p_emb_np is None:
                continue
            score = cosine_similarity(q_emb, p_emb_np)
            results.append((score, p))
        # sort descending by score
        results.sort(key=lambda x: x[0], reverse=True)
        # filter out low or negative scores? We leave that to caller; but remove -1 sentinel.
        filtered = [(s, pg) for s, pg in results if s > -0.0]  # keep anything >= 0
        top = filtered[:top_k]
        return [{"url": pg.url, "title": pg.title, "score": float(s), "summary": getattr(pg, "summary_text", None)} for s, pg in top]
    finally:
        session.close()
