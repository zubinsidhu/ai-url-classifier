# src/process_summary.py
from src.models import init_db, Page
from src.summarizer import summarize_text
from src.embeddings import embed_single
from sqlalchemy.orm import Session
import json
from src.config import cfg
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SessionLocal = init_db()

# process_unprocessed function processes the unprocessed pages.
# We query the database for pages with no summary text or summary text is empty.
# We summarize the text, embed the summary, and store the summary in the database.
# We return the number of pages processed.
def process_unprocessed(limit: int = 20):
    session = SessionLocal()
    try:
        q = session.query(Page).filter((Page.summary_text == None) | (Page.summary_text == "")).limit(limit)
        pages = q.all()
        logger.info("Found %d pages to summarize", len(pages))
        for p in pages:
            if not p.text_excerpt:
                logger.info("Skipping page with empty text: %s", p.url)
                continue
            try:
                out = summarize_text(p.text_excerpt)
                p.summary_text = out["final_summary"]
                p.summary_keywords = out["keywords"]
                emb = embed_single(p.summary_text or p.text_excerpt)
                p.embedding = emb
                session.add(p)
                session.commit()
                logger.info("Summarized & embedded %s (chunks=%d)", p.url, out["num_chunks"])
            except Exception as e:
                session.rollback()
                logger.exception("Failed to process page %s: %s", p.url, e)
    finally:
        session.close()

# If the script is run directly, we process the unprocessed pages.
if __name__ == "__main__":
    process_unprocessed(limit=50)
