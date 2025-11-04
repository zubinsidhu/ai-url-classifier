# src/fetcher.py
"""
Synchronous Playwright fetcher that:
- checks robots.txt (best-effort)
- navigates with timeout
- stores raw HTML to disk
- extracts main content and stores into DB via SQLAlchemy models
- safe duplicate handling (INSERT or UPDATE)
- logging and simple retry logic
"""
import os
import hashlib
import time
import urllib.robotparser
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from src.config import cfg
from src.extractor import extract_main
from src.models import Page, init_db
from sqlalchemy.exc import IntegrityError
from langdetect import detect, LangDetectException
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

SessionLocal = init_db()

# is_allowed_by_robots function checks if the URL is allowed by robots.txt.
# If not, we return False.
# If allowed, we return True.

def is_allowed_by_robots(url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = f"{base}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(cfg.USER_AGENT, url)
        logger.debug("robots.txt check for %s -> %s", url, allowed)
        return allowed
    except Exception as exc:
        logger.warning("robots.txt read failed for %s: %s", robots_url, exc)
        # fallback behavior based on config flag
        return cfg.ALLOW_ROBOTS_FALLBACK

# _write_raw function writes the raw HTML to the disk.
# We use the SHA-256 hash of the HTML as the filename.
# We make the directory if it doesn't exist.
# We return the path to the file.

def _write_raw(html: bytes) -> str:
    os.makedirs(cfg.RAW_DIR, exist_ok=True)
    digest = hashlib.sha256(html).hexdigest()
    path = os.path.join(cfg.RAW_DIR, f"{digest}.html")
    with open(path, "wb") as f:
        f.write(html)
    return path

# _store_page_record function stores the page record in the database.
# We use the URL as the primary key.
# We update the existing record if it already exists.
# We return the path to the file.

def _store_page_record(url: str, status: int, title: str, lang: str, excerpt: str, raw_path: str):
    session = SessionLocal()
    try:
        pmodel = Page(
            url=url,
            status_code=status,
            title=title,
            lang=lang,
            text_excerpt=excerpt,
            html_path=raw_path,
            raw_meta={"fetched_by": "playwright"}
        )
        session.add(pmodel)
        session.commit()
    except IntegrityError:
        session.rollback()
        logger.info("Duplicate URL found, updating existing record: %s", url)
        existing = session.query(Page).filter(Page.url == url).first()
        if existing:
            existing.status_code = status
            existing.title = title
            existing.lang = lang
            existing.text_excerpt = excerpt
            existing.html_path = raw_path
            session.commit()
    finally:
        session.close()

def fetch_and_store(url: str) -> dict:
    """
    Fetch URL with Playwright and store a Page record.
    Returns a dict summary with status or error info.
    """
    if not is_allowed_by_robots(url):
        logger.info("Blocked by robots.txt: %s", url)
        return {"url": url, "status": "blocked_by_robots"}

    attempt = 0
    last_err = None
    while attempt <= cfg.FETCH_RETRIES:
        attempt += 1
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=cfg.USER_AGENT)
                page = context.new_page()
                logger.info("Fetching %s (attempt %d)", url, attempt)
                response = page.goto(url, timeout=cfg.PLAYWRIGHT_TIMEOUT_MS)
                status = response.status if response else None
                html = page.content().encode("utf-8")
                raw_path = _write_raw(html)
                title, text = extract_main(html)
                try:
                    lang = detect(text) if text and len(text) > 50 else None
                except LangDetectException:
                    lang = None
                excerpt = (text[:cfg.MAX_TEXT_SIZE]) if text else ""
                _store_page_record(url, status, title or None, lang, excerpt, raw_path)
                browser.close()
                logger.info("Fetched %s status=%s len_text=%d lang=%s", url, status, len(excerpt), lang)
                return {"url": url, "status_code": status, "len_text": len(excerpt), "lang": lang}
        except PlaywrightTimeoutError as e:
            last_err = str(e)
            logger.warning("Timeout fetching %s: %s", url, e)
        except Exception as e:
            last_err = str(e)
            logger.exception("Error fetching %s: %s", url, e)
        time.sleep(1.0)  # simple backoff between attempts

    logger.error("All fetch attempts failed for %s: %s", url, last_err)
    return {"url": url, "status": "error", "error": last_err}
