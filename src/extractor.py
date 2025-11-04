# src/extractor.py
"""
Extraction utilities:
- primary: readability-lxml (Document)
- fallback: trafilatura
- final fallback: BeautifulSoup plain body text

Functions return (title, text) where title may be None.
"""
from readability import Document
from bs4 import BeautifulSoup
import trafilatura
from typing import Tuple, Optional

def extract_with_readability(html: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Use readability Document to extract main content and title.
    Return (title, text) or (None, None) if it fails or text is too short.
    """
    try:
        doc = Document(html.decode("utf-8", errors="ignore"))
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        title = doc.title() or None
        if text and len(text) > 50:
            return title, text
    except Exception:
        pass
    return None, None

def extract_with_trafilatura(html: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Try trafilatura extraction. trafilatura may return text without a title.
    """
    try:
        text = trafilatura.extract(html.decode("utf-8", errors="ignore"))
        if text and len(text) > 50:
            return None, text
    except Exception:
        pass
    return None, None

def extract_fallback(html: bytes) -> str:
    """
    Last resort: return visible text of body tag or whole document text.
    """
    soup = BeautifulSoup(html, "html.parser")
    if soup.body:
        return soup.body.get_text(separator="\n").strip()
    return soup.get_text(separator="\n").strip()

def extract_main(html: bytes) -> Tuple[Optional[str], str]:
    """
    Robust main extraction pipeline:
    1) readability
    2) trafilatura
    3) fallback body text
    Returns (title, text)
    """
    title, text = extract_with_readability(html)
    if text:
        return title, text
    t2, text2 = extract_with_trafilatura(html)
    if text2:
        # try to read <title> tag if readability didn't provide it
        if not t2:
            soup = BeautifulSoup(html, "html.parser")
            t = soup.title.string if soup.title and soup.title.string else None
            return t, text2
        return t2, text2
    # final fallback
    fallback_text = extract_fallback(html)
    soup = BeautifulSoup(html, "html.parser")
    t = soup.title.string if soup.title and soup.title.string else None
    return t, fallback_text
