# src/config.py
"""
Configuration data for the fetch/extract pipeline.
Settings are read from environment variables with sensible defaults.
"""
from dataclasses import dataclass
import os

@dataclass
class Config:
    DB_URL: str = os.getenv("DB_URL", "sqlite:///pages.db")
    RAW_DIR: str = os.getenv("RAW_DIR", "raw")
    USER_AGENT: str = os.getenv("USER_AGENT", "WebClassifierBot/1.0 (+https://example.org/bot)")
    PLAYWRIGHT_TIMEOUT_MS: int = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "30000"))
    FETCH_RETRIES: int = int(os.getenv("FETCH_RETRIES", "2"))
    RATE_LIMIT_SECONDS: float = float(os.getenv("RATE_LIMIT_SECONDS", "0.5"))
    MAX_TEXT_SIZE: int = int(os.getenv("MAX_TEXT_SIZE", "200000"))  # chars to store
    ALLOW_ROBOTS_FALLBACK: bool = os.getenv("ALLOW_ROBOTS_FALLBACK", "true").lower() == "true"

cfg = Config()
