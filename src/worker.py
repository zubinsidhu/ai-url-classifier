# src/worker.py
"""
Simple CLI worker to process a newline-separated URL file.
This runs synchronously, one URL at a time, and honors a small rate limit.
"""
import time
import argparse
from src.fetcher import fetch_and_store
from src.config import cfg
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    logger.addHandler(ch)

# main function processes the URLs one by one.
# We log the URL and the result.
# We sleep for the rate limit.

def main(urls):
    for u in urls:
        logger.info("[worker] processing %s", u)
        res = fetch_and_store(u)
        logger.info("[worker] result: %s", res)
        time.sleep(cfg.RATE_LIMIT_SECONDS)

# If the script is run directly, we parse the command line arguments and process the URLs.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & extract pages from a list of URLs")
    parser.add_argument("urlfile", help="file with newline-separated URLs")
    args = parser.parse_args()
    with open(args.urlfile) as f:
        urls = [l.strip() for l in f.readlines() if l.strip()]
    main(urls)
