# src/search_demo.py
"""
Interactive demo: run a query and print top-k pages.
Usage:
    PYTHONPATH=$(pwd) python src/search_demo.py "natural language query" --k 5
"""
import argparse
from src.search import search_pages
import json

# main function runs the search demo.
# We parse the command line arguments and run the search.
# We return the results.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query text (in quotes)")
    parser.add_argument("--k", type=int, default=5, help="Top-K results")
    args = parser.parse_args()

    results = search_pages(args.query, top_k=args.k)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
