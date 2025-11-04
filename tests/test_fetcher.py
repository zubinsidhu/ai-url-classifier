from src.fetcher import fetch_and_store

def test_fetcher_main_simple():
    url = "https://en.wikipedia.org/wiki/Large_language_model"
    res = fetch_and_store(url)
    print(res)