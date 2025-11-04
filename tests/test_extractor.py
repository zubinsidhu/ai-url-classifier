# tests/test_extractor.py
from src.extractor import extract_main

SIMPLE_HTML = b"""
<html>
  <head><title>Test Page</title></head>
  <body>
    <h1>Heading</h1>
    <p>This is a short article body. Hello world.</p>
  </body>
</html>
"""

def test_extract_main_simple():
    title, text = extract_main(SIMPLE_HTML)
    assert title is not None
    assert "Test Page" in title
    assert "This is a short article body" in text
