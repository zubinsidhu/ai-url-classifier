import sqlite3
import pandas as pd

conn = sqlite3.connect("pages.db")

df = pd.read_sql("SELECT id, summary_text as text FROM pages WHERE summary_text IS NOT NULL", conn)
df["labels"] = ""   # blank column for you to fill

df.to_csv("train.csv", index=False)
print("train.csv written")