import pandas as pd
import sqlite3

df = pd.read_csv("data/processed/funnel_cleaned.csv")

conn = sqlite3.connect("funnel.db")

df.to_sql("funnel_data", conn, if_exists="replace", index=False)

conn.close()

print("Database created successfully.")