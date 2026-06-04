import sqlite3
import pandas as pd
import os

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

CSV_PATH = os.path.join(project_root, 'data', 'raw', 'user_funnel_events.csv')
DB_PATH = os.path.join(project_root, 'data', 'ecommerce_data.db')

print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
print(f"✅ Loaded {len(df)} rows")

print(f"Creating database: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
print("Creating user_events table...")
df.to_sql('user_events', conn, if_exists='replace', index=False)
print(f"✅ Table created with {len(df)} rows")

# Verify
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM user_events")
print(f"Verified: {cursor.fetchone()[0]} rows")

conn.close()
print("✅ Database ready! Now run your SQL queries.")