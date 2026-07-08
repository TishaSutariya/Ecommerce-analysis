import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "raw" / "Funnel_Analysis_Data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "funnel_cleaned.csv"

df = pd.read_csv(INPUT_PATH)
print("Loaded data:", df.shape)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

if "event_time" in df.columns:
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    if df[col].isnull().any():
        if col == "revenue":
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates()

event_to_stage = {
    "visit": "Visit Website",
    "page_view": "Visit Website",
    "browse": "Visit Website",
    "signup": "Signup",
    "sign_up": "Signup",
    "register": "Signup",
    "add_to_cart": "Add to Cart",
    "cart": "Add to Cart",
    "checkout": "Checkout",
    "purchase": "Purchase",
    "buy": "Purchase",
    "order": "Purchase",
}

if "event" in df.columns:
    df["event"] = df["event"].astype(str).str.strip().str.lower()
    df["funnel_stage"] = df["event"].map(event_to_stage).fillna("Other")
else:
    df["funnel_stage"] = "Other"

stage_order = {
    "Visit Website": 1,
    "Signup": 2,
    "Add to Cart": 3,
    "Checkout": 4,
    "Purchase": 5,
    "Other": 0,
}

df["funnel_rank"] = df["funnel_stage"].map(stage_order)

if "user_id" in df.columns:
    sort_cols = ["user_id"]
    if "event_time" in df.columns:
        sort_cols.append("event_time")
    sort_cols.append("funnel_rank")
    df = df.sort_values(sort_cols)

    df["prev_rank"] = df.groupby("user_id")["funnel_rank"].shift(1)
    df["valid_sequence"] = np.where(df["prev_rank"].isna() | (df["funnel_rank"] >= df["prev_rank"]), 1, 0)

if "device" in df.columns:
    df["device_type"] = df["device"]

if "region" in df.columns:
    df["location"] = df["region"]

if "channel" in df.columns:
    df["traffic_source"] = df["channel"]

if "revenue" in df.columns:
    df["purchase_amount"] = df["revenue"]
else:
    df["purchase_amount"] = 0

if "funnel_rank" in df.columns and "user_id" in df.columns:
    user_max_rank = df.groupby("user_id")["funnel_rank"].max()
    bounced_users = user_max_rank[user_max_rank == 1].index
    df["bounced"] = df["user_id"].isin(bounced_users).astype(int)
else:
    df["bounced"] = 0

if "event_time" in df.columns and "user_id" in df.columns:
    df["session_duration"] = df.groupby("user_id")["event_time"].transform(
        lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 60
    )
else:
    df["session_duration"] = 60

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Saved cleaned file to:", OUTPUT_PATH)
print("Final shape:", df.shape)