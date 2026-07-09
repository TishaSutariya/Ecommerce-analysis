import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "funnel_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

if "device" in df.columns and "device_type" not in df.columns:
    df["device_type"] = df["device"]

if "channel" in df.columns and "traffic_source" not in df.columns:
    df["traffic_source"] = df["channel"]

if "region" in df.columns and "location" not in df.columns:
    df["location"] = df["region"]

if "bounced" not in df.columns:
    df["bounced"] = 0

if "funnel_stage" in df.columns:
    df["purchase_flag"] = (df["funnel_stage"] == "Purchase").astype(int)
else:
    df["purchase_flag"] = 0

if "user_id" not in df.columns:
    raise ValueError("user_id column is required.")

user_target = df.groupby("user_id")["purchase_flag"].max().reset_index()
user_target.columns = ["user_id", "converted"]

agg_map = {}

for col in ["device_type", "traffic_source", "location"]:
    if col in df.columns:
        agg_map[col] = "first"

for col in ["bounced", "session_duration", "page_views", "scroll_depth", "session_count", "visit_count", "days_since_last_visit", "engagement_score"]:
    if col in df.columns:
        agg_map[col] = "mean" if col == "session_duration" else "sum"

user_df = df.groupby("user_id").agg(agg_map).reset_index()
user_df = user_df.merge(user_target, on="user_id", how="left")

label_encoders = {}

for col in ["device_type", "traffic_source", "location"]:
    if col in user_df.columns:
        user_df[col] = user_df[col].astype(str).fillna("Unknown")
        le = LabelEncoder()
        user_df[col + "_encoded"] = le.fit_transform(user_df[col])
        label_encoders[col] = le

feature_cols = [c for c in user_df.columns if c.endswith("_encoded") or c in ["bounced", "session_duration", "page_views", "scroll_depth", "session_count", "visit_count", "days_since_last_visit", "engagement_score"]]
X = user_df[feature_cols].fillna(0)
y = user_df["converted"].fillna(0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "test_accuracy": accuracy_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred, zero_division=0),
    "test_recall": recall_score(y_test, y_pred, zero_division=0),
    "test_f1": f1_score(y_test, y_pred, zero_division=0),
    "test_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "n_users": int(len(user_df)),
    "n_converted": int(user_df["converted"].sum())
}

payload = {
    "model": model,
    "feature_cols": feature_cols,
    "label_encoders": label_encoders,
    "metrics": metrics
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)

print("Model trained and saved successfully.")
print(metrics)