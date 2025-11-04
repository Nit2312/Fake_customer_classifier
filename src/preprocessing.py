import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from configs.config import ENCODER_PATH, DATA_PATH

def load_data():
    return pd.read_csv(DATA_PATH)

def preprocess_data(df):
    target_col = "is_fake"
    # Drop identifier-like columns that should not be used as predictive features
    drop_cols = [target_col]
    if "customer_id" in df.columns:
        drop_cols.append("customer_id")
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    cat_cols = X.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, y, label_encoders

def save_encoders(label_encoders):
    joblib.dump(label_encoders, ENCODER_PATH)
    print(f"[ENCODER] Saved label encoders to {ENCODER_PATH}")

def load_encoders():
    return joblib.load(ENCODER_PATH)
