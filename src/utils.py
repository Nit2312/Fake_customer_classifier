import os
import joblib
from configs.config import MODEL_PATH, ENCODER_PATH

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"[MODEL] Saved model to {MODEL_PATH}")

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[MODEL] Loaded model from {MODEL_PATH}")
        return model
    else:
        print("[MODEL] No trained model found.")
        return None
