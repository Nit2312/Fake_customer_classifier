import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from configs.config import (
    DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS
)
from src.data_generator import generate_synthetic_data
from src.preprocessing import load_data, preprocess_data, save_encoders, load_encoders
from src.utils import save_model, load_model
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def train_pipeline():
    try:
        # Step 1: Check if data exists
        if not os.path.exists(DATA_PATH):
            logger.info("[PIPELINE] Data not found, generating new synthetic data...")
            generate_synthetic_data(DATA_PATH)
        else:
            logger.info("[PIPELINE] Data found, loading existing file...")

        df = load_data()

        # Step 2: Check for model existence
        if os.path.exists(MODEL_PATH):
            logger.info("[PIPELINE] Existing model detected, skipping training.")
            model = load_model()
            encoders = load_encoders()
            return model, encoders, df.drop(columns=["is_fake"]).columns.tolist()

        logger.info("[PIPELINE] No model found. Starting training process...")

        # Step 3: Preprocess data
        X, y, encoders = preprocess_data(df)

        # Step 4: Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Step 5: Train model (XGBoost)
        model = XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist"
        )
        model.fit(X_train, y_train)

        # Step 6: Evaluate
        preds = model.predict(X_test)
        logger.info("Training Complete")
        logger.info("Classification Report:\n%s", classification_report(y_test, preds))
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))

        # Step 7: Save model & encoders
        save_model(model)
        save_encoders(encoders)

        logger.info("Pipeline complete. Model and encoders ready!")
        return model, encoders, X.columns.tolist()
    except Exception as e:
        logger.exception("Pipeline failed")
        raise CustomException("Error in training pipeline", e)

if __name__ == "__main__":
    train_pipeline()

# Backward compatibility for app import
def train_model(*args, **kwargs):
    model, encoders, _ = train_pipeline()
    return model, encoders
