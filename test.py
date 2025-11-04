import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from configs.config import DATA_PATH
from src.preprocessing import load_encoders
from src.utils import load_model
from src.preprocessing import load_data
from utils.logger import get_logger
from utils.custom_exception import CustomException


def prepare_features_for_saved_model(df: pd.DataFrame, encoders: dict):
    # Build X by dropping target and identifier
    X = df.drop(columns=[c for c in ["is_fake", "customer_id"] if c in df.columns]).copy()

    # Apply saved label encoders to categorical columns
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = le.transform(X[col].astype(str))

    return X


def main():
    logger = get_logger(__name__)
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Dataset not found. Run the app or pipeline to generate and train first.")

        df = load_data()
        if "is_fake" not in df.columns:
            raise CustomException("Target column 'is_fake' missing in dataset.")

        model = load_model()
        if model is None:
            raise CustomException("Trained model not found. Delete artifacts and run the app/pipeline to train.")

        encoders = load_encoders()
        X = prepare_features_for_saved_model(df, encoders)
        y = df["is_fake"]

        # Align features to model expectations
        for col in set(model.feature_names_in_) - set(X.columns):
            X[col] = 0
        X = X[model.feature_names_in_]

        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Accuracy (full dataset using saved model): {acc*100:.2f}%")

        # Also log detailed metrics
        cls = classification_report(y, preds)
        cm = confusion_matrix(y, preds)
        logger.info("Saved-model accuracy: %.4f", acc)
        logger.info("Classification report:\n%s", cls)
        logger.info("Confusion matrix:\n%s", cm)
    except Exception as e:
        logger.exception("Evaluation failed")
        raise CustomException("Error during saved model evaluation", e)


if __name__ == "__main__":
    main()
