import os

# Project directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# File paths
DATA_PATH = os.path.join(DATA_DIR, "synthetic_customer_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "customer_fraud_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 300
FAKE_PERCENTAGE = 0.15
NUM_SAMPLES = 10000
CASES_PER_CUSTOMER = 7

# Create dirs if missing
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
