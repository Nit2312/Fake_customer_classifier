import streamlit as st
import os
import pandas as pd
import joblib

from src.data_generator import generate_synthetic_data
from pipeline.train_pipeline import train_model
from configs.config import DATA_PATH, MODEL_PATH, ENCODER_PATH
from utils.logger import get_logger
from utils.custom_exception import CustomException

# --------------------------------------------
# 1️⃣  PAGE SETUP
# --------------------------------------------
st.set_page_config(
    page_title="Fake Customer Classifier",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.title("🧠 Fake Customer Classifier Dashboard")
st.markdown("""
Welcome to the **Fake Customer Classifier System**.  
This app allows you to:
1. Generate or load synthetic customer data  
2. Train a fraud detection model  
3. Predict whether a new customer is **Fake (1)** or **Genuine (0)**
""")

# --------------------------------------------
# 2️⃣  CONFIG & PATHS
# --------------------------------------------
logger = get_logger(__name__)
data_path = DATA_PATH
model_path = MODEL_PATH
encoders_path = ENCODER_PATH

# --------------------------------------------
# 3️⃣  LOAD OR GENERATE DATA
# --------------------------------------------
try:
    if not os.path.exists(data_path):
        st.warning("⚠️ Data file not found! Generating new synthetic dataset...")
        generate_synthetic_data(data_path)
        df = pd.read_csv(data_path)
        st.success("✅ Synthetic dataset generated successfully.")
        logger.info("Synthetic dataset generated at %s with %d records", data_path, len(df))
    else:
        df = pd.read_csv(data_path)
        st.success(f"✅ Loaded dataset: `{data_path}` ({len(df)} records)")
        logger.info("Loaded dataset from %s with %d records", data_path, len(df))
except Exception as e:
    logger.exception("Failed during data load/generation")
    raise CustomException("Data load/generation failed", e)

st.dataframe(df.head())

# --------------------------------------------
# 4️⃣  LOAD OR TRAIN MODEL
# --------------------------------------------
try:
    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        st.warning("⚠️ Model or encoders not found. Training a new model...")
        model, encoders = train_model(data_path)
        st.success("✅ Model trained and saved successfully.")
        logger.info("Model trained and saved.")
    else:
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        st.success("✅ Model and encoders loaded successfully.")
        logger.info("Model and encoders loaded from disk.")
except Exception as e:
    logger.exception("Failed during model load/train")
    raise CustomException("Model load/train failed", e)

# --------------------------------------------
# 5️⃣  SIDEBAR NAVIGATION
# --------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard Overview", "Customer Prediction"]
)

# --------------------------------------------
# 6️⃣  DASHBOARD OVERVIEW
# --------------------------------------------
if page == "Dashboard Overview":
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(20))

    st.write("### 🔹 Basic Statistics")
    st.dataframe(df.describe())

    st.write("### 🔹 Label Distribution")
    st.bar_chart(df["is_fake"].value_counts())

# --------------------------------------------
# 7️⃣  CUSTOMER PREDICTION PAGE
# --------------------------------------------
if page == "Customer Prediction":
    st.header("📋 Customer Prediction Interface")

    st.markdown("""
    Use the fields below to enter customer details.  
    ℹ️ **Note:** The model predicts whether the customer is likely *Fake (1)* or *Genuine (0)*.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        account_age_days = st.number_input("Account Age (days)", min_value=0, max_value=5000, value=100)
        phone_verified_label = st.selectbox("Phone Verified?", ["Yes", "No"])
        phone_verified = 1 if phone_verified_label == "Yes" else 0
        email_domain_type = st.selectbox("Email Domain Type", list(encoders["email_domain_type"].classes_))

    with col2:
        address_similarity_score = st.slider("Address Similarity Score", 0.0, 1.0, 0.5,
                                             help="How similar is the delivery address to the billing address?")
        total_orders = st.number_input("Total Orders", min_value=0, max_value=1000, value=20)
        avg_order_value = st.number_input("Average Order Value (₹)", min_value=0.0, max_value=10000.0, value=500.0)

    with col3:
        cancel_rate = st.slider("Cancel Rate", 0.0, 1.0, 0.2)
        order_frequency_per_month = st.number_input("Order Frequency (per month)", min_value=0, max_value=30, value=5, step=1)
        num_categories_purchased = st.number_input("Number of Categories Purchased", 1, 20, 5)

    col4, col5, col6 = st.columns(3)

    with col4:
        category_concentration_ratio = st.slider("Category Concentration Ratio", 0.0, 1.0, 0.5)
        num_payment_methods = st.slider("Number of Payment Methods", 1, 10, 2)

    with col5:
        payment_failure_rate = st.slider("Payment Failure Rate", 0.0, 1.0, 0.1)
        refund_rate = st.slider("Refund Rate", 0.0, 1.0, 0.2)

    with col6:
        same_card_label = st.selectbox("Same Card Used Across Multiple Accounts?", ["Yes", "No"])
        same_card_diff_accounts = 1 if same_card_label == "Yes" else 0
        replacement_rate = st.slider("Replacement Rate", 0.0, 1.0, 0.1)
        replacement_to_order_ratio = st.slider("Replacement-to-Order Ratio", 0.0, 1.0, 0.05)
        common_return_reason = st.selectbox("Most Common Return Reason", list(encoders["common_return_reason"].classes_))

    # ------------------------------
    # Build input DataFrame
    # ------------------------------
    input_data = pd.DataFrame([{
        "account_age_days": account_age_days,
        "email_domain_type": email_domain_type,
        "phone_verified": phone_verified,
        "address_similarity_score": address_similarity_score,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "cancel_rate": cancel_rate,
        "order_frequency_per_month": int(order_frequency_per_month),
        "num_categories_purchased": num_categories_purchased,
        "category_concentration_ratio": category_concentration_ratio,
        "num_payment_methods": num_payment_methods,
        "payment_failure_rate": payment_failure_rate,
        "refund_rate": refund_rate,
        "same_card_diff_accounts": same_card_diff_accounts,
        "replacement_rate": replacement_rate,
        "replacement_to_order_ratio": replacement_to_order_ratio,
        "common_return_reason": common_return_reason
    }])

    # Label encode categorical columns
    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[col])

    # ------------------------------
    # Make prediction
    # ------------------------------
    if st.button("🔍 Predict"):
        try:
            # Ensure all columns match model expectations
            missing_cols = set(model.feature_names_in_) - set(input_data.columns)
            for col in missing_cols:
                input_data[col] = 0  # Default placeholder for missing numeric columns

            # Reorder columns
            input_data = input_data[model.feature_names_in_]

            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][prediction]
            logger.info("Prediction made with probability %.4f", float(prob))

            if prediction == 1:
                st.error(f"🚨 Predicted: **FAKE CUSTOMER (1)** with {prob*100:.2f}% confidence.")
            else:
                st.success(f"✅ Predicted: **GENUINE CUSTOMER (0)** with {prob*100:.2f}% confidence.")

            st.markdown("""
            **Legend:**
            - `1` → 🚨 Fake Customer  
            - `0` → ✅ Genuine Customer
            """)

            with st.expander("🔎 Model Input Data"):
                st.dataframe(input_data)
        except Exception as e:
            logger.exception("Prediction failed")
            st.error("Prediction failed. See logs for details.")
            raise CustomException("Prediction failed", e)

    # Simplified UI: only Predict and input details are kept
