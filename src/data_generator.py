import numpy as np
import pandas as pd
import random
from configs.config import DATA_PATH, NUM_SAMPLES

def generate_synthetic_data(file_path=DATA_PATH):
    np.random.seed(42)
    random.seed(42)

    def random_email_domain():
        return random.choices(
            ["gmail", "yahoo", "outlook", "tempmail", "protonmail", "companymail"],
            weights=[0.35, 0.25, 0.15, 0.15, 0.05, 0.05],
            k=1
        )[0]

    def random_return_reason():
        return random.choice(["Defective", "Wrong Item", "Changed Mind", "Other"])

    # Generate independent rows; no identifier column
    total_rows = NUM_SAMPLES

    df = pd.DataFrame({
        "account_age_days": np.random.randint(0, 2000, total_rows),
        "email_domain_type": [random_email_domain() for _ in range(total_rows)],
        "phone_verified": np.random.choice([0, 1], total_rows, p=[0.2, 0.8]),
        "address_similarity_score": np.random.uniform(0, 1, total_rows),
        "total_orders": np.random.poisson(20, total_rows).clip(0).astype(int),
        "avg_order_value": np.random.normal(600, 250, total_rows).clip(50, 5000),
        "cancel_rate": np.random.uniform(0, 0.6, total_rows),
        "order_frequency_per_month": np.random.uniform(0, 30, total_rows),
        "num_categories_purchased": np.random.randint(1, 21, total_rows),
        "category_concentration_ratio": np.random.uniform(0.1, 1.0, total_rows),
        "num_payment_methods": np.random.randint(1, 11, total_rows),
        "payment_failure_rate": np.random.uniform(0, 0.6, total_rows),
        "refund_rate": np.random.uniform(0, 0.5, total_rows),
        "same_card_diff_accounts": np.random.choice([0, 1], total_rows, p=[0.95, 0.05]),
        "replacement_rate": np.random.uniform(0, 0.4, total_rows),
        "replacement_to_order_ratio": np.random.uniform(0, 0.4, total_rows),
        "common_return_reason": [random_return_reason() for _ in range(total_rows)]
    })

    def label_fake_customer(row):
        # Stronger, rarer signals; require multiple to mark as fraud
        signals = [
            int(row["account_age_days"] < 50 and row["phone_verified"] == 0),
            int(row["cancel_rate"] > 0.5),
            int(row["payment_failure_rate"] > 0.45),
            int(row["replacement_rate"] > 0.3),
            int(row["email_domain_type"] in ("tempmail",)),
            int(row["same_card_diff_accounts"] == 1),
            int(row["address_similarity_score"] < 0.10),
            int(row["refund_rate"] > 0.40),
        ]
        return 1 if sum(signals) >= 2 else 0

    df["is_fake"] = df.apply(label_fake_customer, axis=1)

    # Note: No forced 50/50 class balancing to avoid injecting label noise.

    df.to_csv(file_path, index=False)
    print(f"[DATA] Synthetic dataset generated at: {file_path}")
