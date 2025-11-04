import numpy as np
import pandas as pd
import random
from configs.config import DATA_PATH, NUM_SAMPLES, FAKE_PERCENTAGE, CASES_PER_CUSTOMER

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

    # Ensure at least CASES_PER_CUSTOMER rows per unique customer
    num_customers = max(1, NUM_SAMPLES // CASES_PER_CUSTOMER)
    customer_ids = np.repeat(np.arange(1, num_customers + 1), CASES_PER_CUSTOMER)
    total_rows = len(customer_ids)

    # Per-customer baselines + per-case noise
    base_account_age = np.random.randint(0, 2000, num_customers)
    base_total_orders = np.random.poisson(20, num_customers)
    base_avg_order_value = np.random.normal(600, 250, num_customers).clip(50, 5000)
    base_order_freq = np.random.uniform(0, 15, num_customers)
    base_num_payments = np.random.randint(1, 6, num_customers)
    base_num_categories = np.random.randint(1, 10, num_customers)
    base_domain = [random_email_domain() for _ in range(num_customers)]
    base_phone_verified = np.random.choice([0, 1], num_customers, p=[0.2, 0.8])

    # Expand to per-case values with small noise
    def expand_with_noise(base, low=0.9, high=1.1, clip_min=None, clip_max=None):
        vals = np.repeat(base, CASES_PER_CUSTOMER) * np.random.uniform(low, high, total_rows)
        if clip_min is not None or clip_max is not None:
            vals = np.clip(vals, clip_min if clip_min is not None else vals.min(), clip_max if clip_max is not None else vals.max())
        return vals

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "account_age_days": expand_with_noise(base_account_age, 0.8, 1.2).round().astype(int),
        "email_domain_type": np.repeat(base_domain, CASES_PER_CUSTOMER),
        "phone_verified": np.repeat(base_phone_verified, CASES_PER_CUSTOMER),
        "address_similarity_score": np.random.uniform(0, 1, total_rows),
        "total_orders": expand_with_noise(base_total_orders, 0.7, 1.3, 0, None).round().astype(int),
        "avg_order_value": expand_with_noise(base_avg_order_value, 0.7, 1.3, 50, 5000),
        "cancel_rate": np.random.uniform(0, 0.6, total_rows),
        "order_frequency_per_month": expand_with_noise(base_order_freq, 0.7, 1.3, 0, 30),
        "num_categories_purchased": expand_with_noise(base_num_categories, 0.7, 1.3, 1, 20).round().astype(int),
        "category_concentration_ratio": np.random.uniform(0.1, 1.0, total_rows),
        "num_payment_methods": expand_with_noise(base_num_payments, 0.7, 1.3, 1, 10).round().astype(int),
        "payment_failure_rate": np.random.uniform(0, 0.6, total_rows),
        "refund_rate": np.random.uniform(0, 0.5, total_rows),
        "same_card_diff_accounts": np.random.choice([0, 1], total_rows, p=[0.95, 0.05]),
        "replacement_rate": np.random.uniform(0, 0.4, total_rows),
        "replacement_to_order_ratio": np.random.uniform(0, 0.4, total_rows),
        "common_return_reason": [random_return_reason() for _ in range(total_rows)]
    })

    def label_fake_customer(row):
        conditions = [
            (row["account_age_days"] < 30 and row["phone_verified"] == 0),
            (row["cancel_rate"] > 0.4),
            (row["payment_failure_rate"] > 0.3),
            (row["replacement_rate"] > 0.2),
            (row["email_domain_type"] == "tempmail"),
            (row["same_card_diff_accounts"] == 1),
            # Low similarity between billing and delivery addresses can indicate fraud
            (row["address_similarity_score"] < 0.15)
        ]
        return 1 if any(conditions) else 0

    df["is_fake"] = df.apply(label_fake_customer, axis=1)

    # Balance classes to 50/50
    total = len(df)
    target_fake = total // 2
    current_fake = int(df["is_fake"].sum())
    if current_fake < target_fake:
        to_flip = target_fake - current_fake
        idx = df[df["is_fake"] == 0].sample(to_flip, random_state=42).index
        df.loc[idx, "is_fake"] = 1
    elif current_fake > target_fake:
        to_flip = current_fake - target_fake
        idx = df[df["is_fake"] == 1].sample(to_flip, random_state=42).index
        df.loc[idx, "is_fake"] = 0

    df.to_csv(file_path, index=False)
    print(f"[DATA] Synthetic dataset generated at: {file_path}")
