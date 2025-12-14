"""
Generate synthetic fraud detection dataset for testing.
Creates a small dataset that mimics credit card transactions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_sample_data(n_samples=1000, fraud_rate=0.05):
    """
    Generate synthetic transaction data.

    Args:
        n_samples: Number of transactions to generate
        fraud_rate: Proportion of fraudulent transactions

    Returns:
        DataFrame with synthetic transaction data
    """
    print(f"Generating {n_samples} sample transactions...")

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Generate timestamps (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    timestamps = [
        start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        for _ in range(n_samples)
    ]

    # Generate transaction IDs
    transaction_ids = [f"TXN{str(i).zfill(8)}" for i in range(n_samples)]

    # Generate user IDs (500 unique users)
    user_ids = [f"USER{random.randint(1, 500):04d}" for _ in range(n_samples)]

    # Generate merchants
    merchants = [
        "Amazon", "Walmart", "Target", "Starbucks", "Shell",
        "McDonald's", "Apple Store", "Best Buy", "Costco", "Home Depot"
    ]
    merchant_ids = [random.choice(merchants) for _ in range(n_samples)]

    # Merchant categories
    categories = ["retail", "food", "gas", "electronics", "grocery", "entertainment"]
    merchant_categories = [random.choice(categories) for _ in range(n_samples)]

    # Generate amounts (legitimate: $5-500, fraud: $100-2000)
    amounts = []
    is_fraud = []

    for i in range(n_samples):
        if i < n_fraud:
            # Fraudulent transaction - higher amounts
            amount = np.random.exponential(scale=300) + 100
            fraud = 1
        else:
            # Legitimate transaction - lower amounts
            amount = np.random.exponential(scale=50) + 5
            fraud = 0

        amounts.append(round(amount, 2))
        is_fraud.append(fraud)

    # Shuffle to mix fraud and legit
    indices = list(range(n_samples))
    random.shuffle(indices)

    # Generate locations (lat/lon)
    # US coordinates roughly
    locations_lat = [round(random.uniform(25.0, 48.0), 6) for _ in range(n_samples)]
    locations_lon = [round(random.uniform(-125.0, -65.0), 6) for _ in range(n_samples)]

    # Generate card hashes
    card_hashes = [f"CARD{random.randint(10000, 99999)}" for _ in range(n_samples)]

    # Device IDs
    device_ids = [f"DEV{random.randint(1000, 9999)}" for _ in range(n_samples)]

    # IP addresses (simplified)
    ip_addresses = [f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
                    for _ in range(n_samples)]

    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': [transaction_ids[i] for i in indices],
        'timestamp': [timestamps[i] for i in indices],
        'amount': [amounts[i] for i in indices],
        'merchant_id': [merchant_ids[i] for i in indices],
        'merchant_category': [merchant_categories[i] for i in indices],
        'user_id': [user_ids[i] for i in indices],
        'card_number_hash': [card_hashes[i] for i in indices],
        'location_lat': [locations_lat[i] for i in indices],
        'location_lon': [locations_lon[i] for i in indices],
        'device_id': [device_ids[i] for i in indices],
        'ip_address': [ip_addresses[i] for i in indices],
        'is_fraud': [is_fraud[i] for i in indices]
    })

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    fraud_count = df['is_fraud'].sum()
    print(f"Generated {len(df)} transactions:")
    print(f"  - Legitimate: {len(df) - fraud_count}")
    print(f"  - Fraudulent: {fraud_count}")
    print(f"  - Fraud rate: {fraud_count/len(df)*100:.2f}%")

    return df


if __name__ == "__main__":
    import os

    # Create data directory if not exists
    os.makedirs("data/raw", exist_ok=True)

    # Generate data
    df = generate_sample_data(n_samples=5000, fraud_rate=0.05)

    # Save to CSV
    output_file = "data/raw/sample_transactions.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✓ Sample data saved to: {output_file}")
    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nData summary:")
    print(df.describe())
