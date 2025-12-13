"""
Feature engineering module for fraud detection.

Creates advanced features from transaction data:
- Aggregation features (user behavior patterns)
- Velocity features (transaction frequency)
- Time-based features
- Geospatial features
"""

import sys
from pathlib import Path
from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType

sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Creates ML-ready features from cleansed transaction data."""

    @staticmethod
    def create_user_aggregation_features(df):
        """
        Calculate user-level aggregation features.

        Features:
        - Average transaction amount (7d, 30d windows)
        - Transaction count (24h, 7d windows)
        - Distinct merchants visited
        - Standard deviation of amounts
        """
        logger.info("Creating user aggregation features...")

        # Define time windows (in seconds)
        window_24h = 24 * 3600
        window_7d = 7 * 24 * 3600
        window_30d = 30 * 24 * 3600

        # Convert timestamp to unix timestamp for window calculations
        df = df.withColumn("unix_timestamp", F.unix_timestamp("timestamp"))

        # Window specs
        window_24h_spec = Window.partitionBy("user_id").orderBy("unix_timestamp") \
                                .rangeBetween(-window_24h, 0)

        window_7d_spec = Window.partitionBy("user_id").orderBy("unix_timestamp") \
                               .rangeBetween(-window_7d, 0)

        window_30d_spec = Window.partitionBy("user_id").orderBy("unix_timestamp") \
                                .rangeBetween(-window_30d, 0)

        # Aggregation features
        # 24-hour window
        df = df.withColumn("user_tx_count_24h", F.count("*").over(window_24h_spec))
        df = df.withColumn("user_avg_amount_24h", F.avg("amount").over(window_24h_spec))

        # 7-day window
        df = df.withColumn("user_tx_count_7d", F.count("*").over(window_7d_spec))
        df = df.withColumn("user_avg_amount_7d", F.avg("amount").over(window_7d_spec))
        df = df.withColumn("user_std_amount_7d", F.stddev("amount").over(window_7d_spec))
        df = df.withColumn("user_max_amount_7d", F.max("amount").over(window_7d_spec))

        # 30-day window
        df = df.withColumn("user_tx_count_30d", F.count("*").over(window_30d_spec))

        # Replace nulls in std deviation (happens when only 1 transaction in window)
        df = df.withColumn("user_std_amount_7d",
                          F.when(F.col("user_std_amount_7d").isNull(), 0)
                          .otherwise(F.col("user_std_amount_7d")))

        logger.info("User aggregation features created")
        return df

    @staticmethod
    def create_velocity_features(df):
        """
        Transaction velocity features - detect rapid successive transactions.

        Features:
        - Time since last transaction (same user)
        - Time since last transaction (same card)
        - Transaction frequency score
        """
        logger.info("Creating velocity features...")

        # Window to get previous transaction timestamp
        user_window = Window.partitionBy("user_id").orderBy("unix_timestamp")

        df = df.withColumn("prev_tx_timestamp",
                          F.lag("unix_timestamp").over(user_window))

        df = df.withColumn("time_since_last_tx",
                          F.when(F.col("prev_tx_timestamp").isNotNull(),
                                F.col("unix_timestamp") - F.col("prev_tx_timestamp"))
                          .otherwise(999999))  # Large value for first transaction

        # Velocity score: inversely proportional to time since last tx
        # High score = suspicious rapid transactions
        df = df.withColumn("velocity_score",
                          F.when(F.col("time_since_last_tx") < 60, 10.0)  # < 1 min: very suspicious
                          .when(F.col("time_since_last_tx") < 300, 5.0)  # < 5 min
                          .when(F.col("time_since_last_tx") < 3600, 2.0)  # < 1 hour
                          .otherwise(0.0))

        logger.info("Velocity features created")
        return df

    @staticmethod
    def create_deviation_features(df):
        """
        Deviation from user's normal behavior.

        Features:
        - Amount deviation from user average
        - Unusual merchant for this user
        - Unusual time of day for this user
        """
        logger.info("Creating deviation features...")

        # Amount deviation ratio
        df = df.withColumn("amount_deviation_ratio",
                          F.when(F.col("user_avg_amount_7d") > 0,
                                F.col("amount") / F.col("user_avg_amount_7d"))
                          .otherwise(1.0))

        # Flag if amount is significantly higher than usual
        df = df.withColumn("is_amount_anomaly",
                          F.when(F.col("amount_deviation_ratio") > 3.0, 1).otherwise(0))

        # Hour deviation (simplified - flag night transactions as potentially risky)
        df = df.withColumn("is_night_tx",
                          F.when(F.col("transaction_hour").isin([0, 1, 2, 3, 4, 5]), 1).otherwise(0))

        logger.info("Deviation features created")
        return df

    @staticmethod
    def create_merchant_features(df):
        """
        Merchant-related features.

        Features:
        - Merchant risk score (based on fraud rate)
        - Category risk score
        """
        logger.info("Creating merchant features...")

        # Calculate fraud rate per merchant (using training data)
        # Note: In production, this would come from a pre-computed lookup table
        merchant_fraud_rate = df.groupBy("merchant_id").agg(
            (F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)) /
             F.count("*")).alias("merchant_fraud_rate_temp")
        )

        df = df.join(merchant_fraud_rate, on="merchant_id", how="left")
        df = df.withColumn("merchant_fraud_rate",
                          F.coalesce(F.col("merchant_fraud_rate_temp"), F.lit(0.0)))
        df = df.drop("merchant_fraud_rate_temp")

        # Similar for category
        category_fraud_rate = df.groupBy("merchant_category").agg(
            (F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)) /
             F.count("*")).alias("category_fraud_rate_temp")
        )

        df = df.join(category_fraud_rate, on="merchant_category", how="left")
        df = df.withColumn("category_fraud_rate",
                          F.coalesce(F.col("category_fraud_rate_temp"), F.lit(0.0)))
        df = df.drop("category_fraud_rate_temp")

        logger.info("Merchant features created")
        return df

    def create_all_features(self, df):
        """Execute all feature engineering steps."""
        logger.info("Starting comprehensive feature engineering...")

        # Add unix timestamp if not present
        if "unix_timestamp" not in df.columns:
            df = df.withColumn("unix_timestamp", F.unix_timestamp("timestamp"))

        # Apply all feature engineering functions
        df = self.create_user_aggregation_features(df)
        df = self.create_velocity_features(df)
        df = self.create_deviation_features(df)
        df = self.create_merchant_features(df)

        # Count total features
        feature_cols = [col for col in df.columns
                       if col not in ["transaction_id", "timestamp", "user_id", "is_fraud"]]

        logger.info(f"Feature engineering complete. Created {len(feature_cols)} features")

        return df


if __name__ == "__main__":
    # Quick test
    from utils import SparkSessionManager

    with SparkSessionManager("FeatureEngineering_Test") as spark:
        # Create dummy data for testing
        test_data = [
            ("tx1", "user1", 100.0, "2024-01-01 10:00:00", "m1", "retail", 0),
            ("tx2", "user1", 150.0, "2024-01-01 10:30:00", "m2", "food", 0),
            ("tx3", "user1", 500.0, "2024-01-01 10:35:00", "m3", "retail", 1),
        ]

        test_df = spark.createDataFrame(
            test_data,
            ["transaction_id", "user_id", "amount", "timestamp", "merchant_id", "merchant_category", "is_fraud"]
        )
        test_df = test_df.withColumn("timestamp", F.to_timestamp("timestamp"))

        fe = FeatureEngineer()
        result_df = fe.create_all_features(test_df)

        print("\nGenerated features:")
        result_df.select("transaction_id", "user_tx_count_24h", "velocity_score",
                        "amount_deviation_ratio").show()
