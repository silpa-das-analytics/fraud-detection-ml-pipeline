"""
Bronze to Silver ETL transformation.

Applies data quality checks, cleansing, and standardization to raw transaction data.
Implements validation rules and flags suspicious records.
"""

import sys
from pathlib import Path
from pyspark.sql import functions as F, Window
from pyspark.sql.types import IntegerType

sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_logger, SparkSessionManager

logger = setup_logger(__name__)


class BronzeToSilverTransformer:
    """Transforms raw Bronze data into cleansed Silver layer."""

    def __init__(self, bronze_path, silver_path):
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)

    def validate_and_cleanse(self, df):
        """
        Apply data quality rules and cleansing logic.

        Quality checks:
        - Remove duplicates (by transaction_id)
        - Validate amounts (must be positive)
        - Handle missing values (impute or flag)
        - Standardize timestamps
        - Detect anomalies (e.g., impossible locations)
        """
        logger.info("Starting data validation and cleansing...")

        initial_count = df.count()

        # 1. Remove exact duplicates
        df = df.dropDuplicates()

        # 2. Remove duplicates based on transaction_id (keep latest by ingestion_timestamp)
        window_spec = Window.partitionBy("transaction_id").orderBy(F.desc("ingestion_timestamp"))
        df = df.withColumn("row_num", F.row_number().over(window_spec))
        df = df.filter(F.col("row_num") == 1).drop("row_num")

        # 3. Validate amounts (must be positive)
        df = df.withColumn("is_amount_valid", F.when(F.col("amount") > 0, 1).otherwise(0))
        invalid_amounts = df.filter(F.col("is_amount_valid") == 0).count()

        if invalid_amounts > 0:
            logger.warning(f"Found {invalid_amounts} records with invalid amounts")
            # Flag but don't remove - might be useful for analysis
            df = df.withColumn("quality_flag",
                             F.when(F.col("is_amount_valid") == 0, "INVALID_AMOUNT").otherwise("OK"))
        else:
            df = df.withColumn("quality_flag", F.lit("OK"))

        # 4. Handle missing merchant info
        df = df.withColumn("merchant_id",
                          F.when(F.col("merchant_id").isNull(), "UNKNOWN").otherwise(F.col("merchant_id")))

        df = df.withColumn("merchant_category",
                          F.when(F.col("merchant_category").isNull(), "UNCATEGORIZED")
                          .otherwise(F.col("merchant_category")))

        # 5. Validate geographic coordinates (if present)
        # Latitude: -90 to 90, Longitude: -180 to 180
        df = df.withColumn("has_valid_location",
                          F.when(
                              (F.col("location_lat").isNotNull()) &
                              (F.col("location_lon").isNotNull()) &
                              (F.col("location_lat").between(-90, 90)) &
                              (F.col("location_lon").between(-180, 180)),
                              1
                          ).otherwise(0))

        # 6. Add cleansing metadata
        df = df.withColumn("cleansed_timestamp", F.current_timestamp())
        df = df.withColumn("processing_date", F.current_date())

        final_count = df.count()
        logger.info(f"Validation complete. Records: {initial_count} -> {final_count}")

        return df

    def enrich_data(self, df):
        """
        Add derived columns and enrichments.

        Enrichments:
        - Extract time features (hour, day_of_week, is_weekend)
        - Categorize amounts (small, medium, large)
        - Add sequence numbers for tracking
        """
        logger.info("Enriching data with derived features...")

        # Time-based features
        df = df.withColumn("transaction_hour", F.hour("timestamp"))
        df = df.withColumn("transaction_day_of_week", F.dayofweek("timestamp"))  # 1=Sunday, 7=Saturday
        df = df.withColumn("is_weekend",
                          F.when(F.col("transaction_day_of_week").isin([1, 7]), 1).otherwise(0))

        # Amount categorization (can be adjusted based on domain knowledge)
        df = df.withColumn("amount_category",
                          F.when(F.col("amount") < 50, "small")
                          .when(F.col("amount") < 200, "medium")
                          .otherwise("large"))

        # Add row ID for tracking through pipeline
        df = df.withColumn("silver_row_id", F.monotonically_increasing_id())

        logger.info("Enrichment complete")
        return df

    def run(self, spark):
        """Execute full Bronze -> Silver transformation."""
        logger.info(f"Starting Bronze -> Silver ETL: {self.bronze_path} -> {self.silver_path}")

        try:
            # Read from Bronze
            df = spark.read.parquet(str(self.bronze_path))
            logger.info(f"Read {df.count()} records from Bronze layer")

            # Apply transformations
            df = self.validate_and_cleanse(df)
            df = self.enrich_data(df)

            # Show quality summary
            quality_summary = df.groupBy("quality_flag").count().collect()
            logger.info("Quality summary:")
            for row in quality_summary:
                logger.info(f"  {row['quality_flag']}: {row['count']:,} records")

            # Write to Silver (partitioned by processing_date)
            logger.info(f"Writing to Silver layer: {self.silver_path}")
            df.write.mode("overwrite").partitionBy("processing_date").parquet(str(self.silver_path))

            logger.info("Bronze -> Silver ETL completed successfully")
            return df.count()

        except Exception as e:
            logger.error(f"ETL failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bronze to Silver ETL")
    parser.add_argument("--bronze", default="data/bronze/transactions", help="Bronze layer path")
    parser.add_argument("--silver", default="data/processed/transactions_silver", help="Silver layer path")

    args = parser.parse_args()

    transformer = BronzeToSilverTransformer(args.bronze, args.silver)

    with SparkSessionManager("FraudDetection_BronzeToSilver") as spark:
        record_count = transformer.run(spark)
        print(f"\n✓ Transformed {record_count:,} records to Silver layer")
