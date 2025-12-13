"""
Batch data ingestion module for fraud detection pipeline.

Loads transaction data from various sources (CSV, JSON, Parquet) into Bronze layer.
Handles incremental loads and checkpointing.

Author: Silpa Das
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logger, SparkSessionManager

logger = setup_logger(__name__)


# Define schema for credit card transactions
TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("amount", DoubleType(), False),
    StructField("merchant_id", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("user_id", StringType(), False),
    StructField("card_number_hash", StringType(), True),  # Hashed for privacy
    StructField("location_lat", DoubleType(), True),
    StructField("location_lon", DoubleType(), True),
    StructField("device_id", StringType(), True),
    StructField("ip_address", StringType(), True),
    StructField("is_fraud", IntegerType(), True),  # Label (0=legit, 1=fraud)
])


class BatchDataIngestion:
    """Handles batch ingestion of transaction data into Bronze layer."""

    def __init__(self, source_path, target_path, file_format="csv"):
        """
        Initialize ingestion job.

        Args:
            source_path: Path to source data files
            target_path: Path to Bronze layer storage
            file_format: Format of source files (csv, json, parquet)
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.file_format = file_format.lower()

        logger.info(f"Initialized batch ingestion: {self.source_path} -> {self.target_path}")

    def ingest(self, spark, partition_by="ingestion_date"):
        """
        Execute batch ingestion with schema validation.

        Args:
            spark: Active Spark session
            partition_by: Column to partition output data

        Returns:
            Row count of ingested data
        """
        logger.info(f"Starting batch ingestion from {self.source_path}")

        try:
            # Read based on file format
            if self.file_format == "csv":
                df = spark.read.csv(
                    str(self.source_path),
                    header=True,
                    schema=TRANSACTION_SCHEMA,
                    mode="PERMISSIVE"  # null for malformed records
                )
            elif self.file_format == "json":
                df = spark.read.json(
                    str(self.source_path),
                    schema=TRANSACTION_SCHEMA
                )
            elif self.file_format == "parquet":
                df = spark.read.parquet(str(self.source_path))
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")

            # Add metadata columns
            df = df.withColumn("ingestion_timestamp", F.current_timestamp())
            df = df.withColumn("ingestion_date", F.current_date())
            df = df.withColumn("source_file", F.input_file_name())

            # Basic data quality check - remove complete nulls
            initial_count = df.count()
            df = df.dropna(how="all")
            final_count = df.count()

            if initial_count != final_count:
                logger.warning(f"Dropped {initial_count - final_count} completely null rows")

            # Write to Bronze layer with partitioning
            logger.info(f"Writing {final_count} records to Bronze layer")
            df.write.mode("append").partitionBy(partition_by).parquet(str(self.target_path))

            logger.info(f"Ingestion completed successfully. Records: {final_count}")
            return final_count

        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
            raise


def download_sample_data():
    """
    Downloads sample fraud detection dataset from Kaggle.

    NOTE: Requires Kaggle API credentials (~/.kaggle/kaggle.json)
    Alternative: Manual download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    logger.info("Downloading sample dataset...")

    try:
        import kaggle

        # Download credit card fraud dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path='data/raw',
            unzip=True
        )
        logger.info("Sample data downloaded to data/raw/")

    except ImportError:
        logger.warning("Kaggle library not installed. Install with: pip install kaggle")
        logger.info("Or manually download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")

    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")
        logger.info("Please download manually from Kaggle")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Batch data ingestion")
    parser.add_argument("--source", default="data/raw/creditcard.csv", help="Source data path")
    parser.add_argument("--target", default="data/bronze/transactions", help="Target path")
    parser.add_argument("--format", default="csv", help="File format")
    parser.add_argument("--download", action="store_true", help="Download sample data")

    args = parser.parse_args()

    if args.download:
        download_sample_data()

    # Run ingestion
    ingestion = BatchDataIngestion(args.source, args.target, args.format)

    with SparkSessionManager("FraudDetection_Ingestion") as spark:
        record_count = ingestion.ingest(spark)
        print(f"\n✓ Ingested {record_count:,} records successfully")
