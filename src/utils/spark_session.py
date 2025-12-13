"""
Spark session initialization and configuration utilities.

Author: Silpa Das
Date: 2024-11-15
"""

import os
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import logging

logger = logging.getLogger(__name__)


def get_spark_session(app_name="FraudDetection", config_overrides=None):
    """
    Initialize or retrieve existing Spark session with optimized configs.

    Args:
        app_name (str): Name of the Spark application
        config_overrides (dict): Optional config key-value pairs to override defaults

    Returns:
        SparkSession: Configured Spark session
    """
    # Default configurations - tuned for local development
    # TODO: Move these to config file for production
    default_config = {
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.sql.shuffle.partitions": "8",  # Reduced for local mode
        "spark.default.parallelism": "8",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.driver.maxResultSize": "2g",
        # Enable Arrow for faster pandas conversion
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        # Parquet optimization
        "spark.sql.parquet.compression.codec": "snappy",
        "spark.sql.parquet.mergeSchema": "false",
    }

    # Merge with user overrides if provided
    if config_overrides:
        default_config.update(config_overrides)

    # Build Spark config
    conf = SparkConf()
    for key, value in default_config.items():
        conf.set(key, value)

    # Create session
    try:
        spark = (SparkSession.builder
                .appName(app_name)
                .config(conf=conf)
                .enableHiveSupport()  # Enable Hive support for table operations
                .getOrCreate())

        logger.info(f"Spark session created successfully: {app_name}")
        logger.info(f"Spark version: {spark.version}")

        # Set log level to reduce verbosity
        spark.sparkContext.setLogLevel("WARN")

        return spark

    except Exception as e:
        logger.error(f"Failed to create Spark session: {str(e)}")
        raise


def stop_spark_session(spark):
    """Gracefully stop Spark session and cleanup resources."""
    if spark:
        try:
            spark.stop()
            logger.info("Spark session stopped successfully")
        except Exception as e:
            logger.warning(f"Error stopping Spark session: {str(e)}")


# Context manager for automatic cleanup
class SparkSessionManager:
    """Context manager for Spark sessions - ensures cleanup even on errors."""

    def __init__(self, app_name="FraudDetection", config_overrides=None):
        self.app_name = app_name
        self.config_overrides = config_overrides
        self.spark = None

    def __enter__(self):
        self.spark = get_spark_session(self.app_name, self.config_overrides)
        return self.spark

    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_spark_session(self.spark)
        # Don't suppress exceptions
        return False


if __name__ == "__main__":
    # Quick test
    with SparkSessionManager("TestApp") as spark:
        df = spark.range(10)
        print(f"Row count: {df.count()}")
        print("Spark session test passed!")
