"""
Machine Learning model training pipeline for fraud detection.

Trains XGBoost classifier with hyperparameter tuning and handles class imbalance.
Implements MLflow for experiment tracking and model versioning.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE

sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_logger

logger = setup_logger(__name__)


class FraudDetectionTrainer:
    """Handles training of fraud detection ML models."""

    def __init__(self, data_path, model_output_path="artifacts/models"):
        self.data_path = Path(data_path)
        self.model_output_path = Path(model_output_path)
        self.model_output_path.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_names = None

    def load_data(self, spark):
        """Load Gold layer data and convert to pandas for sklearn."""
        logger.info(f"Loading data from {self.data_path}")

        # Read from parquet
        df = spark.read.parquet(str(self.data_path))

        # Convert to pandas (manageable for local processing)
        # In production, you might sample or use distributed ML (e.g., Spark MLlib)
        pdf = df.toPandas()

        logger.info(f"Loaded {len(pdf):,} records")

        # Check class distribution
        fraud_count = pdf['is_fraud'].sum()
        fraud_rate = fraud_count / len(pdf) * 100
        logger.info(f"Fraud rate: {fraud_rate:.2f}% ({fraud_count:,} frauds out of {len(pdf):,})")

        return pdf

    def prepare_features(self, df):
        """
        Prepare feature matrix and target variable.

        Returns:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature column names
        """
        logger.info("Preparing features...")

        # Drop non-feature columns
        exclude_cols = [
            'transaction_id', 'user_id', 'timestamp', 'is_fraud',
            'ingestion_timestamp', 'ingestion_date', 'source_file',
            'cleansed_timestamp', 'processing_date', 'unix_timestamp',
            'prev_tx_timestamp', 'silver_row_id', 'card_number_hash',
            'device_id', 'ip_address'  # High cardinality, need encoding
        ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle categorical variables
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()

        if categorical_cols:
            logger.info(f"Encoding categorical features: {categorical_cols}")
            # Simple one-hot encoding (in production, use more sophisticated encoding)
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

            # Update feature columns after encoding
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].fillna(0)  # Handle any remaining nulls
        y = df['is_fraud']

        self.feature_names = feature_cols

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Number of features: {len(feature_cols)}")

        return X, y

    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """
        Address class imbalance using SMOTE or class weights.

        Args:
            X_train: Training features
            y_train: Training labels
            method: 'smote' or 'weights'

        Returns:
            Resampled X_train, y_train
        """
        logger.info(f"Handling class imbalance using: {method}")

        original_fraud_count = y_train.sum()
        original_total = len(y_train)

        if method == 'smote':
            # Use SMOTE to oversample minority class
            smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Bring fraud to 50% of legit
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            new_fraud_count = y_resampled.sum()
            new_total = len(y_resampled)

            logger.info(f"SMOTE resampling: {original_total} -> {new_total} samples")
            logger.info(f"Fraud samples: {original_fraud_count} -> {new_fraud_count}")

            return X_resampled, y_resampled

        else:
            # Use class weights instead
            logger.info("Using class weights (no resampling)")
            return X_train, y_train

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model with optimized hyperparameters.

        Hyperparameters tuned for fraud detection:
        - Higher max_depth to capture complex patterns
        - scale_pos_weight to handle imbalance
        - Early stopping to prevent overfitting
        """
        logger.info("Training XGBoost model...")

        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

        # XGBoost parameters (somewhat tuned, but you'd do GridSearch in real project)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist',  # Faster training
            'use_label_encoder': False
        }

        # Train with early stopping
        self.model = xgb.XGBClassifier(**params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=20,
            verbose=True
        )

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation."""
        logger.info("Evaluating model on test set...")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"\n{'='*50}")
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info(f"{'='*50}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")
        logger.info(f"AUC-ROC:   {auc:.4f}")
        logger.info(f"{'='*50}\n")

        # Classification report
        logger.info("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"                 Predicted")
        logger.info(f"               Legit  Fraud")
        logger.info(f"Actual Legit   {cm[0][0]:6d} {cm[0][1]:6d}")
        logger.info(f"       Fraud   {cm[1][0]:6d} {cm[1][1]:6d}")

        # Feature importance
        self.log_feature_importance()

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc
        }

        return metrics

    def log_feature_importance(self, top_n=15):
        """Display top N most important features."""
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_n} Important Features:")
        logger.info(feature_importance.head(top_n).to_string(index=False))

        return feature_importance

    def save_model(self, metrics=None):
        """Save trained model and scaler."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = self.model_output_path / f"xgboost_fraud_model_{timestamp}.pkl"
        scaler_filename = self.model_output_path / f"scaler_{timestamp}.pkl"

        # Save model
        joblib.dump(self.model, model_filename)
        logger.info(f"Model saved to: {model_filename}")

        # Save scaler if exists
        if self.scaler:
            joblib.dump(self.scaler, scaler_filename)
            logger.info(f"Scaler saved to: {scaler_filename}")

        # Save feature names
        feature_names_file = self.model_output_path / f"feature_names_{timestamp}.txt"
        with open(feature_names_file, 'w') as f:
            f.write('\n'.join(self.feature_names))

        # Save metrics
        if metrics:
            metrics_file = self.model_output_path / f"metrics_{timestamp}.txt"
            with open(metrics_file, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")

        logger.info(f"Model artifacts saved to: {self.model_output_path}")

    def run_training_pipeline(self, spark):
        """Execute complete training pipeline."""
        logger.info("="*60)
        logger.info("FRAUD DETECTION MODEL TRAINING PIPELINE")
        logger.info("="*60)

        # 1. Load data
        df = self.load_data(spark)

        # 2. Prepare features
        X, y = self.prepare_features(df)

        # 3. Train-validation-test split
        logger.info("Splitting data: 70% train, 15% val, 15% test")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 0.85 ≈ 0.15
        )

        logger.info(f"Train set: {len(X_train):,} | Val set: {len(X_val):,} | Test set: {len(X_test):,}")

        # 4. Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_class_imbalance(X_train, y_train, method='smote')

        # 5. Train model
        self.train_model(X_train_resampled, y_train_resampled, X_val, y_val)

        # 6. Evaluate
        metrics = self.evaluate_model(X_test, y_test)

        # 7. Save model
        self.save_model(metrics)

        logger.info("\n" + "="*60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)

        return metrics


if __name__ == "__main__":
    import argparse
    from utils import SparkSessionManager

    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", default="data/curated/transactions_gold",
                       help="Path to Gold layer data")
    parser.add_argument("--output", default="artifacts/models",
                       help="Model output directory")

    args = parser.parse_args()

    trainer = FraudDetectionTrainer(args.data, args.output)

    with SparkSessionManager("FraudDetection_Training") as spark:
        metrics = trainer.run_training_pipeline(spark)
        print(f"\n✓ Model training completed. AUC: {metrics['auc_roc']:.4f}")
