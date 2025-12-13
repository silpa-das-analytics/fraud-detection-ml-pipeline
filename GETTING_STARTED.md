# Getting Started with Fraud Detection ML Pipeline

## Quick Start Guide

### 1. Prerequisites
```bash
# Install Python 3.9+
python --version

# Install Java for PySpark
java -version
```

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/silpa-das-analytics/fraud-detection-ml-pipeline.git
cd fraud-detection-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Sample Data

**Option A: Using Kaggle API** (Recommended)
```bash
# Setup Kaggle credentials first: https://www.kaggle.com/docs/api
# Download dataset
python -c "from src.ingestion.batch_ingest import download_sample_data; download_sample_data()"
```

**Option B: Manual Download**
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place in `data/raw/creditcard.csv`

### 4. Run the Pipeline

```bash
# Step 1: Ingest data (Bronze layer)
python src/ingestion/batch_ingest.py

# Step 2: Transform to Silver layer
python src/etl/bronze_to_silver.py

# Step 3: Feature engineering (Gold layer)
python src/etl/feature_engineering.py

# Step 4: Train model
python src/models/train.py

# Step 5: Evaluate
python src/models/evaluate.py
```

### 5. Using Docker (Alternative)

```bash
# Build image
docker build -t fraud-detection .

# Run container
docker run -p 8000:8000 fraud-detection
```

## Project Structure Overview

```
fraud-detection-ml-pipeline/
├── src/
│   ├── ingestion/       # Data loading
│   ├── etl/            # Transformations
│   ├── models/         # ML training
│   ├── utils/          # Shared utilities
│   └── validation/     # Quality checks
├── data/
│   ├── raw/            # Bronze layer
│   ├── processed/      # Silver layer
│   └── curated/        # Gold layer
├── config/             # Configuration files
└── artifacts/          # Model outputs
```

## Key Features

- **Medallion Architecture**: Bronze → Silver → Gold layers
- **PySpark ETL**: Scalable data processing
- **ML Training**: XGBoost with SMOTE
- **Data Quality**: Automated validation checks
- **CI/CD**: GitHub Actions pipeline
- **Docker**: Containerized deployment

## Next Steps

1. Explore Jupyter notebooks in `notebooks/`
2. Customize configs in `config/config.yaml`
3. Add your own features in `src/etl/feature_engineering.py`
4. Tune model params in `config/config.yaml`

## Troubleshooting

**Issue: Spark errors**
- Ensure Java 8+ is installed
- Check `JAVA_HOME` environment variable

**Issue: Memory errors**
- Reduce `spark.driver.memory` in config
- Process smaller data batches

**Issue: Kaggle download fails**
- Setup API credentials: `~/.kaggle/kaggle.json`
- Or download manually from Kaggle website

## Contributing

Feel free to open issues or submit PRs!

## License

MIT License - see LICENSE file
