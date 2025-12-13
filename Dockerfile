FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set Java home for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p logs artifacts/models artifacts/reports

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "src/api/main.py"]
