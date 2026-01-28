# Learning-Analytics: MultiversX Blockchain Data Analysis & Price Prediction

A comprehensive data engineering and machine learning pipeline that downloads MultiversX blockchain data from Google BigQuery, enriches it with market data (EGLD price), engineers features using PySpark, and trains multiple regression models to predict network transaction volume trends.

## 🚀 Key Features

* **BigQuery Integration**: Automated extraction of blockchain metrics (blocks, transactions, fees, gas) from the official `bigquery-public-data.crypto_multiversx_mainnet_eu` dataset.
* **Market Data Enrichment**: Merges on-chain data with daily EGLD-USD price and volume data via Yahoo Finance.
* **Spark Feature Engineering**: Scalable processing using PySpark to generate:
    * Cyclical date features (sine/cosine transforms for day-of-week, month, etc.)
    * Rolling window statistics (7/14/28-day moving averages, volatility, trends)
    * Lag features for time-series forecasting.
* **Multi-Model Training**: Trains and compares four distinct models:
    * Linear Regression (ElasticNet)
    * Decision Tree Regressor
    * Random Forest Regressor
    * XGBoost (via `xgboost.spark`)
* **Automated Evaluation**: Generates performance metrics (RMSE, R²) and visualization plots (time-series, residuals, scatter plots).

## 📋 Prerequisites

* **Python 3.8+**
* **Java 8 or 11** (Required for PySpark)
* **Google Cloud Platform Account**:
    * A GCP project with BigQuery API enabled.
    * Google Cloud SDK installed and authenticated.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/ramonaioana108/Learning-Analytics-MultiversX.git](https://github.com/ramonaioana108/Learning-Analytics-MultiversX.git)
    cd Learning-Analytics-MultiversX
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install pandas yfinance google-cloud-bigquery pyspark pyarrow xgboost matplotlib
    ```

3.  **Set up Google Credentials**:
    Ensure your environment is authenticated to query BigQuery.
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
    # Or if using gcloud sdk:
    # gcloud auth application-default login
    ```

## 📦 Pipeline Usage

You can run the entire pipeline using the master CLI tool `multiversx.py`, or execute individual stages manually.

### Option A: All-in-One Command
Run the full pipeline (Download → Merge → Process → Train) in one go:

```bash
python multiversx.py pipeline \
  --start 2024-01-01 --end 2024-12-31 \
  --mode daily \
  --ticker EGLD-USD \
  --enable_xgboost
```

```bash
python3 plots.py --run_dir ./models/latest_model
```