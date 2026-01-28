# train_spark_models.py
import json
from datetime import datetime
from pathlib import Path
from itertools import product

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, expm1, greatest,
    min as Fmin, max as Fmax, avg, stddev
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

PARQUET_PATH = "./data_parquet/multiversx_features_tuned.parquet"


def main():
    spark = (
        SparkSession.builder
        .appName("MultiversX-Train-Tuned")
        .master("local[8]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print(f"Reading: {PARQUET_PATH}")

    # Try/Except added to handle cases where the path doesn't exist yet
    try:
        df = spark.read.parquet(PARQUET_PATH)

        print("-" * 30)
        print("Schema:")
        df.printSchema()  # This prints the schema to the console
        print("-" * 30)

        print("Preparing dataset...")

    except Exception as e:
        print(f"Error reading parquet file: {e}")


# --- THIS IS THE PART YOU WERE MISSING ---
if __name__ == "__main__":
    main()
