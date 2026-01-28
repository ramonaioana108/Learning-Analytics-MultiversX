# test_saved_models.py
import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

PARQUET_PATH = "./data_parquet/multiversx_daily_features.parquet"


def eval_model(pred, name):
    rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(pred)
    r2 = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2").evaluate(pred)
    print(f"{name:18s}  RMSE={rmse:.6f}  R2={r2:.6f}")
    return rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="models/<RUN_ID>")
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    spark = (
        SparkSession.builder
        .appName("MultiversX-Test-Saved")
        .master("local[8]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .getOrCreate()
    )

    df = spark.read.parquet(PARQUET_PATH)
    test = df.filter((col("date") >= args.start) &
                     (col("date") <= args.end)).cache()
    print("Test rows:", test.count())

    for name in ["linear_regression", "random_forest", "gbt_regressor"]:
        path = run_dir / name
        print("\nLoading model:", name)
        model = PipelineModel.load(str(path))
        pred = model.transform(test)
        eval_model(pred, name)

    spark.stop()


if __name__ == "__main__":
    main()
