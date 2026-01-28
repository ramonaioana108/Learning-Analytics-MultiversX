import os
import json
import math
import argparse
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import List, Optional


# ============================================================
# DOWNLOAD (BigQuery)
# ============================================================
BQ_DATASET = "bigquery-public-data.crypto_multiversx_mainnet_eu"


def _bq_list_tables(client, dataset: str) -> List[str]:
    from google.cloud import bigquery
    ds = bigquery.DatasetReference.from_string(dataset)
    return [t.table_id for t in client.list_tables(ds)]


def _bq_pick_default_table(tables: List[str]) -> Optional[str]:
    priorities = ["blocks", "transactions", "txs", "transfers"]
    lower = {t.lower(): t for t in tables}
    for p in priorities:
        if p in lower:
            return lower[p]
    return tables[0] if tables else None


def _month_ranges(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        yield cur, min(end, nxt - timedelta(days=1))
        cur = nxt


def run_download(*, project: Optional[str], table: Optional[str], start: str, end: str,
                 mode: str, by_month: bool, out_dir: str) -> Path:
    from google.cloud import bigquery

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    client = bigquery.Client(project=project) if project else bigquery.Client()

    tables = _bq_list_tables(client, BQ_DATASET)
    chosen_table = table or _bq_pick_default_table(tables)
    if not chosen_table:
        raise ValueError(f"No tables found in dataset {BQ_DATASET}")
    if chosen_table not in tables:
        raise ValueError(
            f"Table {chosen_table} not found. Available: {tables}")

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    ranges = [(start_d, end_d)]
    if by_month:
        ranges = list(_month_ranges(start_d, end_d))

    print("=== DOWNLOAD ===")
    print("Dataset:", BQ_DATASET)
    print("Table:", chosen_table)
    print("Mode:", mode)
    print("Monthly split:", by_month)
    print("Out dir:", out_dir_p)

    last_out = None
    for r_start, r_end in ranges:
        print(f"\nDownloading: {r_start} -> {r_end}")

        if mode == "daily":
            query = f"""
            SELECT
              DATE(timestamp) AS day,
              COUNT(*) AS blocks_count,
              SUM(txCount) AS tx_count_sum,
              SUM(SAFE_CAST(accumulatedFees AS FLOAT64)) AS accumulated_fees_sum,
              SUM(SAFE_CAST(developerFees AS FLOAT64)) AS developer_fees_sum,
              AVG(SAFE_CAST(gasProvided AS FLOAT64)) AS gas_provided_avg,
              AVG(SAFE_CAST(gasRefunded AS FLOAT64)) AS gas_refunded_avg,
              AVG(SAFE_CAST(gasPenalized AS FLOAT64)) AS gas_penalized_avg
            FROM `{BQ_DATASET}.{chosen_table}`
            WHERE DATE(timestamp) BETWEEN @start AND @end
            GROUP BY day
            ORDER BY day
            """
        else:
            query = f"""
            SELECT
              timestamp,
              txCount,
              SAFE_CAST(gasProvided AS FLOAT64) AS gasProvided,
              SAFE_CAST(gasRefunded AS FLOAT64) AS gasRefunded,
              SAFE_CAST(gasPenalized AS FLOAT64) AS gasPenalized,
              SAFE_CAST(accumulatedFees AS FLOAT64) AS accumulatedFees,
              SAFE_CAST(developerFees AS FLOAT64) AS developerFees
            FROM `{BQ_DATASET}.{chosen_table}`
            WHERE DATE(timestamp) BETWEEN @start AND @end
            ORDER BY timestamp
            """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "DATE", str(r_start)),
                bigquery.ScalarQueryParameter("end", "DATE", str(r_end)),
            ]
        )

        job = client.query(query, job_config=job_config)
        df = job.result().to_dataframe(create_bqstorage_client=True)

        fname = f"{chosen_table}_{mode}_{r_start}_{r_end}.parquet"
        out_path = out_dir_p / fname
        df.to_parquet(out_path, index=False)
        last_out = out_path
        print("Saved:", out_path, "| rows:", len(df))

    if last_out is None:
        raise RuntimeError("Download produced no files.")
    return last_out


# ============================================================
# MERGE PRICE (Yahoo Finance)
# ============================================================
def run_merge_price(*, input_path: Path, out_dir: str, ticker: str, out_file: str) -> Path:
    import pandas as pd
    import yfinance as yf
    import pyarrow.parquet as pq

    print("\n=== MERGE PRICE ===")
    print(f"Loading blockchain data from {input_path}...")

    table = pq.read_table(str(input_path))
    df_blocks = pd.DataFrame(table.to_pylist())

    if "day" not in df_blocks.columns:
        raise RuntimeError(
            "Expected a 'day' column in blocks daily parquet. (Use --mode daily)")

    df_blocks["day"] = pd.to_datetime(df_blocks["day"])
    min_date = df_blocks["day"].min()
    max_date = df_blocks["day"].max()
    print(f"Date Range: {min_date.date()} -> {max_date.date()}")

    print(f"Downloading {ticker} price data...")
    yf_ticker = yf.Ticker(ticker)
    df_price = yf_ticker.history(
        start=min_date, end=max_date + pd.Timedelta(days=1))

    df_price = df_price.reset_index()
    if "Date" in df_price.columns:
        df_price["Date"] = pd.to_datetime(
            df_price["Date"]).dt.tz_localize(None)

    df_price = df_price[["Date", "Close", "Volume"]].rename(columns={
        "Date": "day",
        "Close": "price_usd",
        "Volume": "price_volume",
    })

    print("Merging datasets...")
    df_merged = pd.merge(df_blocks, df_price, on="day", how="left")

    df_merged["price_usd"] = df_merged["price_usd"].ffill().bfill()
    df_merged["price_volume"] = df_merged["price_volume"].ffill().bfill()

    out_path = Path(out_dir) / out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Converting timestamps to Spark-compatible format (datetime64[us])...")
    for c in df_merged.select_dtypes(include=["datetime64[ns]"]).columns:
        df_merged[c] = df_merged[c].astype("datetime64[us]")

    df_merged.to_parquet(out_path, index=False)
    print(f"✅ Saved merged data to: {out_path} | Rows: {len(df_merged)}")
    return out_path


# ============================================================
# PROCESS (Spark feature engineering)
# ============================================================
PROC_N_LAGS = 3
DATE_CANDIDATES = ["date", "day", "block_date", "dt", "blockDay"]

POSSIBLE_METRICS = [
    "blocks_count",
    "tx_count_sum",
    "accumulated_fees_sum",
    "developer_fees_sum",
    "gas_provided_avg",
    "gas_refunded_avg",
    "gas_penalized_avg",
    "price_usd",
    "price_volume",
]

LABEL_ALIASES = {
    "tx_count": "tx_count_sum",
    "accumulated_fees": "accumulated_fees_sum",
    "developer_fees": "developer_fees_sum",
}


def _first_existing(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def run_process(*, inputs: List[Path], out_dir: str, out_file: str, label_col: str,
                master: str, driver_memory: str, executor_memory: str,
                shuffle_partitions: int, parallelism: int) -> Path:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, to_date, lit,
        dayofweek, dayofyear, dayofmonth,
        month as Fmonth, weekofyear,
        last_day,
        sin, cos, lag,
        datediff, min as Fmin, max as Fmax,
        log1p, avg, stddev,
        isnan, when
    )
    from pyspark.sql.window import Window
    from pyspark.sql.types import DecimalType

    def ensure_date_column(df):
        date_col = _first_existing(df.columns, DATE_CANDIDATES)
        if not date_col:
            raise RuntimeError(
                f"Could not find date column. Tried: {DATE_CANDIDATES}. Have: {df.columns}"
            )
        return df.withColumn("date", to_date(col(date_col)))

    def resolve_label_col(requested: str, available_cols):
        if requested in available_cols:
            return requested
        if requested in LABEL_ALIASES and LABEL_ALIASES[requested] in available_cols:
            return LABEL_ALIASES[requested]
        return None

    def add_cyclical(df, base_col, period, out_prefix):
        two_pi = 2.0 * math.pi
        return (
            df.withColumn(base_col, col(base_col).cast("double"))
              .withColumn(f"{out_prefix}_sin", sin(lit(two_pi) * col(base_col) / lit(float(period))))
              .withColumn(f"{out_prefix}_cos", cos(lit(two_pi) * col(base_col) / lit(float(period))))
        )

    def nan_to_null(df, cols):
        for c in cols:
            if c in df.columns:
                df = df.withColumn(
                    c, when(isnan(col(c)), None).otherwise(col(c)))
        return df

    spark = (
        SparkSession.builder
        .appName("MultiversX-Process-Features-Tuned")
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(parallelism))
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.parquet.mergeSchema", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("\n=== PROCESS ===")
    print("Inputs:")
    for p in inputs:
        print(" -", p)

    df = spark.read.option("mergeSchema", "true").parquet(
        *[str(p) for p in inputs])

    for f in df.schema.fields:
        if isinstance(f.dataType, DecimalType):
            df = df.withColumn(f.name, col(f.name).cast("double"))

    df = ensure_date_column(df)

    available_metrics = [c for c in POSSIBLE_METRICS if c in df.columns]
    if not available_metrics:
        raise RuntimeError(
            f"No expected metrics found. Expected one of: {POSSIBLE_METRICS}. Got: {df.columns}"
        )

    resolved_label = resolve_label_col(label_col, df.columns)
    if not resolved_label:
        raise RuntimeError(
            f"Label col '{label_col}' not found. Available: {df.columns}. Aliases: {LABEL_ALIASES}"
        )

    print(f"Resolved label_col: {label_col} -> {resolved_label}")
    print("Available metrics:", available_metrics)

    cleaned = df.select(
        "date",
        *[col(c).cast("double").alias(c) for c in available_metrics]
    ).repartitionByRange(8, col("date")).sortWithinPartitions("date")

    min_date = cleaned.select(Fmin("date").alias("m")).collect()[0]["m"]
    cleaned = cleaned.withColumn("t", datediff(
        col("date"), lit(min_date)).cast("double"))

    w = Window.orderBy(col("date"))
    cleaned = cleaned.withColumn("label", lag(col(resolved_label), -1).over(w))

    cleaned = cleaned.withColumn("dow", (dayofweek(col("date")) - 1))
    cleaned = add_cyclical(cleaned, "dow", 7, "dow")

    cleaned = cleaned.withColumn("doy", (dayofyear(col("date")) - 1))
    cleaned = add_cyclical(cleaned, "doy", 365, "doy")

    cleaned = cleaned.withColumn(
        "month", (Fmonth(col("date")) - lit(1)).cast("double"))
    cleaned = add_cyclical(cleaned, "month", 12, "month")

    cleaned = cleaned.withColumn(
        "woy", (weekofyear(col("date")) - lit(1)).cast("double"))
    cleaned = add_cyclical(cleaned, "woy", 53, "woy")

    cleaned = cleaned.withColumn("is_month_start", when(
        dayofmonth(col("date")) == lit(1), 1.0).otherwise(0.0))
    cleaned = cleaned.withColumn("is_month_end", when(
        col("date") == last_day(col("date")), 1.0).otherwise(0.0))

    cleaned = cleaned.withColumn("tx_log", log1p(col(resolved_label)))
    cleaned = cleaned.withColumn("label_log", log1p(col("label")))

    w7 = Window.orderBy(col("date")).rowsBetween(-6, 0)
    w14 = Window.orderBy(col("date")).rowsBetween(-13, 0)
    w28 = Window.orderBy(col("date")).rowsBetween(-27, 0)

    cleaned = cleaned.withColumn("tx_log_ma7", avg(col("tx_log")).over(w7))
    cleaned = cleaned.withColumn("tx_log_ma14", avg(col("tx_log")).over(w14))
    cleaned = cleaned.withColumn("tx_log_ma28", avg(col("tx_log")).over(w28))

    cleaned = cleaned.withColumn("tx_log_std7", stddev(col("tx_log")).over(w7))
    cleaned = cleaned.withColumn(
        "tx_log_std14", stddev(col("tx_log")).over(w14))
    cleaned = cleaned.withColumn("tx_log_std7", when(
        col("tx_log_std7").isNull(), 0.0).otherwise(col("tx_log_std7")))
    cleaned = cleaned.withColumn("tx_log_std14", when(
        col("tx_log_std14").isNull(), 0.0).otherwise(col("tx_log_std14")))

    cleaned = cleaned.withColumn(
        "tx_log_z7",
        when(col("tx_log_std7") > lit(0.0), (col("tx_log") -
             col("tx_log_ma7")) / col("tx_log_std7")).otherwise(lit(0.0))
    )
    cleaned = cleaned.withColumn(
        "tx_log_z14",
        when(col("tx_log_std14") > lit(0.0), (col("tx_log") -
             col("tx_log_ma14")) / col("tx_log_std14")).otherwise(lit(0.0))
    )

    cleaned = cleaned.withColumn("tx_log_min14", Fmin(col("tx_log")).over(w14))
    cleaned = cleaned.withColumn("tx_log_max14", Fmax(col("tx_log")).over(w14))
    cleaned = cleaned.withColumn("tx_log_rng14", col(
        "tx_log_max14") - col("tx_log_min14"))

    cleaned = cleaned.withColumn("tx_log_min28", Fmin(col("tx_log")).over(w28))
    cleaned = cleaned.withColumn("tx_log_max28", Fmax(col("tx_log")).over(w28))
    cleaned = cleaned.withColumn("tx_log_rng28", col(
        "tx_log_max28") - col("tx_log_min28"))

    cleaned = cleaned.withColumn("trend7", col(
        "tx_log_ma7") - lag(col("tx_log_ma7"), 7).over(w))
    cleaned = cleaned.withColumn("trend14", col(
        "tx_log_ma14") - lag(col("tx_log_ma14"), 14).over(w))
    cleaned = cleaned.withColumn("regime_up", when(
        col("tx_log_ma7") > col("tx_log_ma28"), 1.0).otherwise(0.0))

    cleaned = cleaned.withColumn("tx_log_lag7", lag(col("tx_log"), 7).over(w))
    cleaned = cleaned.withColumn(
        "tx_log_lag14", lag(col("tx_log"), 14).over(w))
    cleaned = cleaned.withColumn(
        "tx_log_lag28", lag(col("tx_log"), 28).over(w))

    cleaned = cleaned.withColumn("delta1", col(
        "tx_log") - lag(col("tx_log"), 1).over(w))
    cleaned = cleaned.withColumn("delta7", col(
        "tx_log") - lag(col("tx_log"), 7).over(w))

    if "price_usd" in cleaned.columns:
        cleaned = cleaned.withColumn("price_log", log1p(col("price_usd")))
        cleaned = cleaned.withColumn("price_ret_1d", col(
            "price_log") - lag(col("price_log"), 1).over(w))
        cleaned = cleaned.withColumn("price_ret_7d", col(
            "price_log") - lag(col("price_log"), 7).over(w))

        cleaned = cleaned.withColumn(
            "price_volatility_7d", stddev(col("price_log")).over(w7))
        cleaned = cleaned.withColumn(
            "price_volatility_7d",
            when(col("price_volatility_7d").isNull(),
                 0.0).otherwise(col("price_volatility_7d"))
        )

    for m in available_metrics:
        for k in range(1, PROC_N_LAGS + 1):
            cleaned = cleaned.withColumn(f"{m}_lag{k}", lag(col(m), k).over(w))

    derived_tx = [
        "tx_log_std7", "tx_log_std14",
        "tx_log_z7", "tx_log_z14",
        "delta1", "delta7",
        "trend7", "trend14",
        "tx_log_rng14", "tx_log_rng28"
    ]
    for m in derived_tx:
        for k in range(1, PROC_N_LAGS + 1):
            cleaned = cleaned.withColumn(f"{m}_lag{k}", lag(col(m), k).over(w))

    must_have = [
        "label", "label_log",
        "tx_log", "tx_log_ma7", "tx_log_ma14", "tx_log_ma28",
        "tx_log_std7", "tx_log_std14",
        "tx_log_z7", "tx_log_z14",
        "tx_log_min14", "tx_log_max14", "tx_log_rng14",
        "tx_log_min28", "tx_log_max28", "tx_log_rng28",
        "trend7", "trend14", "regime_up",
        "tx_log_lag7", "tx_log_lag14", "tx_log_lag28",
        "delta1", "delta7",
        "dow_sin", "dow_cos",
        "doy_sin", "doy_cos",
        "month_sin", "month_cos",
        "woy_sin", "woy_cos",
        "is_month_start", "is_month_end",
        "t",
    ]
    for m in available_metrics:
        for k in range(1, PROC_N_LAGS + 1):
            must_have.append(f"{m}_lag{k}")
    for m in derived_tx:
        for k in range(1, PROC_N_LAGS + 1):
            must_have.append(f"{m}_lag{k}")

    cleaned = nan_to_null(cleaned, must_have).dropna(subset=must_have)

    for raw_cal in ["dow", "doy", "month", "woy"]:
        if raw_cal in cleaned.columns:
            cleaned = cleaned.drop(raw_cal)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / out_file

    cleaned.write.mode("overwrite").parquet(str(out_path))
    print(f"✅ Saved features: {out_path}")
    print("Rows:", cleaned.count(), "| Columns:", len(cleaned.columns))

    spark.stop()
    return out_path


# ============================================================
# TRAIN (Spark ML + optional XGBoost)
# ============================================================
def run_train(*, parquet_path: Path, models_dir: str,
              train_end: str, test_start: str, test_end: str,
              master: str, driver_memory: str, executor_memory: str,
              shuffle_partitions: int, parallelism: int,
              enable_xgboost: bool,
              lr_regparam: float, lr_elasticnet: float,
              dt_max_depth: int, dt_min_instances: int,
              rf_trees: int, rf_max_depth: int, rf_min_instances: int,
              rf_subsample: float, rf_feature_subset: str,
              seed: int,
              xgb_workers: int, xgb_eta: float, xgb_max_depth: int,
              xgb_subsample: float, xgb_colsample: float, xgb_lambda: float,
              xgb_estimators: int, xgb_tree_method: str, xgb_device: str) -> Path:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, expm1, greatest, lit
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator

    RAW_LABEL_COL = "label"
    LABEL_COL = "label_log"
    TODAY_LOG_COL = "tx_log"
    LABEL_DIFF_COL = "label_diff"

    # Optional XGBoost
    HAS_XGB = False
    SparkXGBRegressor = None
    if enable_xgboost:
        try:
            from xgboost.spark import SparkXGBRegressor as _SparkXGBRegressor
            SparkXGBRegressor = _SparkXGBRegressor
            HAS_XGB = True
        except Exception:
            HAS_XGB = False

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(models_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def dedupe_keep_order(cols):
        return list(dict.fromkeys(cols))

    def eval_delta_model(pred_df, name: str):
        pred_df = (
            pred_df
            .withColumn("pred_label_log", col(TODAY_LOG_COL) + col("prediction"))
            .withColumn("pred_label_raw", expm1(greatest(col("pred_label_log"), lit(0.0))))
            .cache()
        )
        pred_df.count()
        rmse_log = RegressionEvaluator(
            labelCol=LABEL_COL, predictionCol="pred_label_log", metricName="rmse").evaluate(pred_df)
        r2_log = RegressionEvaluator(
            labelCol=LABEL_COL, predictionCol="pred_label_log", metricName="r2").evaluate(pred_df)
        rmse_raw = RegressionEvaluator(
            labelCol=RAW_LABEL_COL, predictionCol="pred_label_raw", metricName="rmse").evaluate(pred_df)
        pred_df.unpersist()
        print(
            f"{name:18s} RMSE(log)={rmse_log:.6f}  R2(log)={r2_log:.6f}  RMSE(raw)={rmse_raw:.3f}")
        return rmse_log, r2_log, rmse_raw

    def save_model(model, subdir_name: str):
        path = str(run_dir / subdir_name)
        model.write().overwrite().save(path)
        print(f"Saved: {subdir_name} -> {path}")

    spark = (
        SparkSession.builder
        .appName("MultiversX-ML-Fast-Local")
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(parallelism))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("\n=== TRAIN ===")
    print("Reading parquet:", parquet_path)
    df = spark.read.parquet(str(parquet_path))

    df = df.withColumn(LABEL_DIFF_COL, col(LABEL_COL) - col(TODAY_LOG_COL))

    banned = {"date", RAW_LABEL_COL, LABEL_COL, TODAY_LOG_COL, LABEL_DIFF_COL}
    feature_cols = sorted(dedupe_keep_order(
        [c for c in df.columns if c not in banned]))
    needed = dedupe_keep_order(
        ["date", RAW_LABEL_COL, LABEL_COL, TODAY_LOG_COL, LABEL_DIFF_COL] + feature_cols)

    df_clean = df.select(*needed).dropna(subset=needed).cache()
    df_clean.count()

    train = df_clean.filter(col("date") <= train_end).cache()
    test = df_clean.filter((col("date") >= test_start) &
                           (col("date") <= test_end)).cache()

    n_train, n_test = train.count(), test.count()
    print(f"Train rows: {n_train} | Test rows: {n_test}")
    print(f"Using {len(feature_cols)} features.")

    train_base = train.select("date", RAW_LABEL_COL, LABEL_COL,
                              TODAY_LOG_COL, LABEL_DIFF_COL, *feature_cols).cache()
    test_base = test.select("date", RAW_LABEL_COL, LABEL_COL,
                            TODAY_LOG_COL, LABEL_DIFF_COL, *feature_cols).cache()
    train_base.count()
    test_base.count()

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    results = {}

    scaler = StandardScaler(
        inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
    lr = LinearRegression(
        featuresCol="scaledFeatures",
        labelCol=LABEL_DIFF_COL,
        predictionCol="prediction",
        regParam=lr_regparam,
        elasticNetParam=lr_elasticnet
    )
    lr_pipe = Pipeline(stages=[assembler, scaler, lr])
    lr_model = lr_pipe.fit(train_base)
    results["linear_regression"] = eval_delta_model(
        lr_model.transform(test_base), "Linear Reg")
    save_model(lr_model, "linear_regression")

    dt = DecisionTreeRegressor(
        featuresCol="features",
        labelCol=LABEL_DIFF_COL,
        predictionCol="prediction",
        maxDepth=dt_max_depth,
        minInstancesPerNode=dt_min_instances
    )
    dt_pipe = Pipeline(stages=[assembler, dt])
    dt_model = dt_pipe.fit(train_base)
    results["decision_tree"] = eval_delta_model(
        dt_model.transform(test_base), "Decision Tree")
    save_model(dt_model, "decision_tree")

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=LABEL_DIFF_COL,
        predictionCol="prediction",
        numTrees=rf_trees,
        maxDepth=rf_max_depth,
        minInstancesPerNode=rf_min_instances,
        subsamplingRate=rf_subsample,
        featureSubsetStrategy=rf_feature_subset,
        seed=seed
    )
    rf_pipe = Pipeline(stages=[assembler, rf])
    rf_model = rf_pipe.fit(train_base)
    results["random_forest"] = eval_delta_model(
        rf_model.transform(test_base), "Random Forest")
    save_model(rf_model, "random_forest")

    if enable_xgboost:
        if not HAS_XGB:
            print(
                "\n⚠️ XGBoost requested but xgboost.spark not available. Skipping XGBoost.")
        else:
            xgb_params = dict(
                features_col="features",
                label_col=LABEL_DIFF_COL,
                prediction_col="prediction",
                num_workers=min(xgb_workers, os.cpu_count() or 4),
                objective="reg:squarederror",
                eta=xgb_eta,
                max_depth=xgb_max_depth,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                reg_lambda=xgb_lambda,
                n_estimators=xgb_estimators,
                tree_method=xgb_tree_method,
                device=xgb_device,
                eval_metric="rmse",
                verbose_eval=False,
            )
            xgb = SparkXGBRegressor(**xgb_params)
            xgb_pipe = Pipeline(stages=[assembler, xgb])
            xgb_model = xgb_pipe.fit(train_base)
            results["xgboost"] = eval_delta_model(
                xgb_model.transform(test_base), "XGBoost")
            save_model(xgb_model, "xgboost")

    info = {
        "run_id": run_id,
        "parquet_path": str(parquet_path),
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "label_cols": {"raw": RAW_LABEL_COL, "log": LABEL_COL, "today_log": TODAY_LOG_COL, "diff": LABEL_DIFF_COL},
        "feature_cols": feature_cols,
        "results": {k: {"rmse_log": v[0], "r2_log": v[1], "rmse_raw": v[2]} for k, v in results.items()},
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print("✅ Saved run_info.json ->", run_dir / "run_info.json")

    spark.stop()
    return run_dir


# ============================================================
# PIPELINE (single command)
# ============================================================
def cmd_pipeline(args):
    # 1) download
    downloaded = run_download(
        project=args.project,
        table=args.table,
        start=args.start,
        end=args.end,
        mode=args.mode,
        by_month=args.by_month,
        out_dir=args.raw_out_dir,
    )

    # 2) merge price (only makes sense for daily)
    if args.mode != "daily":
        raise RuntimeError(
            "pipeline currently requires --mode daily (because merge-price expects a 'day' column).")

    merged = run_merge_price(
        input_path=downloaded,
        out_dir=args.raw_out_dir,
        ticker=args.ticker,
        out_file=args.merged_out_file,
    )

    # 3) process
    features = run_process(
        inputs=[merged],
        out_dir=args.parquet_out_dir,
        out_file=args.features_out_file,
        label_col=args.label_col,
        master=args.spark_master,
        driver_memory=args.spark_driver_memory,
        executor_memory=args.spark_executor_memory,
        shuffle_partitions=args.spark_shuffle_partitions,
        parallelism=args.spark_parallelism,
    )

    # 4) train
    run_dir = run_train(
        parquet_path=features,
        models_dir=args.models_dir,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        master=args.spark_master,
        driver_memory=args.spark_driver_memory,
        executor_memory=args.spark_executor_memory,
        shuffle_partitions=args.spark_shuffle_partitions,
        parallelism=args.spark_parallelism,
        enable_xgboost=args.enable_xgboost,
        lr_regparam=args.lr_regparam,
        lr_elasticnet=args.lr_elasticnet,
        dt_max_depth=args.dt_max_depth,
        dt_min_instances=args.dt_min_instances,
        rf_trees=args.rf_trees,
        rf_max_depth=args.rf_max_depth,
        rf_min_instances=args.rf_min_instances,
        rf_subsample=args.rf_subsample,
        rf_feature_subset=args.rf_feature_subset,
        seed=args.seed,
        xgb_workers=args.xgb_workers,
        xgb_eta=args.xgb_eta,
        xgb_max_depth=args.xgb_max_depth,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample=args.xgb_colsample,
        xgb_lambda=args.xgb_lambda,
        xgb_estimators=args.xgb_estimators,
        xgb_tree_method=args.xgb_tree_method,
        xgb_device=args.xgb_device,
    )

    print("\n=== DONE ===")
    print("Downloaded:", downloaded)
    print("Merged:", merged)
    print("Features:", features)
    print("Models run dir:", run_dir)


# ============================================================
# CLI
# ============================================================
def build_parser():
    ap = argparse.ArgumentParser(prog="multiversx_pipeline.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "pipeline", help="Run download -> merge-price -> process -> train")
    # download args
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--table", type=str, default=None)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--mode", choices=["daily", "raw"], default="daily")
    p.add_argument("--by_month", action="store_true")

    # output paths
    p.add_argument("--raw_out_dir", default="data_raw")
    p.add_argument("--parquet_out_dir", default="data_parquet")
    p.add_argument("--models_dir", default="models")
    p.add_argument("--merged_out_file",
                   default="blocks_daily_with_price.parquet")
    p.add_argument("--features_out_file",
                   default="multiversx_features_tuned.parquet")

    # price merge args
    p.add_argument("--ticker", default="EGLD-USD")

    # process args
    p.add_argument("--label_col", default="tx_count")

    # train split args
    p.add_argument("--train_end", default="2023-12-31")
    p.add_argument("--test_start", default="2024-01-01")
    p.add_argument("--test_end", default="2024-12-31")

    # Spark tuning
    p.add_argument("--spark_master", default="local[8]")
    p.add_argument("--spark_driver_memory", default="6g")
    p.add_argument("--spark_executor_memory", default="6g")
    p.add_argument("--spark_shuffle_partitions", type=int, default=64)
    p.add_argument("--spark_parallelism", type=int, default=64)

    # Train params
    p.add_argument("--lr_regparam", type=float, default=0.05)
    p.add_argument("--lr_elasticnet", type=float, default=0.3)
    p.add_argument("--dt_max_depth", type=int, default=10)
    p.add_argument("--dt_min_instances", type=int, default=10)
    p.add_argument("--rf_trees", type=int, default=200)
    p.add_argument("--rf_max_depth", type=int, default=8)
    p.add_argument("--rf_min_instances", type=int, default=10)
    p.add_argument("--rf_subsample", type=float, default=0.7)
    p.add_argument("--rf_feature_subset", default="sqrt")
    p.add_argument("--seed", type=int, default=42)

    # XGBoost
    p.add_argument("--enable_xgboost", action="store_true")
    p.add_argument("--xgb_workers", type=int, default=8)
    p.add_argument("--xgb_eta", type=float, default=0.05)
    p.add_argument("--xgb_max_depth", type=int, default=6)
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample", type=float, default=0.8)
    p.add_argument("--xgb_lambda", type=float, default=1.0)
    p.add_argument("--xgb_estimators", type=int, default=600)
    p.add_argument("--xgb_tree_method", default="hist")
    p.add_argument("--xgb_device", default="cpu")

    p.set_defaults(func=cmd_pipeline)
    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
