# make_plots.py
import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, expm1, greatest, lit

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator


# ✅ Updated: new features parquet produced by process_multiversx.py
FEATURES_PARQUET = "./data_parquet/multiversx_features_tuned.parquet"

# Columns used in your latest training
LABEL_RAW_COL = "label"
LABEL_LOG_COL = "label_log"
TODAY_LOG_COL = "tx_log"          # "today" in log space (same-day metric log)
PRED_COL = "prediction"           # model predicts label_diff


def reconstruct_predictions(pred_df):
    """
    Model predicts label_diff (delta in log space).
    Reconstruct:
      prediction_log = tx_log + prediction
      prediction_raw = expm1(max(prediction_log, 0))
    """
    pred_df = pred_df.withColumn(
        "prediction_log",
        col(TODAY_LOG_COL) + col(PRED_COL)
    )
    pred_df = pred_df.withColumn(
        "prediction_raw",
        expm1(greatest(col("prediction_log"), lit(0.0)))
    )
    return pred_df


def eval_metrics_log(pred_df):
    rmse = RegressionEvaluator(
        labelCol=LABEL_LOG_COL,
        predictionCol="prediction_log",
        metricName="rmse"
    ).evaluate(pred_df)

    r2 = RegressionEvaluator(
        labelCol=LABEL_LOG_COL,
        predictionCol="prediction_log",
        metricName="r2"
    ).evaluate(pred_df)

    return rmse, r2


def eval_metrics_raw(pred_df):
    rmse = RegressionEvaluator(
        labelCol=LABEL_RAW_COL,
        predictionCol="prediction_raw",
        metricName="rmse"
    ).evaluate(pred_df)

    r2 = RegressionEvaluator(
        labelCol=LABEL_RAW_COL,
        predictionCol="prediction_raw",
        metricName="r2"
    ).evaluate(pred_df)

    return rmse, r2


def safe_to_pandas(df, max_rows=50000):
    cnt = df.count()
    if cnt > max_rows:
        df = df.limit(max_rows)
    return df.toPandas()


def plot_bar(d, title, ylabel, out_path):
    names = list(d.keys())
    vals = [d[k] for k in names]
    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(y, p, title, out_path, xlabel="Real", ylabel="Predicție"):
    plt.figure()
    plt.scatter(y, p, s=12)
    mn = float(min(y.min(), p.min()))
    mx = float(max(y.max(), p.max()))
    plt.plot([mn, mx], [mn, mx], linewidth=1)  # y=x
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_timeseries(ts, y, p, title, out_path, ylabel="Value"):
    plt.figure()
    plt.plot(ts, y, label="Real")
    plt.plot(ts, p, label="Pred")
    plt.title(title)
    plt.xlabel("Timp")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residual_hist(resid, title, out_path):
    plt.figure()
    plt.hist(resid, bins=40)
    plt.title(title)
    plt.xlabel("Eroare (pred - real)")
    plt.ylabel("Frecvență")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)

    # ✅ Default to your 2024 test interval
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")

    ap.add_argument("--out_dir", type=str, default="plots")

    # ✅ auto-detect models if not provided
    ap.add_argument("--models", nargs="*", default=None,
                    help="Optional list of model subfolders in run_dir. If omitted, auto-detects.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("MultiversX-Make-Plots")
        .master("local[8]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .getOrCreate()
    )

    df = spark.read.parquet(FEATURES_PARQUET).withColumn(
        "date", to_date(col("date")))

    test = (
        df.filter((col("date") >= args.start) & (col("date") <= args.end))
          .cache()
    )
    print("Test rows:", test.count())

    # ✅ Determine which models to load
    if args.models:
        model_names = args.models
    else:
        # auto-detect: any subdir in run_dir that contains Spark model metadata
        candidates = [p.name for p in run_dir.iterdir() if p.is_dir()]
        # keep a sane ordering if they exist
        preferred = ["linear_regression", "decision_tree",
                     "random_forest"]
        model_names = [m for m in preferred if m in candidates]
        # add any extras at the end
        for m in candidates:
            if m not in model_names:
                model_names.append(m)

    if not model_names:
        raise RuntimeError(f"No model subfolders found in: {run_dir}")

    rmse_log = {}
    r2_log = {}
    rmse_raw = {}
    r2_raw = {}
    preds_pd = {}

    for name in model_names:
        model_path = run_dir / name
        print("Loading:", model_path)
        model = PipelineModel.load(str(model_path))

        pred = model.transform(test)
        pred = reconstruct_predictions(pred).cache()

        m_rmse_log, m_r2_log = eval_metrics_log(pred)
        m_rmse_raw, m_r2_raw = eval_metrics_raw(pred)

        rmse_log[name] = float(m_rmse_log)
        r2_log[name] = float(m_r2_log)
        rmse_raw[name] = float(m_rmse_raw)
        r2_raw[name] = float(m_r2_raw)

        small = (
            pred.select("date", LABEL_RAW_COL, LABEL_LOG_COL,
                        "prediction_log", "prediction_raw")
            .orderBy("date")
        )
        pdf = safe_to_pandas(small, max_rows=20000)
        pdf["date"] = pd.to_datetime(pdf["date"])
        preds_pd[name] = pdf

        # --- per-model plots (LOG) ---
        plot_timeseries(
            pdf["date"], pdf[LABEL_LOG_COL], pdf["prediction_log"],
            f"Pred vs Real (LOG) - {name}",
            out_dir / f"timeseries_log_{name}.png",
            ylabel="label_log"
        )
        plot_scatter(
            pdf[LABEL_LOG_COL], pdf["prediction_log"],
            f"Scatter (LOG) Pred vs Real - {name}",
            out_dir / f"scatter_log_{name}.png",
            xlabel="Real (label_log)",
            ylabel="Pred (prediction_log)"
        )
        plot_residual_hist(
            (pdf["prediction_log"] - pdf[LABEL_LOG_COL]),
            f"Residuals histogram (LOG) - {name}",
            out_dir / f"residuals_log_{name}.png"
        )

        # --- per-model plots (RAW) ---
        plot_timeseries(
            pdf["date"], pdf[LABEL_RAW_COL], pdf["prediction_raw"],
            f"Pred vs Real (RAW) - {name}",
            out_dir / f"timeseries_raw_{name}.png",
            ylabel="label"
        )
        plot_scatter(
            pdf[LABEL_RAW_COL], pdf["prediction_raw"],
            f"Scatter (RAW) Pred vs Real - {name}",
            out_dir / f"scatter_raw_{name}.png",
            xlabel="Real (label)",
            ylabel="Pred (prediction_raw)"
        )
        plot_residual_hist(
            (pdf["prediction_raw"] - pdf[LABEL_RAW_COL]),
            f"Residuals histogram (RAW) - {name}",
            out_dir / f"residuals_raw_{name}.png"
        )

        print(f"{name:18s}  RMSE(log)={m_rmse_log:.6f}  R2(log)={m_r2_log:.6f}  RMSE(raw)={m_rmse_raw:.3f}  R2(raw)={m_r2_raw:.6f}")

    # --- comparison plots ---
    plot_bar(rmse_log, "Compararea RMSE (LOG) între modele",
             "RMSE (log)", out_dir / "compare_rmse_log.png")
    plot_bar(r2_log, "Compararea R² (LOG) între modele",
             "R² (log)", out_dir / "compare_r2_log.png")

    plot_bar(rmse_raw, "Compararea RMSE (RAW) între modele",
             "RMSE (raw)", out_dir / "compare_rmse_raw.png")
    plot_bar(r2_raw, "Compararea R² (RAW) între modele",
             "R² (raw)", out_dir / "compare_r2_raw.png")

    # --- all models vs real (LOG) ---
    if preds_pd:
        base = list(preds_pd.values())[0][["date", LABEL_LOG_COL]].copy()
        plt.figure()
        plt.plot(base["date"], base[LABEL_LOG_COL], label="Real (label_log)")
        for name, pdf in preds_pd.items():
            plt.plot(pdf["date"], pdf["prediction_log"], label=f"Pred: {name}")
        plt.title("Predicții vs Real - toate modelele (LOG)")
        plt.xlabel("Timp")
        plt.ylabel("label_log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "timeseries_all_models_log.png", dpi=200)
        plt.close()

    # --- all models vs real (RAW) ---
    if preds_pd:
        base = list(preds_pd.values())[0][["date", LABEL_RAW_COL]].copy()
        plt.figure()
        plt.plot(base["date"], base[LABEL_RAW_COL], label="Real (label)")
        for name, pdf in preds_pd.items():
            plt.plot(pdf["date"], pdf["prediction_raw"], label=f"Pred: {name}")
        plt.title("Predicții vs Real - toate modelele (RAW)")
        plt.xlabel("Timp")
        plt.ylabel("label")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "timeseries_all_models_raw.png", dpi=200)
        plt.close()

    summary = {
        "run_dir": str(run_dir),
        "features_parquet": FEATURES_PARQUET,
        "date_range": {"start": args.start, "end": args.end},
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "rmse_raw": rmse_raw,
        "r2_raw": r2_raw,
        "models_loaded": model_names,
        "generated_at": datetime.now().isoformat()
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved:", out_dir / "metrics_summary.json")

    spark.stop()


if __name__ == "__main__":
    main()
