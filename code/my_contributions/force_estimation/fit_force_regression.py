import argparse
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # basic sanity
    required_cols = {"force_N", "mean_depth_mm", "max_depth_mm", "sum_depth_mm", "area_px"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def get_models() -> dict:
    return {
        "linear_mean": Pipeline([
            ("reg", LinearRegression())
        ]),
        "linear_sum": Pipeline([
            ("reg", LinearRegression())
        ]),
        "ridge_mean+area": Pipeline([
            ("reg", Ridge(alpha=1.0))
        ]),
        "poly2_mean": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", Ridge(alpha=1.0))
        ]),
    }


def evaluate_models(df: pd.DataFrame, out_dir: Path) -> dict:
    X_dict = {
        "linear_mean": df[["mean_depth_mm"]].values,
        "linear_sum": df[["sum_depth_mm"]].values,
        "ridge_mean+area": df[["mean_depth_mm", "area_px"]].values,
        "poly2_mean": df[["mean_depth_mm"]].values,
    }
    y = df["force_N"].values

    kf = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=0)
    models = get_models()

    metrics = {}
    for name, model in models.items():
        X = X_dict[name]
        preds = np.zeros_like(y, dtype=float)
        fold_metrics = []
        for train_idx, test_idx in kf.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            preds[test_idx] = y_pred
            fold_metrics.append({
                "MAE": float(mean_absolute_error(y[test_idx], y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y[test_idx], y_pred))),
                "R2": float(r2_score(y[test_idx], y_pred)),
            })
        metrics[name] = {
            "MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
            "RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
            "R2": float(np.mean([m["R2"] for m in fold_metrics])),
        }

        # Save scatter plot force vs prediction
        plt.figure(figsize=(5,4), dpi=200)
        plt.scatter(y, preds, c="tab:blue", s=16)
        lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
        plt.plot(lims, lims, "k--", linewidth=1)
        plt.xlabel("Measured Force (N)")
        plt.ylabel("Predicted Force (N)")
        plt.title(f"{name}: MAE={metrics[name]['MAE']:.2f}, R2={metrics[name]['R2']:.2f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"pred_vs_meas_{name}.png")
        plt.close()

        # Save force vs mean_depth with fit line where applicable
        if name in ("linear_mean", "poly2_mean", "ridge_mean+area"):
            X_plot = df[["mean_depth_mm"]].values
            model.fit(X, y)
            y_fit = model.predict(X)
            plt.figure(figsize=(5,4), dpi=200)
            plt.scatter(df["mean_depth_mm"], y, c="tab:gray", s=16, label="data")
            order = np.argsort(df["mean_depth_mm"].values)
            plt.plot(df["mean_depth_mm"].values[order], y_fit[order], "r-", label="fit")
            plt.xlabel("Mean Depth (mm)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"force_vs_mean_depth_{name}.png")
            plt.close()

    return metrics


def train_and_export_best(df: pd.DataFrame, out_dir: Path) -> dict:
    models = get_models()
    X = df[["mean_depth_mm"]].values
    y = df["force_N"].values

    # Choose a simple, interpretable model: linear on mean_depth
    model = models["linear_mean"]
    model.fit(X, y)

    # Save coefficients
    reg = model.named_steps["reg"]
    params = {"intercept": float(reg.intercept_), "coef_mean_depth": float(reg.coef_[0])}
    with open(out_dir / "force_regression_linear_mean.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return params


def main():
    parser = argparse.ArgumentParser(description="Fit force regressions and export plots + coefficients")
    parser.add_argument("--csv", default="force_estimation/force_depth_dataset.csv")
    parser.add_argument("--out_dir", default="force_estimation/analysis")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = (project_root / args.csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)

    df = load_dataset(str(csv_path))
    metrics = evaluate_models(df, out_dir)
    params = train_and_export_best(df, out_dir)

    # Save metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(" - plots →", out_dir)
    print(" - metrics.json and force_regression_linear_mean.json")


if __name__ == "__main__":
    main()
