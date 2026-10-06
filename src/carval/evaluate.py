from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from carval.scoring import verdict

CLASSES = ["UNDERPRICED", "FAIR", "OVERPRICED"]


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y_true = np.expm1(np.asarray(y_true_log, dtype=float))
    y_pred = np.clip(np.expm1(np.asarray(y_pred_log, dtype=float)), 0.1, None)
    ape = np.abs(y_pred - y_true) / np.clip(y_true, 1e-6, None)
    log_true = np.asarray(y_true_log, dtype=float)
    log_pred = np.asarray(y_pred_log, dtype=float)
    return {
        "mae_lakh": float(mean_absolute_error(y_true, y_pred)),
        "rmse_lakh": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_lakh": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_20pct": float(np.mean(ape <= 0.20)),
        "mae_log": float(mean_absolute_error(log_true, log_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(log_true, log_pred))),
        "r2_log": float(r2_score(log_true, log_pred)),
    }


def price_bucket(price: pd.Series) -> pd.Series:
    return pd.cut(
        price,
        bins=[-np.inf, 20, 60, np.inf],
        labels=["budget", "mid", "premium"],
    )


def age_bucket(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=[-np.inf, 4, 9, 19, np.inf],
        labels=["0-4", "5-9", "10-19", "20+"],
    )


def segment_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(column, observed=True):
        metrics = regression_metrics(part["y_true_log"], part["y_pred_log"])
        rows.append(
            {"segment": column, "level": str(key), "n": int(len(part)), **metrics}
        )
    return pd.DataFrame(rows)


def range_metrics(y_true_lakh, low, high) -> dict:
    y_true = np.asarray(y_true_lakh, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    covered = (y_true >= low) & (y_true <= high)
    width = high - low
    return {
        "coverage": float(np.mean(covered)),
        "mean_width_lakh": float(np.mean(width)),
        "median_width_lakh": float(np.median(width)),
        "mean_width_to_price": float(np.mean(width / np.clip(y_true, 1e-6, None))),
    }


def apply_log_interval(y_pred_log, q_low: float, q_high: float):
    pred_log = np.asarray(y_pred_log, dtype=float)
    point = np.expm1(pred_log)
    low = np.expm1(pred_log + q_low)
    high = np.expm1(pred_log + q_high)
    low = np.minimum(low, point)
    high = np.maximum(high, point)
    low = np.maximum(low, 0.3)
    return point, low, high


def conformal_quantiles(
    y_true_log, y_pred_log, low_q: float = 0.10, high_q: float = 0.90
):
    resid = np.asarray(y_true_log, dtype=float) - np.asarray(y_pred_log, dtype=float)
    n = len(resid)
    # Slightly wider than the raw percentile so finite-sample coverage stays near 80%.
    lo_level = min(low_q * (n + 1) / n, 1.0)
    hi_level = min(high_q * (n + 1) / n, 1.0)
    q_low = float(np.quantile(resid, lo_level))
    q_high = float(np.quantile(resid, hi_level))
    if q_low > 0:
        q_low = min(q_low, 0.0)
    if q_high < 0:
        q_high = max(q_high, 0.0)
    return q_low, q_high


def label_listings(asking, predicted, low, high) -> list[str]:
    labels = []
    for ask, pred, lo, hi in zip(asking, predicted, low, high):
        labels.append(verdict(float(ask), float(pred), float(lo), float(hi)))
    return labels


def deal_classification(y_true: list[str], y_pred: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    per_class = {}
    for name, p, r, f, n in zip(CLASSES, precision, recall, f1, support):
        per_class[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(n),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": CLASSES,
    }
