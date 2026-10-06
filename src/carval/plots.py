from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carval.evaluate import CLASSES


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_price_unit_fix(df: pd.DataFrame, path: Path) -> None:
    models = [
        "Toyota Prado",
        "Toyota Fortuner",
        "Toyota Land Cruiser",
        "Toyota Hilux",
        "Suzuki Mehran",
        "Toyota Corolla",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=False)
    named = df["make"] + " " + df["model_name"]
    for ax, model in zip(axes.ravel(), models):
        part = df.loc[named == model]
        if part.empty:
            ax.set_visible(False)
            continue
        grouped = part.groupby("year").agg(
            raw=("price", "median"), fixed=("price_lakh", "median")
        )
        ax.plot(
            grouped.index,
            grouped["raw"],
            color="#9aa0a6",
            marker="o",
            ms=3,
            label="raw",
        )
        ax.plot(
            grouped.index,
            grouped["fixed"],
            color="#1f4e79",
            marker="o",
            ms=3,
            label="fixed",
        )
        ax.set_title(model, fontsize=10)
        ax.set_ylabel("median price (lakh)")
        ax.set_xlabel("year")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Price unit fix: median by year", fontsize=12)
    _save(fig, path)


def plot_pred_vs_actual(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(y_true, y_pred, s=8, alpha=0.25, color="#1f4e79", linewidths=0)
    limit = float(np.quantile(np.concatenate([y_true, y_pred]), 0.995))
    ax.plot([0, limit], [0, limit], color="#333333", lw=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("actual price (lakh)")
    ax.set_ylabel("predicted price (lakh)")
    ax.set_title("Test set: predicted vs actual")
    _save(fig, path)


def plot_residuals(y_true, y_pred, path: Path) -> None:
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(y_pred, resid, s=8, alpha=0.25, color="#1f4e79", linewidths=0)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xlabel("predicted price (lakh)")
    ax.set_ylabel("prediction - actual (lakh)")
    ax.set_title("Test residuals")
    _save(fig, path)


def plot_learning_curve(train_sizes, train_scores, val_scores, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(
        train_sizes,
        train_scores.mean(axis=1),
        marker="o",
        label="train",
        color="#9aa0a6",
    )
    ax.plot(
        train_sizes,
        val_scores.mean(axis=1),
        marker="o",
        label="validation",
        color="#1f4e79",
    )
    ax.set_xlabel("training rows")
    ax.set_ylabel("MAE (lakh)")
    ax.set_title("Learning curve")
    ax.legend()
    _save(fig, path)


def plot_importance(
    names,
    importances,
    path: Path,
    xlabel: str = "permutation importance (MAE increase, lakh)",
) -> None:
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(np.array(names)[order], np.array(importances)[order], color="#1f4e79")
    ax.set_xlabel(xlabel)
    ax.set_title("What the model uses")
    _save(fig, path)


def plot_confusion(matrix, path: Path) -> None:
    grid = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(grid, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("from price shift")
    ax.set_title("Perturbation test")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j, i, f"{int(grid[i, j])}", ha="center", va="center", color="#222222"
            )
    _save(fig, path)


def plot_deal_mix(counts: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    labels = [c for c in CLASSES if c in counts.index]
    values = [int(counts[c]) for c in labels]
    colors = {"UNDERPRICED": "#2e7d32", "FAIR": "#1f4e79", "OVERPRICED": "#b85c38"}
    ax.bar(labels, values, color=[colors[c] for c in labels])
    ax.set_ylabel("training listings (out of fold)")
    ax.set_title("Deal labels on the training set")
    _save(fig, path)
