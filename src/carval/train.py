"""Fit the valuation model and write metrics, figures, and the saved pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from carval import __version__
from carval.cleaning import clean_listings
from carval.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURE_COLUMNS,
    FIGURES,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_CSV,
    RANDOM_STATE,
    RARE_MIN_COUNT,
    REPORTS,
    TEST_SIZE,
)
from carval.evaluate import (
    age_bucket,
    apply_log_interval,
    conformal_quantiles,
    deal_classification,
    label_listings,
    price_bucket,
    range_metrics,
    regression_metrics,
    segment_table,
)
from carval.features import add_features
from carval.plots import (
    plot_confusion,
    plot_deal_mix,
    plot_importance,
    plot_learning_curve,
    plot_pred_vs_actual,
    plot_price_unit_fix,
    plot_residuals,
)

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None
try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


class GroupRareCategories(BaseEstimator, TransformerMixin):
    """Collapse infrequent category levels so one-hot columns stay stable."""

    def __init__(self, min_count: int = RARE_MIN_COUNT):
        self.min_count = min_count

    def fit(self, X, y=None):
        frame = np.asarray(X).astype(str)
        self.keep_ = []
        for col in range(frame.shape[1]):
            values, counts = np.unique(frame[:, col], return_counts=True)
            keep = set(values[counts >= self.min_count].tolist())
            keep.add("Other")
            self.keep_.append(keep)
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(len(self.keep_))], dtype=object)
        return np.asarray(input_features, dtype=object)

    def transform(self, X):
        frame = np.asarray(X).astype(str)
        out = np.empty(frame.shape, dtype=object)
        for col, keep in enumerate(self.keep_):
            known = np.array(sorted(keep), dtype=object)
            out[:, col] = np.where(
                np.isin(frame[:, col], known), frame[:, col], "Other"
            )
        return out


class MakeModelMedian(BaseEstimator):
    """Baseline: median log-price of the same make and model."""

    def fit(self, X, y):
        frame = pd.DataFrame(
            {
                "make": np.asarray(X["make"]).astype(str),
                "model": np.asarray(X["model_name"]).astype(str),
                "y": np.asarray(y, dtype=float),
            }
        )
        self.global_ = float(np.median(frame["y"]))
        self.by_make_ = frame.groupby("make")["y"].median().to_dict()
        grouped = frame.groupby(["make", "model"])["y"].median()
        self.by_model_ = {key: float(value) for key, value in grouped.items()}
        return self

    def predict(self, X):
        makes = np.asarray(X["make"]).astype(str)
        models = np.asarray(X["model_name"]).astype(str)
        out = np.empty(len(makes), dtype=float)
        for i, (make, model) in enumerate(zip(makes, models)):
            if (make, model) in self.by_model_:
                out[i] = self.by_model_[(make, model)]
            elif make in self.by_make_:
                out[i] = self.by_make_[make]
            else:
                out[i] = self.global_
        return out


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("rare", GroupRareCategories(min_count=RARE_MIN_COUNT)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def _mae_lakh(y_true, y_pred) -> float:
    return float(
        mean_absolute_error(np.expm1(y_true), np.clip(np.expm1(y_pred), 0.1, None))
    )


def _candidate_spaces() -> dict:
    spaces = {
        "ridge": (
            Ridge(),
            {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
        ),
        "random_forest": (
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=2,
                max_features="sqrt",
            ),
            {
                "model__n_estimators": [120, 180],
                "model__max_depth": [12, 18],
                "model__min_samples_leaf": [2, 6],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.06, 0.1],
                "model__max_depth": [6, 10],
                "model__max_iter": [200, 350],
                "model__min_samples_leaf": [15, 30],
            },
        ),
    }
    if LGBMRegressor is not None:
        spaces["lightgbm"] = (
            LGBMRegressor(
                random_state=RANDOM_STATE,
                n_jobs=2,
                verbosity=-1,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
            {
                "model__n_estimators": [300, 500],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.05, 0.1],
                "model__min_child_samples": [20, 50],
            },
        )
    if XGBRegressor is not None:
        spaces["xgboost"] = (
            XGBRegressor(
                random_state=RANDOM_STATE,
                n_jobs=2,
                verbosity=0,
                objective="reg:squarederror",
            ),
            {
                "model__n_estimators": [250, 400],
                "model__max_depth": [4, 7],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.85, 1.0],
            },
        )
    if CatBoostRegressor is not None:
        spaces["catboost"] = (
            CatBoostRegressor(
                random_seed=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
                thread_count=2,
            ),
            {
                "model__depth": [6, 8],
                "model__learning_rate": [0.05, 0.08],
                "model__iterations": [250, 400],
            },
        )
    return spaces


def _cv_baseline(X: pd.DataFrame, y: np.ndarray, bins: np.ndarray) -> dict:
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = []
    for train_idx, valid_idx in splitter.split(X, bins):
        model = MakeModelMedian()
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        scores.append(_mae_lakh(y[valid_idx], pred))
    return {
        "model": "median_make_model",
        "cv_mae_mean": float(np.mean(scores)),
        "cv_mae_std": float(np.std(scores)),
        "best_params": None,
    }


def _tune(name: str, estimator, space: dict, X, y, bins) -> tuple[Pipeline, dict]:
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    # Precomputed splits so the folds follow price bins, not the continuous target.
    splits = list(
        StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ).split(np.zeros(len(X)), bins)
    )
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=min(4, _grid_size(space)),
        scoring=make_scorer(_mae_lakh, greater_is_better=False),
        cv=splits,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    best_idx = int(search.best_index_)
    mean_neg = float(search.cv_results_["mean_test_score"][best_idx])
    std_neg = float(search.cv_results_["std_test_score"][best_idx])
    row = {
        "model": name,
        "cv_mae_mean": float(-mean_neg),
        "cv_mae_std": float(std_neg),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
    }
    return search.best_estimator_, row


def _grid_size(space: dict) -> int:
    size = 1
    for values in space.values():
        size *= len(values)
    return size


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _library_versions() -> dict:
    import numpy
    import pandas
    import sklearn

    versions = {
        "python": __import__("sys").version.split()[0],
        "carval": __version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for name in ("lightgbm", "xgboost", "catboost", "matplotlib", "shap"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            continue
    return versions


def _out_of_fold(pipe: Pipeline, X, y, bins) -> np.ndarray:
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    oof = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in splitter.split(X, bins):
        model = clone(pipe)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = model.predict(X.iloc[valid_idx])
    return oof


def _perturbation(actual, predicted, low, high) -> dict:
    expected = []
    guessed = []
    per_shift = {}
    for factor, label in ((0.75, "UNDERPRICED"), (1.0, "FAIR"), (1.25, "OVERPRICED")):
        asking = actual * factor
        pred_labels = label_listings(asking, predicted, low, high)
        true_labels = [label] * len(asking)
        per_shift[label] = deal_classification(true_labels, pred_labels)
        expected.extend(true_labels)
        guessed.extend(pred_labels)
    overall = deal_classification(expected, guessed)
    overall["by_shift"] = {
        key: {
            "recall": value["per_class"][key]["recall"],
            "precision": value["per_class"][key]["precision"],
        }
        for key, value in per_shift.items()
    }
    return overall


def train() -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("cleaning")
    clean, cleaning_stats = clean_listings()
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_CSV, index=False)
    (REPORTS / "cleaning_stats.json").write_text(json.dumps(cleaning_stats, indent=2))
    plot_price_unit_fix(clean, FIGURES / "price_unit_before_after.png")

    featured = add_features(clean)
    X = featured[FEATURE_COLUMNS].reset_index(drop=True)
    y = np.log1p(featured["price_lakh"].to_numpy(dtype=float))
    bins = pd.qcut(
        featured["price_lakh"], q=5, labels=False, duplicates="drop"
    ).to_numpy()
    index = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=bins
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    bins_train = np.asarray(
        pd.qcut(np.expm1(y_train), q=5, labels=False, duplicates="drop")
    )

    print("baseline")
    rows = [_cv_baseline(X_train, y_train, bins_train)]
    baseline = MakeModelMedian().fit(X_train, y_train)
    baseline_test = regression_metrics(y_test, baseline.predict(X_test))

    fitted = {}
    for name, (estimator, space) in _candidate_spaces().items():
        print(f"tuning {name}")
        model, row = _tune(name, estimator, space, X_train, y_train, bins_train)
        print(f"  cv mae {row['cv_mae_mean']:.3f} +/- {row['cv_mae_std']:.3f}")
        rows.append(row)
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("cv_mae_mean")
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    tuned = comparison[comparison["model"] != "median_make_model"]
    winner_name = str(tuned.loc[tuned["cv_mae_mean"].idxmin(), "model"])
    winner = fitted[winner_name]
    print(f"winner {winner_name}")

    print("out-of-fold residuals")
    oof = _out_of_fold(winner, X_train, y_train, bins_train)
    q_low, q_high = conformal_quantiles(y_train, oof)
    oof_point, oof_low, oof_high = apply_log_interval(oof, q_low, q_high)
    oof_actual = np.expm1(y_train)
    oof_labels = pd.Series(label_listings(oof_actual, oof_point, oof_low, oof_high))
    oof_coverage = range_metrics(oof_actual, oof_low, oof_high)

    print("test evaluation")
    test_log = winner.predict(X_test)
    test_metrics = regression_metrics(y_test, test_log)
    point, low, high = apply_log_interval(test_log, q_low, q_high)
    actual = np.expm1(y_test)
    interval = range_metrics(actual, low, high)

    test_frame = X_test.copy()
    test_frame["y_true_log"] = y_test
    test_frame["y_pred_log"] = test_log
    test_frame["price_lakh"] = actual
    segments = pd.concat(
        [
            segment_table(
                test_frame.assign(make_group=_top_makes(test_frame["make"])),
                "make_group",
            ),
            segment_table(
                test_frame.assign(price_bucket=price_bucket(test_frame["price_lakh"])),
                "price_bucket",
            ),
            segment_table(test_frame, "fuel"),
            segment_table(
                test_frame.assign(age_bucket=age_bucket(test_frame["car_age"])),
                "age_bucket",
            ),
        ],
        ignore_index=True,
    )
    segments.to_csv(REPORTS / "segment_errors.csv", index=False)

    perturbation = _perturbation(actual, point, low, high)
    plot_pred_vs_actual(actual, point, FIGURES / "pred_vs_actual.png")
    plot_residuals(actual, point, FIGURES / "residuals.png")
    plot_confusion(perturbation["confusion_matrix"], FIGURES / "confusion_matrix.png")
    plot_deal_mix(oof_labels.value_counts(), FIGURES / "deal_label_mix.png")

    print("importance and learning curve")
    sample_n = min(1500, len(X_test))
    sample = X_test.sample(sample_n, random_state=RANDOM_STATE)
    importance = permutation_importance(
        winner,
        sample,
        y_test[sample.index.to_numpy()],
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring=make_scorer(_mae_lakh, greater_is_better=False),
        n_jobs=1,
    )
    order = np.argsort(importance.importances_mean)[::-1]
    importance_rows = [
        {
            "feature": FEATURE_COLUMNS[int(i)],
            "importance_mae": float(importance.importances_mean[i]),
            "std": float(importance.importances_std[i]),
        }
        for i in order
    ]
    pd.DataFrame(importance_rows).to_csv(
        REPORTS / "permutation_importance.csv", index=False
    )
    plot_importance(
        [row["feature"] for row in importance_rows],
        [row["importance_mae"] for row in importance_rows],
        FIGURES / "permutation_importance.png",
    )
    _try_shap(winner, sample, FIGURES / "shap_summary.png")

    from sklearn.model_selection import learning_curve

    sizes, train_scores, val_scores = learning_curve(
        clone(winner),
        X_train,
        y_train,
        cv=3,
        train_sizes=np.linspace(0.25, 1.0, 4),
        scoring=make_scorer(_mae_lakh, greater_is_better=False),
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    # scorer is negative MAE
    plot_learning_curve(
        sizes, -train_scores, -val_scores, FIGURES / "learning_curve.png"
    )

    frequent = (
        featured.iloc[train_idx]
        .groupby(["make", "model_name"])
        .size()
        .reset_index(name="n")
    )
    frequent = frequent[frequent["n"] >= RARE_MIN_COUNT]
    models_by_make: dict[str, list[str]] = {}
    for make, part in frequent.groupby("make"):
        models_by_make[str(make)] = sorted(part["model_name"].astype(str).tolist())
    all_makes = sorted(featured.iloc[train_idx]["make"].astype(str).unique().tolist())

    medians = {col: float(X_train[col].median()) for col in NUMERIC_FEATURES}
    modes = {col: str(X_train[col].mode().iloc[0]) for col in CATEGORICAL_FEATURES}
    bundle = {
        "pipeline": winner,
        "q_low": q_low,
        "q_high": q_high,
        "medians": medians,
        "modes": modes,
        "models_by_make": models_by_make,
        "all_makes": all_makes,
        "model_name": winner_name,
    }
    import joblib

    joblib.dump(bundle, MODEL_PATH)

    predictions = pd.DataFrame(
        {
            "actual": actual,
            "predicted": point,
            "low": low,
            "high": high,
            "make": X_test["make"].to_numpy(),
            "model_name": X_test["model_name"].to_numpy(),
            "year_age": X_test["car_age"].to_numpy(),
            "fuel": X_test["fuel"].to_numpy(),
        }
    )
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)

    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "winner": winner_name,
        "features": FEATURE_COLUMNS,
        "target": "log1p(price in lakh PKR)",
        "reference_year": 2024,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cleaning": cleaning_stats,
        "comparison": comparison.to_dict(orient="records"),
        "baseline_test": baseline_test,
        "test": test_metrics,
        "interval": {
            "method": "split conformal on out-of-fold log residuals (10th/90th)",
            "q_low": q_low,
            "q_high": q_high,
            "oof_coverage": oof_coverage,
            "test": interval,
        },
        "deal_labels_oof": oof_labels.value_counts().to_dict(),
        "perturbation": perturbation,
        "permutation_importance": importance_rows,
        "versions": _library_versions(),
        "best_params": next(
            row["best_params"] for row in rows if row["model"] == winner_name
        ),
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2))
    (REPORTS / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {"winner": winner_name, "test": test_metrics, "interval": interval},
            indent=2,
        )
    )
    return payload


def _top_makes(makes: pd.Series) -> pd.Series:
    top = set(makes.value_counts().head(8).index)
    return makes.where(makes.isin(top), "Other")


def _try_shap(pipeline: Pipeline, sample: pd.DataFrame, path) -> None:
    try:
        import shap
    except Exception:
        return
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    if not hasattr(model, "predict"):
        return
    try:
        transformed = prep.transform(sample)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        values = np.asarray(values)
        if values.ndim != 2:
            return
        names = [str(name) for name in prep.get_feature_names_out()]
        mean_abs = np.mean(np.abs(values), axis=0)
        order = np.argsort(mean_abs)[::-1][:15]
        plot_importance(
            [names[i] for i in order],
            mean_abs[order],
            path,
            xlabel="mean |SHAP value| on log price",
        )
    except Exception as exc:
        print(f"shap skipped: {exc}")


if __name__ == "__main__":
    train()
