from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from carval.config import (
    FEATURE_COLUMNS,
    FUEL_TYPES,
    MAX_YEAR,
    METADATA_PATH,
    MIN_YEAR,
    MODEL_PATH,
    TRANSMISSIONS,
)
from carval.evaluate import apply_log_interval
from carval.features import add_features
from carval.scoring import deal_score, verdict

_FACTOR_LABELS = {
    "car_age": "Age",
    "log_km": "Mileage",
    "km_per_year": "Annual mileage",
    "engine_cc": "Engine size",
    "battery_kwh": "Battery size",
    "is_hybrid": "Hybrid",
    "is_electric": "Electric",
    "transmission_auto": "Transmission",
    "make": "Make",
    "model_name": "Model",
    "fuel": "Fuel",
    "variant_tier": "Variant",
}


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run training first."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


def predict_car(
    make: str,
    model: str,
    variant: str,
    year: int,
    engine_cc: float | None,
    transmission: str,
    fuel_type: str,
    km_driven: float,
    asking_price: float | None = None,
) -> dict:
    """Estimate market value and, if an asking price is given, score the deal."""
    year = int(year)
    if year < MIN_YEAR or year > MAX_YEAR:
        raise ValueError(f"year must be between {MIN_YEAR} and {MAX_YEAR}")
    if transmission not in TRANSMISSIONS:
        raise ValueError(f"transmission must be one of {', '.join(TRANSMISSIONS)}")
    if fuel_type not in FUEL_TYPES:
        raise ValueError(f"fuel_type must be one of {', '.join(FUEL_TYPES)}")
    if km_driven is None or float(km_driven) < 0:
        raise ValueError("km_driven must be zero or positive")
    if asking_price is not None and float(asking_price) <= 0:
        raise ValueError("asking_price must be positive")
    engine, battery = _power(fuel_type, engine_cc)

    bundle = load_bundle()
    canonical_make, canonical_model, warning = _resolve_names(bundle, make, model)
    peers = bundle["models_by_make"].get(canonical_make, [])
    base = {
        "make": canonical_make,
        "variant": variant or "",
        "year": year,
        "fuel": fuel_type,
        "transmission": transmission,
        "engine_cc": engine,
        "battery_kwh": battery,
        "km": float(km_driven),
    }
    if warning and peers:
        # An unseen name would be lumped in with every rare model. Price it as
        # a typical car of this make instead.
        estimated, price_low, price_high, featured = _median_of_make(
            bundle, base, peers
        )
        warning = (
            f"'{model}' is not a common {canonical_make} model in the training data. "
            f"The estimate is the median across common {canonical_make} models."
        )
    else:
        row = pd.DataFrame([{**base, "model_name": canonical_model}])
        featured = add_features(row)
        estimated, price_low, price_high = _predict_one(bundle, featured)
    result = {
        "estimated_price": round(estimated, 2),
        "price_low": round(price_low, 2),
        "price_high": round(price_high, 2),
        "deal_score": None,
        "verdict": None,
        "warning": warning,
        "factors": local_factors(bundle, featured),
    }
    if asking_price is not None:
        ask = float(asking_price)
        result["deal_score"] = round(deal_score(ask / estimated), 1)
        result["verdict"] = verdict(ask, estimated, price_low, price_high)
    return result


def _predict_one(bundle: dict, featured: pd.DataFrame) -> tuple[float, float, float]:
    log_pred = float(bundle["pipeline"].predict(featured[FEATURE_COLUMNS])[0])
    point, low, high = apply_log_interval([log_pred], bundle["q_low"], bundle["q_high"])
    return float(point[0]), float(low[0]), float(high[0])


def _median_of_make(
    bundle: dict, base: dict, models: list[str]
) -> tuple[float, float, float, pd.DataFrame]:
    scored = []
    for model_name in models:
        featured = add_features(pd.DataFrame([{**base, "model_name": model_name}]))
        estimated, low, high = _predict_one(bundle, featured)
        scored.append((estimated, low, high, featured))
    scored.sort(key=lambda item: item[0])
    return scored[len(scored) // 2]


def local_factors(bundle: dict, featured: pd.DataFrame, top_n: int = 4) -> list[str]:
    try:
        phrases = _tree_factors(bundle, featured, top_n)
    except Exception:
        phrases = _ablation_factors(bundle, featured, top_n)
    return phrases


def _phrases(effects: list[tuple[str, float]], top_n: int, cutoff: float) -> list[str]:
    effects = sorted(effects, key=lambda item: abs(item[1]), reverse=True)
    phrases = []
    for column, delta in effects:
        if abs(delta) < cutoff:
            continue
        label = _FACTOR_LABELS.get(column, column)
        direction = "raises" if delta > 0 else "lowers"
        phrases.append(f"{label} {direction} the estimate")
        if len(phrases) == top_n:
            break
    return phrases


def _tree_factors(bundle: dict, featured: pd.DataFrame, top_n: int) -> list[str]:
    import xgboost as xgb

    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    transformed = prep.transform(featured[FEATURE_COLUMNS])
    names = list(prep.get_feature_names_out())
    contrib = model.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[
        0
    ]
    effects = []
    for idx, name in enumerate(names):
        if name.startswith("num__"):
            effects.append((name.split("__", 1)[1], float(contrib[idx])))
        elif name.startswith("cat__") and transformed[0, idx] > 0.5:
            rest = name[5:]
            for column in ("model_name", "variant_tier", "make", "fuel"):
                if rest.startswith(column + "_"):
                    effects.append((column, float(contrib[idx])))
                    break
    return _phrases(effects, top_n, cutoff=0.02)


def _ablation_factors(bundle: dict, featured: pd.DataFrame, top_n: int) -> list[str]:
    base = float(bundle["pipeline"].predict(featured[FEATURE_COLUMNS])[0])
    effects = []
    for column, typical in bundle["medians"].items():
        alt = featured.copy()
        alt[column] = typical
        delta = base - float(bundle["pipeline"].predict(alt[FEATURE_COLUMNS])[0])
        effects.append((column, delta))
    for column, typical in bundle["modes"].items():
        alt = featured.copy()
        alt[column] = typical
        delta = base - float(bundle["pipeline"].predict(alt[FEATURE_COLUMNS])[0])
        effects.append((column, delta))
    return _phrases(effects, top_n, cutoff=0.015)


def _resolve_names(bundle: dict, make: str, model: str) -> tuple[str, str, str | None]:
    models_by_make: dict = bundle["models_by_make"]
    known = {name.casefold(): name for name in bundle.get("all_makes", models_by_make)}
    folded = str(make).strip().casefold()
    if folded not in known:
        known_list = ", ".join(sorted(known.values()))
        raise ValueError(f"Unknown make '{make}'. Known makes include: {known_list}")
    canonical_make = known[folded]
    model_lookup = {
        name.casefold(): name for name in models_by_make.get(canonical_make, [])
    }
    model_folded = str(model).strip().casefold()
    if model_folded in model_lookup:
        return canonical_make, model_lookup[model_folded], None
    warning = (
        f"'{model}' is not a common {canonical_make} model in the training data. "
        f"The estimate falls back to other {canonical_make} cars."
    )
    return canonical_make, str(model).strip(), warning


def _power(fuel_type: str, engine_cc: float | None) -> tuple[float | None, float]:
    if fuel_type == "Electric":
        if engine_cc is not None and 10 <= float(engine_cc) <= 150:
            return None, float(engine_cc)
        return None, 0.0
    if engine_cc is None:
        return None, 0.0
    displacement = float(engine_cc)
    if displacement < 500 or displacement > 8000:
        raise ValueError(
            "engine_cc looks out of range (expected 500 to 8000, or a kWh figure for an electric car)"
        )
    return displacement, 0.0
