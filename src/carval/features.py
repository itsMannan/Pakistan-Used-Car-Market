from __future__ import annotations

import re

import numpy as np
import pandas as pd

from carval.config import REFERENCE_YEAR

_HIGH = {
    "oriel",
    "grande",
    "legender",
    "autobiography",
    "vx",
    "tz",
    "rs",
    "altis",
    "lumiere",
    "z",
}
_MID = {"gli", "vxl", "aspire", "prosmatec", "cvt", "g", "hybrid", "fx"}


def variant_tier(variant: str) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", str(variant).lower()))
    if not tokens:
        return "unknown"
    if tokens & _HIGH:
        return "high"
    if tokens & _MID:
        return "mid"
    return "base"


def add_features(
    df: pd.DataFrame, reference_year: int = REFERENCE_YEAR
) -> pd.DataFrame:
    out = df.copy()
    out["car_age"] = reference_year - out["year"].astype(int)
    age = np.maximum(out["car_age"].to_numpy(), 1)
    km = out["km"].astype(float)
    out["km_per_year"] = km / age
    out["log_km"] = np.log1p(km)
    out["is_hybrid"] = (out["fuel"] == "Hybrid").astype(int)
    out["is_electric"] = (out["fuel"] == "Electric").astype(int)
    out["transmission_auto"] = (out["transmission"] == "Automatic").astype(int)
    # Non-EVs do not have a pack. Zero keeps the median imputer from painting
    # an EV battery size onto every petrol car.
    out["battery_kwh"] = out["battery_kwh"].astype(float).fillna(0.0)
    out["variant_tier"] = out["variant"].map(variant_tier)
    return out
