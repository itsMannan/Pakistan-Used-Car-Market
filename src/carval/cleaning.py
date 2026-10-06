"""Turn the raw PakWheels scrape into a modelling table.

The price column is the awkward part. PakWheels shows lakh until a car crosses
1 crore, then shows crore, and the scrape kept whichever number was on screen.
A 2018 Fortuner at 1.20 is 120 lakh, not 1.2 lakh. A Mehran at 2 lakh is real.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from carval.config import (
    ANCHOR_LAKH,
    CRORE_READING_MAX,
    MAX_YEAR,
    MIN_YEAR,
    MULTI_WORD_MAKES,
    PREMIUM_MAKES,
    RAW_CSV,
    REFERENCE_YEAR,
)

_GEN_RE = re.compile(
    r"\b\d+(?:st|nd|rd|th)\s+(?:\([^)]*\)\s+)?Generation(?:\s+\([^)]*\))?",
    re.IGNORECASE,
)
_NICKNAME_RE = re.compile(r"\b(Reborn|Rebirth|Eagle Eye)\b", re.IGNORECASE)
_FOR_SALE_RE = re.compile(r"\s+for Sale\s*$", re.IGNORECASE)


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop = [c for c in df.columns if c.startswith("Unnamed")]
    return df.drop(columns=drop)


def parse_title(title: str, year: int) -> tuple[str, str, str] | None:
    text = _FOR_SALE_RE.sub("", str(title)).strip()
    token = f" {int(year)} "
    idx = text.rfind(token)
    if idx == -1:
        match = re.search(rf"\b{int(year)}\b", text)
        if match is None:
            return None
        before = text[: match.start()].strip()
        variant = text[match.end() :].strip()
    else:
        before = text[:idx].strip()
        variant = text[idx + len(token) :].strip()

    before = _GEN_RE.sub(" ", before)
    before = re.sub(r"\s+", " ", before).strip(" -")
    if not before:
        return None

    make = None
    model = ""
    for prefix in MULTI_WORD_MAKES:
        if before == prefix or before.startswith(prefix + " "):
            make = prefix
            model = before[len(prefix) :].strip()
            break
    if make is None:
        parts = before.split(" ", 1)
        make = parts[0]
        model = parts[1] if len(parts) > 1 else ""

    model = _NICKNAME_RE.sub(" ", model)
    model = re.sub(r"\s+", " ", model).strip()
    variant = re.sub(r"\s+", " ", variant).strip()
    if not model:
        return None
    return make, model, variant


def parse_km(values: pd.Series) -> pd.Series:
    text = (
        values.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("km", "", case=False, regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce")


def parse_powertrain(
    cc_values: pd.Series, fuel: pd.Series
) -> tuple[pd.Series, pd.Series]:
    text = cc_values.astype(str)
    number = pd.to_numeric(text.str.extract(r"([\d.]+)", expand=False), errors="coerce")
    is_kwh = text.str.contains("kwh", case=False, na=False)
    battery = number.where(is_kwh)
    # Real packs in this market are roughly 10-150 kWh. Larger figures are junk.
    battery = battery.where((battery >= 10) & (battery <= 150))
    engine = number.where(~is_kwh)
    engine = engine.where((engine >= 500) & (engine <= 8000))
    engine = engine.mask(fuel.eq("Electric") | is_kwh)
    return engine, battery


def clean_km(km: pd.Series, year: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Blank odometer readings that cannot belong to a car of that age."""
    age = REFERENCE_YEAR - year
    per_year = km / np.maximum(age, 1)
    implausible = km.notna() & (
        (km > 500_000) | ((age >= 3) & (km < 500)) | ((age >= 1) & (per_year > 100_000))
    )
    return km.mask(implausible), implausible


def _anchor_predictor(anchor: dict[int, float]):
    if not anchor:
        return lambda year: None
    years = np.array(sorted(anchor), dtype=float)
    medians = np.array([anchor[int(y)] for y in years], dtype=float)
    if len(years) >= 3:
        slope, intercept = np.polyfit(years, np.log(medians), 1)
    else:
        slope, intercept = 0.0, float(np.log(np.median(medians)))

    def predict(year: int) -> float | None:
        if year in anchor:
            return float(anchor[year])
        nearest = int(years[int(np.argmin(np.abs(years - year)))])
        if abs(nearest - year) > 8:
            return None
        extrapolated = float(np.exp(intercept + slope * year))
        base = anchor[nearest]
        steps = abs(year - nearest)
        ceiling = base * (1.12**steps)
        floor = base * (0.90**steps)
        return float(np.clip(extrapolated, floor, ceiling))

    return predict


def _mostly_crore(prices: pd.Series, years: pd.Series, make: str) -> bool:
    if len(prices) < 3:
        return False
    median_price = float(prices.median())
    p90 = float(prices.quantile(0.9))
    median_year = float(years.median())
    if median_price > 10 or p90 > 18:
        return False
    if make in PREMIUM_MAKES and median_year >= 2000:
        return True
    if len(prices) >= 8 and median_year >= 2016 and median_price <= 6 and p90 <= 12:
        return True
    return False


def _as_crore_plausible(price: float, ref: float, wide: bool = False) -> bool:
    as_crore = price * 100.0
    if price > CRORE_READING_MAX or as_crore < 90 or ref < 55:
        return False
    # wide covers a new generation that jumps well past the last converted year
    # (a 2014 7 Series near 1.8 crore, a 2022 one near 8).
    if wide and ref >= 80:
        upper = max(ref * 4.0, 400.0)
    elif ref >= 80:
        upper = max(ref * 1.9, 220.0)
    else:
        upper = ref * 1.9
    if not (ref * 0.65 <= as_crore <= upper):
        return False
    crore_gap = abs(np.log(as_crore / ref))
    lakh_gap = abs(np.log(max(price, 0.1) / ref))
    return crore_gap < lakh_gap - np.log(1.25)


def _fix_group(g: pd.DataFrame, make: str) -> tuple[pd.Series, pd.Series]:
    units = pd.Series("lakh", index=g.index)
    values = g["price"].astype(float).copy()
    if _mostly_crore(g["price"], g["year"], make):
        for idx, row in g.iterrows():
            price = float(row["price"])
            if price <= CRORE_READING_MAX and int(row["year"]) >= 1998:
                units.loc[idx] = "crore"
                values.loc[idx] = price * 100.0
        return units, values

    anchor: dict[int, float] = {}
    for year, gy in g.groupby("year"):
        high = gy.loc[gy["price"] >= ANCHOR_LAKH, "price"]
        if len(high) >= 3:
            anchor[int(year)] = float(high.median())
    predict = _anchor_predictor(anchor)

    last_level = None
    crore_mode = False
    group_median = float(g["price"].median())

    for year in sorted(g["year"].unique()):
        year = int(year)
        rows = g[g["year"] == year]
        ref = last_level if crore_mode and last_level is not None else predict(year)
        if ref is None:
            ref = last_level
        year_prices = rows["price"]
        uniform_small = (
            float(year_prices.median()) <= 12
            and float(year_prices.quantile(0.9)) <= 15
            and float((year_prices <= CRORE_READING_MAX).mean()) >= 0.8
        )
        wide = bool(
            uniform_small and year >= 2012 and (crore_mode or make in PREMIUM_MAKES)
        )
        if wide and ref is None and make in PREMIUM_MAKES:
            ref = float(year_prices.median()) * 100.0
        converted = []
        for idx, row in rows.iterrows():
            price = float(row["price"])
            if price >= ANCHOR_LAKH:
                if ref is None:
                    ref = price
                continue
            use_ref = ref
            if use_ref is not None and _as_crore_plausible(price, use_ref, wide=wide):
                units.loc[idx] = "crore"
                values.loc[idx] = price * 100.0
                converted.append(price * 100.0)
                continue
            if (
                use_ref is None
                and make in PREMIUM_MAKES
                and year >= 2005
                and price <= 12
            ):
                units.loc[idx] = "crore"
                values.loc[idx] = price * 100.0
                converted.append(price * 100.0)
                continue
            too_cheap = (
                group_median >= 18
                and use_ref is not None
                and use_ref >= 30
                and year >= 2005
                and price < min(4.0, use_ref * 0.12)
            )
            between = (
                crore_mode
                and use_ref is not None
                and use_ref >= 120
                and 12 < price < ANCHOR_LAKH
            )
            if too_cheap or between:
                units.loc[idx] = "unresolvable"

        if len(converted) >= max(3, 0.5 * len(rows)):
            last_level = float(np.median(converted))
            crore_mode = True
        else:
            high = rows.loc[rows["price"] >= ANCHOR_LAKH, "price"]
            if len(high) >= 3:
                last_level = float(high.median())
                crore_mode = False
            elif converted:
                last_level = float(np.median(converted))

    return units, values


def fix_price_units(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_lakh and price_unit. price_unit is lakh, crore, or unresolvable."""
    out = df.copy()
    units = pd.Series("lakh", index=out.index, dtype=object)
    values = out["price"].astype(float).copy()
    grouped = out.groupby(["make", "model_name"], sort=False)
    for (make, _), idx in grouped.groups.items():
        unit, value = _fix_group(out.loc[idx], make)
        units.loc[idx] = unit.to_numpy()
        values.loc[idx] = value.to_numpy()
    out["price_unit"] = units
    out["price_lakh"] = values
    return out


def clean_listings(
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_raw() if df is None else df.copy()
    stats: dict[str, Any] = {"rows_raw": int(len(raw))}

    priced = raw.dropna(subset=["prices"]).copy()
    stats["dropped_missing_price"] = int(len(raw) - len(priced))

    subset = [c for c in priced.columns if not str(c).startswith("Unnamed")]
    before_dupes = len(priced)
    priced = priced.drop_duplicates(subset=subset, keep="first")
    stats["dropped_duplicates"] = int(before_dupes - len(priced))

    parsed = [
        parse_title(title, year)
        for title, year in zip(priced["Title"], priced["Model"])
    ]
    priced = priced.copy()
    priced["make"] = [item[0] if item else None for item in parsed]
    priced["model_name"] = [item[1] if item else None for item in parsed]
    priced["variant"] = [item[2] if item else "" for item in parsed]
    parsed_ok = priced["make"].notna()
    stats["dropped_unparsed_title"] = int((~parsed_ok).sum())
    priced = priced.loc[parsed_ok].copy()

    priced["year"] = priced["Model"].astype(int)
    stats["dropped_year_out_of_range"] = int(
        (~priced["year"].between(MIN_YEAR, MAX_YEAR)).sum()
    )
    priced = priced.loc[priced["year"].between(MIN_YEAR, MAX_YEAR)].copy()

    priced["price"] = priced["prices"].astype(float)
    fixed = fix_price_units(priced)
    stats["converted_crore_to_lakh"] = int((fixed["price_unit"] == "crore").sum())
    stats["dropped_unresolvable_price"] = int(
        (fixed["price_unit"] == "unresolvable").sum()
    )
    fixed = fixed.loc[fixed["price_unit"] != "unresolvable"].copy()

    fixed["fuel"] = fixed["Engine_type"].astype(str).str.strip()
    fixed["transmission"] = fixed["Transmission"].astype(str).str.strip()
    engine, battery = parse_powertrain(fixed["CC"], fixed["fuel"])
    km, implausible = clean_km(parse_km(fixed["Km_Driven"]), fixed["year"])
    fixed["engine_cc"] = engine.to_numpy()
    fixed["battery_kwh"] = battery.to_numpy()
    fixed["km"] = km.to_numpy()
    stats["km_set_missing"] = int(implausible.sum())
    stats["rows_engine_cc_missing"] = int(fixed["engine_cc"].isna().sum())
    stats["rows_with_battery_kwh"] = int(fixed["battery_kwh"].notna().sum())
    stats["rows_clean"] = int(len(fixed))
    stats["rows_dropped_total"] = int(stats["rows_raw"] - stats["rows_clean"])
    kept = stats["rows_clean"] / stats["rows_raw"]
    stats["share_kept"] = round(float(kept), 4)

    columns = [
        "make",
        "model_name",
        "variant",
        "year",
        "fuel",
        "transmission",
        "engine_cc",
        "battery_kwh",
        "km",
        "price",
        "price_lakh",
        "price_unit",
    ]
    return fixed[columns].reset_index(drop=True), stats
