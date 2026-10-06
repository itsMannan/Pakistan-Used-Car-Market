import pandas as pd
import pytest

from carval.cleaning import clean_km, fix_price_units, parse_km, parse_title


def test_parse_title_strips_sale_and_generation():
    parsed = parse_title("Honda City 5th Generation 2018 Aspire 1.5 for Sale", 2018)
    assert parsed == ("Honda", "City", "Aspire 1.5")


def test_parse_title_civic_nickname_and_mercedes():
    civic = parse_title("Honda Civic Reborn 2012 VTi Oriel 1.8 i-VTEC for Sale", 2012)
    assert civic[0] == "Honda"
    assert civic[1] == "Civic"
    assert "Oriel" in civic[2]
    wagon = parse_title("Suzuki Wagon R  2020 Hybrid FX for Sale", 2020)
    assert wagon == ("Suzuki", "Wagon R", "Hybrid FX")
    benz = parse_title("Mercedes Benz C Class  2001 C180 for Sale", 2001)
    assert benz[0] == "Mercedes Benz"
    assert benz[1] == "C Class"


def test_parse_km_removes_commas_and_unit():
    parsed = parse_km(pd.Series(["26,755 km", "1000 km"]))
    assert parsed.tolist() == [26755, 1000]


def test_clean_km_drops_placeholders_and_keeps_new_cars():
    km = pd.Series([1.0, 120.0, 900_000.0, 80_000.0])
    year = pd.Series([2010, 2024, 2015, 2016])
    cleaned, flagged = clean_km(km, year)
    assert flagged.tolist() == [True, False, True, False]
    assert pd.isna(cleaned.iloc[0])
    assert cleaned.iloc[1] == 120


def _frame(rows):
    return pd.DataFrame(rows)


def test_crore_fix_on_fortuner_like_prices():
    rows = []
    for year, prices in {
        2014: [72, 75, 78, 80],
        2015: [76, 80, 84, 88],
        2016: [82, 86, 90, 92],
        2018: [1.15, 1.22, 1.28, 1.34],
        2019: [1.30, 1.38, 1.42, 1.50],
    }.items():
        for price in prices:
            rows.append(
                {
                    "make": "Toyota",
                    "model_name": "Fortuner",
                    "year": year,
                    "price": price,
                }
            )
    fixed = fix_price_units(_frame(rows))
    older = fixed[fixed["year"] == 2015]
    newer = fixed[fixed["year"] == 2018]
    assert (older["price_unit"] == "lakh").all()
    assert older["price_lakh"].median() == older["price"].median()
    assert (newer["price_unit"] == "crore").all()
    assert newer["price_lakh"].median() == pytest.approx(newer["price"].median() * 100)


def test_mehran_prices_stay_in_lakh():
    rows = []
    for year in range(2008, 2017):
        for price in (6.5, 8.0, 9.5, 11.0):
            rows.append(
                {"make": "Suzuki", "model_name": "Mehran", "year": year, "price": price}
            )
    fixed = fix_price_units(_frame(rows))
    assert (fixed["price_unit"] == "lakh").all()
    assert fixed["price_lakh"].tolist() == fixed["price"].tolist()
