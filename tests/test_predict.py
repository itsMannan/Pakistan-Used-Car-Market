import pytest

from carval.config import MODEL_PATH
from carval.predict import predict_car


def test_year_out_of_range():
    with pytest.raises(ValueError, match="year"):
        predict_car("Toyota", "Corolla", "", 1970, 1300, "Manual", "Petrol", 10000)


def test_bad_fuel_and_engine():
    with pytest.raises(ValueError, match="fuel_type"):
        predict_car("Toyota", "Corolla", "", 2018, 1300, "Manual", "Gas", 10000)
    with pytest.raises(ValueError, match="engine_cc"):
        predict_car("Toyota", "Corolla", "", 2018, 50, "Manual", "Petrol", 10000)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_predict_car_output_shape():
    result = predict_car(
        "toyota",
        "corolla",
        "GLi",
        2018,
        1300,
        "Automatic",
        "Petrol",
        80000,
        asking_price=40,
    )
    assert set(result) >= {
        "estimated_price",
        "price_low",
        "price_high",
        "deal_score",
        "verdict",
    }
    assert result["price_low"] <= result["estimated_price"] <= result["price_high"]
    assert result["verdict"] in {"UNDERPRICED", "FAIR", "OVERPRICED"}
    assert 0 <= result["deal_score"] <= 100
    assert result["warning"] is None


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_unknown_model_warns_instead_of_failing():
    result = predict_car(
        "Toyota",
        "Not A Real Model",
        "",
        2016,
        1500,
        "Manual",
        "Petrol",
        50000,
        asking_price=30,
    )
    assert result["warning"]
    assert result["estimated_price"] > 0
    # A recent large Toyota with an unknown name should follow other Toyotas,
    # not the pooled rare-model bucket.
    large = predict_car(
        "Toyota",
        "Not A Real Model",
        "",
        2022,
        2700,
        "Automatic",
        "Diesel",
        40000,
        asking_price=180,
    )
    assert large["estimated_price"] > 80
    assert large["price_low"] <= large["estimated_price"] <= large["price_high"]
