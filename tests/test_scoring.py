import pytest

from carval.scoring import deal_score, verdict


def test_deal_score_decreases_as_asking_price_rises():
    ratios = [0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6]
    scores = [deal_score(ratio) for ratio in ratios]
    assert scores[0] > 80
    assert scores[-1] < 20
    assert scores == sorted(scores, reverse=True)
    assert deal_score(1.0) == pytest.approx(50.0)


def test_verdict_thresholds():
    assert verdict(80, 100, 85, 115) == "UNDERPRICED"
    assert verdict(100, 100, 85, 115) == "FAIR"
    assert verdict(120, 100, 85, 115) == "OVERPRICED"
    # Inside a wide range, the 10% ratio rule still marks a cheap ask.
    assert verdict(88, 100, 70, 140) == "UNDERPRICED"
    assert verdict(112, 100, 70, 140) == "OVERPRICED"
    assert verdict(95, 100, 70, 140) == "FAIR"
