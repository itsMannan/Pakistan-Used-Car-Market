from __future__ import annotations

import numpy as np

# Steepness of the logistic. 6 puts a 25% discount near 80 and a 25% premium near 20.
DEAL_SCORE_K = 6.0
UNDER_RATIO = 0.90
OVER_RATIO = 1.10


def deal_score(ratio: float) -> float:
    """Map asking/estimate to 0-100. 100 is the best deal for the buyer."""
    score = 100.0 / (1.0 + np.exp(DEAL_SCORE_K * (float(ratio) - 1.0)))
    return float(np.clip(score, 0.0, 100.0))


def verdict(asking: float, predicted: float, low: float, high: float) -> str:
    """Label a listing from the interval and from the asking/estimate ratio.

    UNDERPRICED if the ask is below the range or under 90% of the estimate.
    OVERPRICED if it is above the range or over 110% of the estimate.
    FAIR otherwise. There is no human label for this; it is relative to the model.
    """
    if predicted <= 0:
        raise ValueError("predicted price must be positive")
    ratio = float(asking) / float(predicted)
    if asking < low or ratio < UNDER_RATIO:
        return "UNDERPRICED"
    if asking > high or ratio > OVER_RATIO:
        return "OVERPRICED"
    return "FAIR"
