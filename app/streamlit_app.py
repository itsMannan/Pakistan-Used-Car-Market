from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from carval.config import FUEL_TYPES, MAX_YEAR, MIN_YEAR, TRANSMISSIONS  # noqa: E402
from carval.predict import load_bundle, predict_car  # noqa: E402

st.set_page_config(page_title="Used car valuation", layout="centered")
st.title("Pakistan used-car valuation")
st.caption("Market value from PakWheels listings, with a range and a deal score.")


def _bar(result: dict, asking: float | None) -> str:
    low = result["price_low"]
    high = result["price_high"]
    est = result["estimated_price"]
    points = [low, high, est]
    if asking is not None:
        points.append(asking)
    span_low = min(points)
    span_high = max(points)
    pad = max((span_high - span_low) * 0.08, 0.5)
    left, right = span_low - pad, span_high + pad

    def pct(value: float) -> float:
        return max(0.0, min(100.0, (value - left) / (right - left) * 100))

    band_left = pct(low)
    band_width = max(pct(high) - band_left, 1)
    estimate = pct(est)
    ask_mark = ""
    if asking is not None:
        ask_mark = (
            f"<div style='position:absolute;left:{pct(asking):.1f}%;top:-6px;"
            f"width:2px;height:28px;background:#8a3b24'></div>"
        )
    return (
        "<div style='position:relative;height:18px;margin:28px 0 8px;background:#eceff1;"
        "border-radius:4px'>"
        f"<div style='position:absolute;left:{band_left:.1f}%;width:{band_width:.1f}%;"
        "top:0;bottom:0;background:#c5d4e8;border-radius:4px'></div>"
        f"<div style='position:absolute;left:{estimate:.1f}%;top:-6px;width:2px;"
        "height:28px;background:#1f4e79'></div>"
        f"{ask_mark}</div>"
        "<div style='font-size:12px;color:#555'>"
        "Blue mark is the estimate, the band is the range, the rust mark is the asking price."
        "</div>"
    )


def _show_result(result: dict, asking: float | None) -> None:
    st.subheader(f"{result['estimated_price']:.1f} lakh")
    st.write(
        f"Likely range: {result['price_low']:.1f} to {result['price_high']:.1f} lakh PKR"
    )
    if result["verdict"]:
        colors = {
            "UNDERPRICED": ("#e8f5e9", "#1b5e20"),
            "FAIR": ("#e8eef5", "#1f4e79"),
            "OVERPRICED": ("#fdecea", "#8a3b24"),
        }
        bg, fg = colors[result["verdict"]]
        st.markdown(
            f"<div style='display:inline-block;padding:6px 12px;border-radius:6px;"
            f"background:{bg};color:{fg};font-weight:600'>{result['verdict']}</div>",
            unsafe_allow_html=True,
        )
        st.write(f"Deal score: {result['deal_score']:.0f} / 100")
        st.markdown(_bar(result, asking), unsafe_allow_html=True)
    if result["warning"]:
        st.warning(result["warning"])
    if result["factors"]:
        st.markdown("**What is moving this estimate**")
        for line in result["factors"]:
            st.write(f"- {line}")


try:
    bundle = load_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

makes = bundle.get("all_makes") or sorted(bundle["models_by_make"])

left, right = st.columns(2)
make = left.selectbox("Make", makes)
model_choices = bundle["models_by_make"].get(make, []) + ["Other / not listed"]
model_choice = right.selectbox("Model", model_choices)
model_name = model_choice
if model_choice == "Other / not listed":
    model_name = st.text_input("Model name")

variant = left.text_input("Variant", placeholder="GLi, Oriel, VXL")
year = right.number_input(
    "Year", min_value=MIN_YEAR, max_value=MAX_YEAR, value=2018, step=1
)
fuel = left.selectbox("Fuel", FUEL_TYPES)
transmission = right.selectbox("Transmission", TRANSMISSIONS)
engine_cc = left.number_input(
    "Engine cc (kWh if electric)", min_value=0.0, value=1300.0, step=50.0
)
km_driven = right.number_input(
    "Kilometres driven", min_value=0.0, value=60000.0, step=1000.0
)
has_ask = st.checkbox("I have an asking price", value=True)
asking = st.number_input("Asking price (lakh PKR)", min_value=0.0, value=35.0, step=0.5)

if st.button("Estimate"):
    if not str(model_name).strip():
        st.error("Enter a model name.")
    else:
        try:
            result = predict_car(
                make=make,
                model=str(model_name).strip(),
                variant=variant,
                year=int(year),
                engine_cc=None if engine_cc == 0 else float(engine_cc),
                transmission=transmission,
                fuel_type=fuel,
                km_driven=float(km_driven),
                asking_price=float(asking) if has_ask and asking > 0 else None,
            )
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
        else:
            _show_result(result, float(asking) if has_ask and asking > 0 else None)
