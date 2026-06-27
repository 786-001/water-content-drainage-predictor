from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Water Content Drainage Predictor",
    page_icon="💧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f7f5; color: #17231f; }
    [data-testid="stHeader"] { background: rgba(255,255,255,.92); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #d9e0dc; }
    .brand { color:#196b4b; font-size:.76rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .main-title { margin:.15rem 0 .25rem; font-size:2.4rem; line-height:1.08; font-weight:800; }
    .subtitle { color:#66736e; margin-bottom:1.4rem; }
    .result-box { background:#fff; border:1px solid #d9e0dc; border-top:3px solid #196b4b; border-radius:7px; padding:1.1rem 1.25rem; }
    .result-label { color:#196b4b; font-size:.7rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .result-rate { font-size:2.8rem; font-weight:800; line-height:1.05; }
    .result-unit { color:#66736e; font-size:.95rem; font-weight:600; }
    .status { display:inline-block; margin-top:.65rem; padding:.3rem .55rem; border-radius:999px; background:#e6f3ec; color:#0f5138; font-size:.75rem; font-weight:800; }
    .coverage-warning { padding:.7rem .8rem; border-left:3px solid #a36516; background:#fbf0dc; color:#76511d; font-size:.82rem; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #d9e0dc; border-radius:7px; padding:.75rem 1rem; }
    .small-note { color:#7c8883; font-size:.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets() -> tuple[dict, dict]:
    bundle = joblib.load(ROOT / "model.joblib")
    metadata = json.loads((ROOT / "model_metadata.json").read_text(encoding="utf-8"))
    return bundle, metadata


BUNDLE, METADATA = load_assets()
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["feature_columns"]


def model_input(fine: float, wg: int, slope: float, time_hours: float, water_percent: float) -> pd.DataFrame:
    return pd.DataFrame(
        [[fine, wg, slope, time_hours, water_percent / 100.0]],
        columns=FEATURES,
    )


def predict(fine: float, wg: int, slope: float, time_hours: float, water_percent: float) -> tuple[float, np.ndarray]:
    frame = model_input(fine, wg, slope, time_hours, water_percent)
    rate = float(MODEL.predict(frame)[0])
    tree_rates = np.array([float(tree.predict(frame.values)[0]) for tree in MODEL.estimators_])
    return rate, tree_rates


def status_for(rate: float) -> str:
    if rate < METADATA["target_quantiles"]["0.1"]:
        return "Rapid drainage"
    if rate < -1e-8:
        return "Draining"
    if rate > 1e-8:
        return "Wetting trend"
    return "Stable"


def projection(fine: float, wg: int, slope: float, time_hours: float, water_percent: float) -> pd.DataFrame:
    current_fraction = water_percent / 100.0
    maximum_time = METADATA["ranges"]["Time (hours)"]["max"]
    records = [{"Hour": 0, "Water content (%)": water_percent}]
    for offset in range(1, 13):
        frame = model_input(
            fine,
            wg,
            slope,
            min(time_hours + offset - 1, maximum_time),
            current_fraction * 100,
        )
        current_fraction = float(np.clip(current_fraction + MODEL.predict(frame)[0], 0, 1))
        records.append({"Hour": offset, "Water content (%)": current_fraction * 100})
    return pd.DataFrame(records).set_index("Hour")


st.markdown('<div class="brand">Sensor decision support</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Water Content Drainage Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Estimate water-content change rate from field conditions and a current moisture reading.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Site conditions")
    fine = st.number_input("Fine content (%)", min_value=0.0, value=6.0, step=0.1)
    slope = st.number_input("Cross slope (%)", min_value=0.0, value=2.0, step=0.1)
    time_hours = st.number_input("Elapsed time (hours)", min_value=1.0, value=72.0, step=1.0)
    water_percent = st.number_input("Water content (%)", min_value=0.0, value=17.0, step=0.01)
    geotextile = st.toggle("Wicking geotextile", value=True)
    st.caption("Capillary drainage geotextile installed")

    ranges = METADATA["ranges"]
    outside = []
    if not ranges["fine"]["min"] <= fine <= ranges["fine"]["max"]:
        outside.append("fine content")
    if not ranges["slope"]["min"] <= slope <= ranges["slope"]["max"]:
        outside.append("cross slope")
    if not ranges["Time (hours)"]["min"] <= time_hours <= ranges["Time (hours)"]["max"]:
        outside.append("elapsed time")
    water_fraction = water_percent / 100.0
    if not ranges["w%"]["min"] <= water_fraction <= ranges["w%"]["max"]:
        outside.append("water content")

    if outside:
        st.markdown(
            f'<div class="coverage-warning">{", ".join(outside).capitalize()} outside trained coverage; interpret cautiously.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("Inputs are within the measured experimental range.")


wg = 1 if geotextile else 0
raw_rate, tree_rates = predict(fine, wg, slope, time_hours, water_percent)
rate_pph = raw_rate * 100
lower, upper = np.quantile(tree_rates, [0.1, 0.9]) * 100
next_hour = float(np.clip(water_percent + rate_pph, 0, 100))
status = status_for(raw_rate)

st.markdown(
    f"""
    <div class="result-box">
      <div class="result-label">Predicted water-content change rate</div>
      <div><span class="result-rate">{rate_pph:.4f}</span> <span class="result-unit">percentage points / hour</span></div>
      <span class="status">{status}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Next-hour water content", f"{next_hour:.2f}%")
metric_2.metric("Tree-model spread", f"{lower:.4f} to {upper:.4f}")

within_bounds = not outside
exact_levels = fine in METADATA["unique_values"]["fine"] and slope in METADATA["unique_values"]["slope"]
coverage = "Exact design level" if within_bounds and exact_levels else "Custom value" if within_bounds else "Outside trained range"
metric_3.metric("Training coverage", coverage)

left, right = st.columns([1.45, 1])
with left:
    st.subheader("12-hour projected water content")
    st.line_chart(projection(fine, wg, slope, time_hours, water_percent), color="#196b4b")
    st.markdown('<div class="small-note">Recursive hourly estimate; use as directional guidance.</div>', unsafe_allow_html=True)

with right:
    st.subheader("Cross-slope comparison")
    scenario_records = []
    for scenario_slope in METADATA["unique_values"]["slope"]:
        scenario_rate, _ = predict(fine, wg, scenario_slope, time_hours, water_percent)
        scenario_records.append(
            {"Cross slope": f"{scenario_slope:g}%", "Rate (pp/h)": scenario_rate * 100}
        )
    scenario_data = pd.DataFrame(scenario_records).set_index("Cross slope")
    st.bar_chart(scenario_data, color="#2f7d78")
    best = min(scenario_records, key=lambda item: item["Rate (pp/h)"])
    st.caption(f"Most favorable modeled rate among measured levels: {best['Cross slope']}.")

st.subheader("Model feature influence")
importance = pd.DataFrame(METADATA["feature_importance"])
importance["feature"] = importance["feature"].replace(
    {
        "w%": "Water content",
        "Time (hours)": "Elapsed time",
        "fine": "Fine content",
        "slope": "Cross slope",
        "wg": "Wicking geotextile",
    }
)
st.bar_chart(importance.set_index("feature"), horizontal=True, color="#4776a6")

st.caption(
    f"Random forest: {METADATA['estimators']} trees, {METADATA['sample_count']:,} observations, "
    f"holdout R² {METADATA['metrics']['r2']:.3f}. Validate predictions with field measurements before operational use."
)
