import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path(__file__).parent / "model"

# ------------------------------------------------------------------
# Styling (theme-safe: explicit text colors on all custom backgrounds)
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--text-color, inherit);
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #8b93a3;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fb;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
        color: #1a1a2e;
    }
    .metric-card b { color: #1a1a2e; }
    .result-ok {
        background: linear-gradient(135deg, #e8f9f0 0%, #d4f5e0 100%);
        border-left: 6px solid #2ecc71;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        color: #14532d;
    }
    .result-ok h3, .result-ok p, .result-ok b { color: #14532d; }
    .result-warn {
        background: linear-gradient(135deg, #fff8e6 0%, #fdf0c8 100%);
        border-left: 6px solid #f39c12;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        color: #7a4a00;
    }
    .result-warn h3, .result-warn p, .result-warn b { color: #7a4a00; }
    .result-danger {
        background: linear-gradient(135deg, #fdecec 0%, #fbdada 100%);
        border-left: 6px solid #e74c3c;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        color: #7f1d1d;
    }
    .result-danger h3, .result-danger p, .result-danger b { color: #7f1d1d; }
    .footer-note {
        color: #9ca3af;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 3rem;
    }
    div.stButton > button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #16213e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load Model Artifacts (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "failure_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    type_encoder = joblib.load(MODEL_DIR / "type_encoder.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    feature_stats = joblib.load(MODEL_DIR / "feature_stats.pkl")
    return model, scaler, type_encoder, feature_names, feature_stats


try:
    model, scaler, type_encoder, feature_names, feature_stats = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<p class="main-header">Machine Failure Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predictive maintenance for industrial machinery — trained on the '
    'AI4I 2020 Predictive Maintenance Dataset (UCI)</p>',
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(
        "⚠️ Model files not found in the `model/` folder. Please run "
        "`machine_failure_prediction.ipynb` first to train and save the model, then restart this app."
    )
    st.stop()

st.info(
    "ℹ️ **Disclaimer:** This tool is a demonstration of predictive maintenance modeling on a "
    "public benchmark dataset. It is not calibrated for any specific real machine and should not "
    "replace certified maintenance inspections or manufacturer guidance.",
    icon="ℹ️",
)

# ------------------------------------------------------------------
# Sidebar — Input Mode
# ------------------------------------------------------------------
st.sidebar.header(" Sensor Input")
input_mode = st.sidebar.radio(
    "Choose input method:",
    ["Preset scenario", "Manual input (sliders)"],
    index=0,
)

presets = {
    "Normal operation": dict(type="M", air_temp=298.5, proc_temp=308.9, speed=1500, torque=40.0, tool_wear=50),
    "Elevated tool wear": dict(type="L", air_temp=299.0, proc_temp=309.5, speed=1400, torque=45.0, tool_wear=200),
    "High torque + high wear (overstrain risk)": dict(type="L", air_temp=300.0, proc_temp=311.0, speed=1350, torque=65.0, tool_wear=220),
    "High temp + low speed (heat dissipation risk)": dict(type="M", air_temp=303.5, proc_temp=312.5, speed=1280, torque=45.0, tool_wear=100),
}

if input_mode == "Preset scenario":
    preset_choice = st.sidebar.selectbox("Select a scenario:", list(presets.keys()))
    p = presets[preset_choice]
    machine_type = p["type"]
    air_temp = p["air_temp"]
    proc_temp = p["proc_temp"]
    speed = p["speed"]
    torque = p["torque"]
    tool_wear = p["tool_wear"]
    st.sidebar.caption("Switch to Manual input to fine-tune any of these values.")
else:
    st.sidebar.caption("Adjust sensor readings, then click Predict.")
    machine_type = st.sidebar.selectbox("Product Quality Type", ["L", "M", "H"], index=0,
                                         help="L = Low, M = Medium, H = High quality variant")
    air_temp = st.sidebar.slider("Air Temperature [K]",
                                  float(feature_stats.loc["Air temperature [K]", "min"]),
                                  float(feature_stats.loc["Air temperature [K]", "max"]),
                                  float(feature_stats.loc["Air temperature [K]", "mean"]))
    proc_temp = st.sidebar.slider("Process Temperature [K]",
                                   float(feature_stats.loc["Process temperature [K]", "min"]),
                                   float(feature_stats.loc["Process temperature [K]", "max"]),
                                   float(feature_stats.loc["Process temperature [K]", "mean"]))
    speed = st.sidebar.slider("Rotational Speed [rpm]",
                               float(feature_stats.loc["Rotational speed [rpm]", "min"]),
                               float(feature_stats.loc["Rotational speed [rpm]", "max"]),
                               float(feature_stats.loc["Rotational speed [rpm]", "mean"]))
    torque = st.sidebar.slider("Torque [Nm]",
                                float(feature_stats.loc["Torque [Nm]", "min"]),
                                float(feature_stats.loc["Torque [Nm]", "max"]),
                                float(feature_stats.loc["Torque [Nm]", "mean"]))
    tool_wear = st.sidebar.slider("Tool Wear [min]",
                                   float(feature_stats.loc["Tool wear [min]", "min"]),
                                   float(feature_stats.loc["Tool wear [min]", "max"]),
                                   float(feature_stats.loc["Tool wear [min]", "mean"]))

# Derived features
power_w = torque * (speed * 2 * np.pi / 60)
temp_diff = proc_temp - air_temp
type_encoded = type_encoder.transform([machine_type])[0]

input_values = {
    "Type_encoded": type_encoded,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": proc_temp,
    "Rotational speed [rpm]": speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Power [W]": power_w,
    "Temp difference [K]": temp_diff,
}

# ------------------------------------------------------------------
# Main Panel — Input Summary
# ------------------------------------------------------------------
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.subheader("Current Sensor Readings")
    summary_df = pd.DataFrame({
        "Sensor": ["Product Type", "Air Temperature [K]", "Process Temperature [K]",
                   "Rotational Speed [rpm]", "Torque [Nm]", "Tool Wear [min]",
                   "Power [W] (derived)", "Temp Difference [K] (derived)"],
        "Value": [machine_type, round(air_temp, 2), round(proc_temp, 2),
                  round(speed, 1), round(torque, 2), round(tool_wear, 1),
                  round(power_w, 1), round(temp_diff, 2)],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    predict_clicked = st.button("Run Failure Prediction", use_container_width=True)

with col_right:
    st.subheader(" Model Info")
    st.markdown(f"""
    <div class="metric-card">
    <b>Algorithm:</b> {type(model).__name__}<br>
    <b>Features used:</b> {len(feature_names)}<br>
    <b>Classes:</b> No Failure / Failure<br>
    <b>Dataset:</b> AI4I 2020 (10,000 samples)<br>
    <b>Base failure rate:</b> ~3.4%
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if predict_clicked:
    input_df = pd.DataFrame([input_values])[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    failure_prob = proba[1] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if failure_prob < 20:
            st.markdown(f"""
            <div class="result-ok">
            <h3>Low Risk — Normal Operation</h3>
            <p>Estimated failure probability: <b>{failure_prob:.1f}%</b><br>
            The machine's current readings are consistent with healthy operation.</p>
            </div>
            """, unsafe_allow_html=True)
        elif failure_prob < 60:
            st.markdown(f"""
            <div class="result-warn">
            <h3>⚠️ Moderate Risk — Monitor Closely</h3>
            <p>Estimated failure probability: <b>{failure_prob:.1f}%</b><br>
            Some readings are drifting outside typical ranges. Consider scheduling an inspection.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-danger">
            <h3> High Risk — Failure Likely</h3>
            <p>Estimated failure probability: <b>{failure_prob:.1f}%</b><br>
            Current sensor readings closely resemble historical failure conditions. Recommend immediate inspection.</p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=failure_prob,
            number={"suffix": "%"},
            title={"text": "Failure Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if failure_prob >= 60 else ("#f39c12" if failure_prob >= 20 else "#2ecc71")},
                "steps": [
                    {"range": [0, 20], "color": "#e8f9f0"},
                    {"range": [20, 60], "color": "#fff6e0"},
                    {"range": [60, 100], "color": "#fdecec"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Contextual signal flags based on known failure-mode physics from the dataset documentation
    st.subheader("Contributing Signal Flags")
    flag_cols = st.columns(4)
    with flag_cols[0]:
        st.metric("Power [W]", f"{power_w:.0f}", help="Failures common when power is outside ~3500–9000 W range")
    with flag_cols[1]:
        st.metric("Temp Diff [K]", f"{temp_diff:.1f}", help="Heat dissipation risk rises when temp diff < 8.6K and speed is low")
    with flag_cols[2]:
        st.metric("Tool Wear [min]", f"{tool_wear:.0f}", help="Overstrain risk rises sharply as tool wear increases with high torque")
    with flag_cols[3]:
        st.metric("Torque [Nm]", f"{torque:.1f}", help="Combined with high tool wear, elevated torque drives overstrain failures")



