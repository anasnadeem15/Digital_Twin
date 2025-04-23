import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# Load models
MODELS = {
    'power_model': pickle.load(open('models/trained_model.pkl', 'rb')),
    'scaler': pickle.load(open('models/scaler.pkl', 'rb')),
    'source_model': pickle.load(open('models/power_classification_model.pkl', 'rb'))
}

TARIFFS = {'solar': 12.5, 'grid': 25.0}

# Page config
st.set_page_config(page_title="CNC G-code Analyzer", layout="centered")

# Load and apply custom CSS
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Load and render custom HTML header (optional)
with open("index.html") as html:
    st.markdown(html.read(), unsafe_allow_html=True)

# Title
st.title("CNC Lathe G-code Analyzer")

# G-code uploader
st.markdown("### Upload your G-code file (.nc, .txt, .gcode, .tap)")
file = st.file_uploader("Choose file", type=["nc", "txt", "gcode", "tap"])

# G-code Parser
def parse_gcode(gcode):
    feed = speed = depth = 0.0
    x_values = []
    z_values = []

    for line in gcode.split('\n'):
        line = line.strip().upper()
        if any(line.startswith(c) for c in ('(', '%', 'O', 'N', '#')):
            continue

        if 'F' in line:
            try:
                f_val = float(line.split('F')[1].split()[0])
                feed = max(feed, abs(f_val))
            except:
                pass

        if 'S' in line and 'G96' not in line:
            try:
                s_val = float(line.split('S')[1].split()[0])
                speed = max(speed, abs(s_val))
            except:
                pass

        if any(cmd in line for cmd in ['G00', 'G01', 'G02', 'G03']):
            for param in line.split():
                if param.startswith('X'):
                    try:
                        x_val = float(param[1:])
                        x_values.append(x_val)
                    except:
                        continue
                elif param.startswith('Z'):
                    try:
                        z_val = float(param[1:])
                        z_values.append(z_val)
                    except:
                        continue

    if x_values:
        depth = (max(x_values) - min(x_values)) / 2
    elif z_values:
        depth = max(z_values) - min(z_values)

    if speed == 0.0:
        speed = 1000.0
    if feed == 0.0:
        feed = 0.1

    return round(feed, 4), round(speed, 0), round(abs(depth), 4)

# If file uploaded
if file:
    gcode = file.read().decode("utf-8")
    feed, speed, depth = parse_gcode(gcode)

    st.success(f"✅ Parsed G-code: \n- **Feed Rate:** {feed} mm/rev \n- **Spindle Speed:** {speed} RPM \n- **Depth of Cut:** {depth} mm")

    if st.button("🔍 Predict Performance"):
        X_scaled = MODELS['scaler'].transform([[feed, depth, speed]])
        power = round(float(MODELS['power_model'].predict(X_scaled)[0]), 2)
        source = "Solar" if MODELS['source_model'].predict(X_scaled)[0] == 1 else "Grid"

        tool_life = (200 / speed) * (1 / (feed ** 0.8)) * (1 / (depth ** 0.4)) * 60
        tool_life = round(tool_life, 1)
        operation_time = tool_life / 60
        energy_consumed = (power * operation_time) / 1000

        costs = {
            'solar': round(energy_consumed * TARIFFS['solar'], 2),
            'grid': round(energy_consumed * TARIFFS['grid'], 2)
        }

        st.markdown("### 📊 Prediction Results")
        st.markdown(f"- **Power Consumption:** `{power} W`")
        st.markdown(f"- **Recommended Source:** `{source}`")
        st.markdown(f"- **Tool Life:** `{tool_life} minutes`")
        st.markdown(f"- **Cost Estimate:** Solar = Rs. `{costs['solar']}` | Grid = Rs. `{costs['grid']}`")
        st.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
