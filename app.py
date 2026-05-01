"""
app.py — CrashSense Real-Time Accident Risk Prediction Dashboard
Run with:  streamlit run app.py
"""

import os
import time
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium

from model    import load_model, predict_real_time_risk, get_feature_importances, RISK_COLORS
from utils    import preprocess_data_v2, create_grids, aggregate_grid
from realtime import (
    get_live_weather,
    apply_simulation,
    weather_condition_from_live,
    WEATHER_CONDITIONS,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrashSense · Real-Time Risk Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a1220 100%);
    color: #e2e8f0;
}

/* ── Header ── */
.cs-header {
    background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(249,115,22,0.10), rgba(34,197,94,0.08));
    border: 1px solid rgba(239,68,68,0.30);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
}
.cs-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #EF4444, #F97316, #FBBF24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.cs-header p { color: #94a3b8; margin: 6px 0 0; font-size: 1rem; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.metric-value { font-size: 2.2rem; font-weight: 800; }
.metric-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .8px; margin-top: 4px; }
.metric-high   { color: #EF4444; }
.metric-medium { color: #F97316; }
.metric-low    { color: #22C55E; }
.metric-total  { color: #818cf8; }

/* ── Section cards ── */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Weather badge ── */
.weather-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.30);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    color: #a5b4fc;
    margin-bottom: 12px;
}

/* ── Risk badges ── */
.risk-high   { color:#EF4444; font-weight:700; }
.risk-medium { color:#F97316; font-weight:700; }
.risk-low    { color:#22C55E; font-weight:700; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Buttons ── */
.stButton>button {
    background: linear-gradient(135deg, #EF4444, #DC2626);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: opacity .2s, transform .2s;
}
.stButton>button:hover { opacity:.85; transform:translateY(-1px); }

/* ── Alert banner ── */
.alert-live {
    background: rgba(34,197,94,0.12);
    border-left: 4px solid #22C55E;
    border-radius: 8px;
    padding: 10px 16px;
    color: #86efac;
    font-size: 0.88rem;
    margin-bottom: 12px;
}
.alert-sim {
    background: rgba(99,102,241,0.12);
    border-left: 4px solid #818cf8;
    border-radius: 8px;
    padding: 10px 16px;
    color: #a5b4fc;
    font-size: 0.88rem;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Caching: data & model
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CrashSense model…")
def _load_model():
    return load_model()


@st.cache_data(show_spinner="Processing accident data… (first run only)")
def _load_and_process_data(data_path: str, n_rows: int = 500_000) -> pd.DataFrame:
    df = pd.read_csv(data_path, nrows=n_rows, low_memory=False)
    df = preprocess_data_v2(df)
    df = create_grids(df)
    grid = aggregate_grid(df)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_map(grid: pd.DataFrame, zoom: int = 4) -> folium.Map:
    """Build a Folium map with color-coded CircleMarkers."""
    lat_c = grid["Lat_Grid"].mean()
    lng_c = grid["Lng_Grid"].mean()

    m = folium.Map(
        location=[lat_c, lng_c],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )

    # Sub-sample for performance in the browser
    sample = grid.sample(min(len(grid), 3000), random_state=42)

    for _, row in sample.iterrows():
        risk   = row.get("Risk_Level", "Low")
        color  = RISK_COLORS.get(risk, "#22C55E")
        vis    = round(row.get("Avg_Visibility", 10), 2)
        rain   = int(row.get("Rain_Accidents", 0))
        hour   = round(row.get("Avg_Hour", 12), 1)
        cnt    = int(row.get("Accident_Count", 0))

        popup_html = f"""
        <div style="font-family:Inter,sans-serif;min-width:160px">
          <b style="color:{color};font-size:1rem">{risk} Risk</b><br>
          <hr style="margin:4px 0">
          🚗 Accidents: {cnt}<br>
          👁️ Visibility: {vis} mi<br>
          🌧️ Rain Events: {rain}<br>
          🕐 Avg Hour: {hour}h
        </div>"""

        folium.CircleMarker(
            location=[row["Lat_Grid"], row["Lng_Grid"]],
            radius=5 if risk == "High" else (4 if risk == "Medium" else 3),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.70,
            weight=1,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{risk} Risk | {cnt} accidents",
        ).add_to(m)

    # Legend
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:rgba(10,14,26,0.9);border:1px solid rgba(255,255,255,0.15);
                border-radius:12px;padding:12px 18px;font-family:Inter,sans-serif;
                color:#e2e8f0;font-size:0.82rem">
      <b>Risk Level</b><br>
      <span style="color:#EF4444">●</span> High<br>
      <span style="color:#F97316">●</span> Medium<br>
      <span style="color:#22C55E">●</span> Low
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    return m


def risk_distribution_chart(grid: pd.DataFrame, title: str = "Risk Distribution") -> go.Figure:
    counts = grid["Risk_Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    colors = [RISK_COLORS[k] for k in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=colors,
        text=counts.values.tolist(),
        textposition="outside",
        textfont=dict(color="white", size=14),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=48, b=20, l=20, r=20),
        height=280,
    )
    return fig


def comparison_chart(base: pd.DataFrame, sim: pd.DataFrame) -> go.Figure:
    cats   = ["High", "Medium", "Low"]
    base_c = base["Risk_Level"].value_counts().reindex(cats, fill_value=0)
    sim_c  = sim["Risk_Level"].value_counts().reindex(cats, fill_value=0)

    fig = go.Figure(data=[
        go.Bar(name="Before",    x=cats, y=base_c.values, marker_color=["#EF4444","#F97316","#22C55E"],
               opacity=0.6),
        go.Bar(name="Simulated", x=cats, y=sim_c.values,  marker_color=["#EF4444","#F97316","#22C55E"],
               opacity=1.0),
    ])
    fig.update_layout(
        barmode="group",
        title=dict(text="Before vs After Simulation", font=dict(color="#f1f5f9", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=48, b=20, l=20, r=20),
        height=300,
        legend=dict(font=dict(color="#94a3b8")),
    )
    return fig


def feature_importance_chart(model) -> go.Figure:
    fi = get_feature_importances(model)
    colors = px.colors.sequential.Plasma_r[:len(fi)]

    fig = go.Figure(go.Bar(
        x=fi.values[::-1],
        y=fi.index[::-1],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in fi.values[::-1]],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        title=dict(text="Feature Importances", font=dict(color="#f1f5f9", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Importance"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=48, b=20, l=20, r=20),
        height=320,
    )
    return fig


def metric_card(value, label, css_class="metric-total", suffix=""):
    return f"""
    <div class="metric-card">
      <div class="metric-value {css_class}">{value}{suffix}</div>
      <div class="metric-label">{label}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:12px 0 20px">
          <span style="font-size:2.5rem">🚨</span>
          <h2 style="color:#f1f5f9;margin:4px 0 0;font-size:1.3rem;font-weight:800">CrashSense</h2>
          <p style="color:#64748b;font-size:0.78rem;margin:2px 0">Real-Time Risk Intelligent accident hotspot detection</p>
        </div>
        <hr style="border-color:rgba(255,255,255,0.06);margin-bottom:16px">
        """, unsafe_allow_html=True)

        # ── Mode selector ──
        mode = st.radio(
            "🔀 Prediction Mode",
            ["🌐 Live Weather", "🎛️ Simulation Mode"],
            index=0,
        )
        st.markdown("<hr style='border-color:rgba(255,255,255,0.06)'>", unsafe_allow_html=True)

        # ── City input ──
        city = st.text_input("📍 City", value="Los Angeles", placeholder="e.g. New York")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.06)'>", unsafe_allow_html=True)
        st.markdown("**🎛️ Simulation Controls**")

        # ── Time slider ──
        hour = st.slider("🕐 Hour of Day", 0, 23, 14)

        # ── Weather dropdown ──
        cond = st.selectbox(
            "🌦️ Weather Condition",
            list(WEATHER_CONDITIONS.keys()),
            index=0,
        )

        # ── Data path ──
        st.markdown("<hr style='border-color:rgba(255,255,255,0.06)'>", unsafe_allow_html=True)
        st.markdown("**📂 Data**")
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        data_path = st.text_input(
            "CSV Path",
            value=os.path.join(base_dir, "data", "US_Accidents_March23.csv"),
            label_visibility="collapsed",
        )
        n_rows = st.number_input(
            "Rows to load (for speed)",
            min_value=50_000,
            max_value=3_000_000,
            value=300_000,
            step=50_000,
            format="%d",
        )

        run_btn = st.button("🔄 Run Prediction", use_container_width=True)

    return mode, city, hour, cond, data_path, int(n_rows), run_btn


# ─────────────────────────────────────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("""
    <div class="cs-header">
      <h1>🚨 CrashSense · Real-Time Risk Dashboard</h1>
      <p>AI-powered accident hotspot detection · Spatial Risk Forecasting · Live Weather Integration</p>
    </div>
    """, unsafe_allow_html=True)

    mode, city, hour, cond, data_path, n_rows, run_btn = render_sidebar()

    # ── Session state init ──
    if "grid_base" not in st.session_state:
        st.session_state.grid_base = None
    if "grid_pred" not in st.session_state:
        st.session_state.grid_pred = None
    if "grid_sim"  not in st.session_state:
        st.session_state.grid_sim  = None
    if "weather"   not in st.session_state:
        st.session_state.weather   = None
    

    # ── Load model eagerly ──
    try:
        model = _load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # ── Run pipeline ──
    if run_btn or st.session_state.grid_base is None:
        if not os.path.exists(data_path):
            st.error(f"Data file not found:\n`{data_path}`\n\nPlease check the path in the sidebar.")
            st.stop()

        with st.spinner("⚙️ Loading & processing data…"):
            grid_base = _load_and_process_data(data_path, n_rows)
            st.session_state.grid_base = grid_base

        # Baseline prediction (with full historical features)
        with st.spinner("🧠 Running baseline risk prediction…"):
            grid_pred = predict_real_time_risk(grid_base, {}, model)
            st.session_state.grid_pred = grid_pred

        # Weather / simulation
        if "Live" in mode:
            with st.spinner(f"🌐 Fetching live weather for {city}…"):
                weather = get_live_weather(city)
                st.session_state.weather = weather

            live_cond = weather_condition_from_live(weather)
            overrides = apply_simulation(
                hour   = pd.Timestamp.now().hour,
                weather_condition = live_cond,
                grid_mean_accident_count = grid_base["Accident_Count"].mean(),
            )
            # Use live visibility if available
            overrides["Avg_Visibility"] = weather["Avg_Visibility"]
        else:
            st.session_state.weather = None
            overrides = apply_simulation(
                hour              = hour,
                weather_condition = cond,
                grid_mean_accident_count = grid_base["Accident_Count"].mean(),
            )

        with st.spinner("🔮 Generating real-time predictions…"):
            grid_sim = predict_real_time_risk(grid_base, overrides, model)
            st.session_state.grid_sim = grid_sim

    grid_base = st.session_state.grid_base
    grid_pred = st.session_state.grid_pred
    grid_sim  = st.session_state.grid_sim
    weather   = st.session_state.weather

    if grid_base is None:
        st.info("👈 Press **Run Prediction** in the sidebar to start.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    #  Metric Cards
    # ─────────────────────────────────────────────────────────────────────────
    counts = grid_sim["Risk_Level"].value_counts().reindex(["High","Medium","Low"], fill_value=0)
    total  = len(grid_sim)
    pct_h  = round(counts.get("High",   0) / total * 100, 1)
    pct_m  = round(counts.get("Medium", 0) / total * 100, 1)
    pct_l  = round(counts.get("Low",    0) / total * 100, 1)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card(total, "Total Grid Cells", "metric-total"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card(f"{pct_h}", "% High Risk Areas", "metric-high", "%"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card(f"{pct_m}", "% Medium Risk Areas", "metric-medium", "%"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card(f"{pct_l}", "% Low Risk Areas", "metric-low", "%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Map + Weather Panel
    # ─────────────────────────────────────────────────────────────────────────
    col_map, col_side = st.columns([2, 1], gap="large")

    with col_map:
        with st.container(border=True):
            st.markdown('<div class="section-title">📍 Live Risk Map</div>', unsafe_allow_html=True)

            if "Live" in mode and weather:
                if weather["error"]:
                    st.markdown(f'<div class="alert-sim">⚠️ {weather["error"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-live">🟢 Live weather data active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-sim">🎛️ Simulation mode active</div>', unsafe_allow_html=True)

            with st.spinner("Rendering map…"):
                m = build_risk_map(grid_sim)
                st_folium(m, width="100%", height=480)

    with col_side:
        # ── Weather Panel ──
        with st.container(border=True):
            st.markdown('<div class="section-title">🌦️ Weather Panel</div>', unsafe_allow_html=True)

            if "Live" in mode and weather:
                temp_c = weather.get("temperature_c", "--")
                hum    = weather.get("humidity", "--")
                vis    = weather.get("visibility_mi", "--")
                wind   = weather.get("wind_speed_mph", "--")
                desc   = weather.get("description", "N/A")

                st.markdown(f"""
                <div class="weather-badge">🌍 {weather.get('city','—')}&nbsp;·&nbsp;{desc}</div>
                """, unsafe_allow_html=True)

                wc1, wc2 = st.columns(2)
                wc1.metric("🌡️ Temp", f"{temp_c} °C")
                wc2.metric("💧 Humidity", f"{hum}%")
                wc1.metric("👁️ Visibility", f"{vis} mi")
                wc2.metric("💨 Wind", f"{wind} mph")

                st.markdown("**Interpreted ML Features**")
                is_rain = "✅ Yes" if weather.get("Is_Rain") else "❌ No"
                st.markdown(f"""
                | Feature | Value |
                |---------|-------|
                | Is_Rain | {is_rain} |
                | Avg_Visibility | {vis} mi |
                | Avg_Hour | *current* |
                """)
            else:
                cond_meta = WEATHER_CONDITIONS.get(cond, {})
                is_rain = "✅ Yes" if cond_meta.get("Rain") else "❌ No"
                vis_val = cond_meta.get("visibility_mi", 10.0)

                st.markdown(f"""
                <div class="weather-badge">🎛️ Simulation · {cond}</div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                | Feature | Value |
                |---------|-------|
                | Is_Rain | {is_rain} |
                | Avg_Visibility | {vis_val} mi |
                | Avg_Hour | {hour}:00 |
                """)

        # ── Risk Distribution ──
        with st.container(border=True):
            st.markdown('<div class="section-title">📊 Risk Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(
                risk_distribution_chart(grid_sim, ""),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  Scenario Comparison + Feature Impact
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    comp_col, feat_col = st.columns([1, 1], gap="large")

    with comp_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">🔄 Scenario Comparison (Before vs After)</div>', unsafe_allow_html=True)

            base_counts = grid_pred["Risk_Level"].value_counts().reindex(["High","Medium","Low"], fill_value=0)
            sim_counts  = grid_sim["Risk_Level"].value_counts().reindex(["High","Medium","Low"], fill_value=0)

            delta_high = int(sim_counts["High"] - base_counts["High"])
            sign       = "+" if delta_high >= 0 else ""
            arrow      = "🔺" if delta_high > 0 else ("🔻" if delta_high < 0 else "➡️")

            st.markdown(f"""
            **Impact on High-Risk Cells:** {arrow} {sign}{delta_high} cells
            """)

            st.plotly_chart(
                comparison_chart(grid_pred, grid_sim),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with feat_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">🧠 Feature Impact (Model Importances)</div>', unsafe_allow_html=True)
            st.plotly_chart(
                feature_importance_chart(model),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  Data table (optional expandable)
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📋 View Prediction Data Table"):
        display_cols = [
            "Lat_Grid", "Lng_Grid", "Accident_Count",
            "Avg_Visibility", "Rain_Accidents", "Avg_Hour",
            "Risk_Level",
        ]
        st.dataframe(
            grid_sim[display_cols].sort_values("Accident_Count", ascending=False).head(200),
            use_container_width=True,
            height=320,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Footer
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.06);margin-top:32px">
    <div style="text-align:center;color:#334155;font-size:0.78rem;padding-bottom:12px">
      CrashSense · Real-Time Accident Risk Dashboard · Powered by RandomForest + OpenWeatherMap
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
