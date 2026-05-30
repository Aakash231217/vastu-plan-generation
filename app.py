import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.constants import COMMON_ROOMS, OPTIONAL_ROOMS, VASTU_RULES
from core.generator import generate_population, rank_plans
from core.visualizer import draw_comparison, fig_to_bytes


st.set_page_config(
    page_title="AI Floorplan Generator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root { --bg: #0D1117; --card: #161B22; --accent: #1F4E79; --text: #C9D1D9; }
    .stApp { background-color: var(--bg); color: var(--text); }
    .stSidebar { background-color: #0D1117 !important; }
    .main-header {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 55%, #1B5E20 100%);
        padding: 1.6rem 2rem 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(31,78,121,0.35);
    }
    .main-header h1 { color: white; font-size: 2rem; margin:0; letter-spacing:0; }
    .main-header p  { color: #D7E5EF; font-size: 0.95rem; margin:0.35rem 0 0 0; }
    .section-header {
        border-left: 4px solid #1F4E79;
        padding-left: 0.75rem;
        margin: 1.2rem 0 0.6rem 0;
        font-size: 1.1rem; font-weight: bold; color: #4FC3F7;
    }
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .room-list {
        background: #111820;
        border: 1px solid #273447;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        font-size: 0.82rem;
        line-height: 1.45;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: bold; font-size: 1rem;
        width: 100%; transition: all 0.2s;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #2E75B6, #1F4E79); transform: translateY(-1px); }
</style>
""",
    unsafe_allow_html=True,
)


def pretty_room(name: str) -> str:
    return (
        name.replace("_", " ")
        .replace("01", "1")
        .replace("02", "2")
        .replace("03", "3")
        .title()
    )


def metric_row(raw_fp, opt_fp):
    raw = raw_fp.scores
    opt = opt_fp.scores
    gain = opt["Fitness"] - raw["Fitness"]
    cols = st.columns(5)
    metrics = [
        ("Raw Fitness", f"{raw['Fitness']:.3f}", ""),
        ("Optimised Fitness", f"{opt['Fitness']:.3f}", f"{gain:+.3f}"),
        ("Vastu", f"{opt['VS'] * 100:.0f}%", f"{(opt['VS'] - raw['VS']) * 100:+.0f}%"),
        ("Adjacency", f"{opt['AQ'] * 100:.0f}%", f"{(opt['AQ'] - raw['AQ']) * 100:+.0f}%"),
        ("Space Usage", f"{opt['SU'] * 100:.0f}%", f"{(opt['SU'] - raw['SU']) * 100:+.0f}%"),
    ]
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


def score_table(fp):
    score = fp.scores
    return pd.DataFrame(
        {
            "Metric": ["Fitness", "Vastu", "Adjacency", "Space Usage", "Area Compliance"],
            "Score": [
                f"{score['Fitness']:.3f}",
                f"{score['VS'] * 100:.0f}%",
                f"{score['AQ'] * 100:.0f}%",
                f"{score['SU'] * 100:.0f}%",
                f"{score['AC'] * 100:.0f}%",
            ],
        }
    )


st.markdown(
    """
<div class="main-header">
    <h1>AI Floorplan Generator</h1>
    <p>Graph VAE + Geometry Predictor + Vastu/Space-Usage Optimisation</p>
    <p style="font-size:0.78rem; opacity:0.78;">Generated plans vs optimised plans with fitness-score comparison</p>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    st.markdown("### Plot Dimensions")
    c1, c2 = st.columns(2)
    with c1:
        plot_w = st.number_input("Width (ft)", min_value=20, max_value=100, value=40, step=5)
    with c2:
        plot_h = st.number_input("Depth (ft)", min_value=20, max_value=100, value=60, step=5)
    st.caption(f"Plot area: {plot_w * plot_h} sq ft")

    st.markdown("### Road Side")
    road_side = st.selectbox(
        "Road-facing side",
        options=["S", "N", "E", "W"],
        index=0,
        format_func={"S": "South", "N": "North", "E": "East", "W": "West"}.get,
    )
    st.caption("Setbacks: 2 ft on road side, 1 ft on all other sides.")

    st.markdown("### Common Rooms")
    st.caption("Included in every generated plan")
    common_html = "<div class='room-list'><ul style='margin:0 0 0 1rem; padding:0;'>"
    for room in COMMON_ROOMS:
        common_html += f"<li>{pretty_room(room)}</li>"
    common_html += "</ul></div>"
    st.markdown(common_html, unsafe_allow_html=True)

    st.markdown("### Optional Rooms")
    optional_rooms = []
    defaults = {
        "bedroom_01": True,
        "bedroom_02": True,
        "bedroom_03": False,
        "balcony": True,
        "study": False,
        "pooja_room": True,
    }
    for room in OPTIONAL_ROOMS:
        if st.checkbox(pretty_room(room), value=defaults.get(room, False)):
            optional_rooms.append(room)

    room_names = COMMON_ROOMS + optional_rooms

    st.markdown("### Vastu Rules")
    all_vastu_names = [rule[0] for rule in VASTU_RULES]
    selected_vastu = st.multiselect(
        "Active Vastu Rules",
        options=all_vastu_names,
        default=all_vastu_names,
        label_visibility="collapsed",
    )

    st.markdown("### Objective Weights")
    with st.expander("Customise weights"):
        w_aq = st.slider("Adjacency Quality", 0.0, 1.0, 0.25, 0.05)
        w_vs = st.slider("Vastu Score", 0.0, 1.0, 0.35, 0.05)
        w_su = st.slider("Space Usage", 0.0, 1.0, 0.30, 0.05)
        w_ac = st.slider("Area Compliance", 0.0, 1.0, 0.10, 0.05)
        total_w = w_aq + w_vs + w_su + w_ac
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.2f}; ideally 1.00")

    st.markdown("---")
    generate_btn = st.button("Generate Floor Plans", use_container_width=True)


if generate_btn:
    progress_bar = st.progress(0, text="Initialising generation pipeline...")
    status = st.empty()

    with st.spinner(""):
        status.markdown("**Step 1/4** - Graph VAE generates candidate room graphs")
        progress_bar.progress(20, text="Graph VAE: generating candidate topologies...")
        time.sleep(0.2)

        status.markdown("**Step 2/4** - Geometry predictor places rooms")
        progress_bar.progress(45, text="Geometry Predictor: producing raw plans...")
        plan_pairs = generate_population(
            room_names=room_names,
            plot_w=plot_w,
            plot_h=plot_h,
            n=3,
            required_rooms=COMMON_ROOMS,
            optional_rooms=optional_rooms,
            road_side=road_side,
        )
        time.sleep(0.2)

        status.markdown("**Step 3/4** - Optimisation removes wasted space")
        progress_bar.progress(75, text="Optimiser: applying Vastu + space-usage objective...")
        ranked = rank_plans(
            plan_pairs,
            selected_vastu,
            w_aq=w_aq,
            w_vs=w_vs,
            w_su=w_su,
            w_ac=w_ac,
        )
        top3 = ranked[:3]

        status.markdown("**Step 4/4** - Scoring raw vs optimised plans")
        progress_bar.progress(100, text="Complete")
        time.sleep(0.2)

    progress_bar.empty()
    status.empty()

    best_raw = top3[0]["raw"]
    best_opt = top3[0]["optimised"]
    st.success("Generated 3 raw plans and 3 optimised plans for score comparison.")

    st.markdown('<div class="section-header">Best Plan Score Improvement</div>', unsafe_allow_html=True)
    metric_row(best_raw, best_opt)

    st.markdown('<div class="section-header">Generated Plans vs Optimised Plans</div>', unsafe_allow_html=True)
    tabs = st.tabs([f"Plan {idx + 1}" for idx in range(len(top3))])
    for idx, (tab, pair) in enumerate(zip(tabs, top3)):
        with tab:
            raw_fp = pair["raw"]
            opt_fp = pair["optimised"]
            metric_row(raw_fp, opt_fp)
            comparison_fig = draw_comparison(raw_fp, opt_fp, rank=idx + 1, road_side=road_side)
            st.image(fig_to_bytes(comparison_fig), use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.markdown("**Generated plan objective scores**")
                st.dataframe(score_table(raw_fp), hide_index=True, use_container_width=True)
            with right:
                st.markdown("**Optimised plan objective scores**")
                st.dataframe(score_table(opt_fp), hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header">Ranked Comparison</div>', unsafe_allow_html=True)
    rows = []
    for idx, pair in enumerate(top3):
        raw = pair["raw"].scores
        opt = pair["optimised"].scores
        rows.append(
            {
                "Plan": f"Plan {idx + 1}",
                "Raw Fitness": f"{raw['Fitness']:.3f}",
                "Optimised Fitness": f"{opt['Fitness']:.3f}",
                "Gain": f"{(opt['Fitness'] - raw['Fitness']):+.3f}",
                "Vastu": f"{opt['VS'] * 100:.0f}%",
                "Adjacency": f"{opt['AQ'] * 100:.0f}%",
                "Space Usage": f"{opt['SU'] * 100:.0f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 2rem; color: #8B949E;">
            <h3 style="color: #4FC3F7;">Configure the plot and optional rooms in the sidebar</h3>
            <p>The demo will generate 3 raw plans, optimise them, and compare fitness scores.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Model Flow</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "1. Plot + Setbacks", "Green plot rectangle, red setback line: 2 ft road side and 1 ft other sides."),
        (c2, "2. Required Rooms", "External staircase, foyer, living, kitchen+dining+utility, bathrooms, master bedroom."),
        (c3, "3. AI Generation", "Graph VAE + Geometry Predictor produce raw room placements."),
        (c4, "4. Optimisation", "Fitness improves using Vastu, adjacency, area compliance, and space-usage scores."),
    ]
    for col, title, desc in cards:
        col.markdown(
            f"""
            <div class="metric-card" style="text-align:left; min-height:145px;">
                <div style="font-size:1rem; margin-bottom:0.4rem;"><b style="color:#4FC3F7;">{title}</b></div>
                <div style="font-size:0.82rem; color:#8B949E; line-height:1.4;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
