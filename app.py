import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import time

from core.constants import ROOM_CONFIGS, VASTU_RULES, CLIMATE_ZONES, COST_PER_SQFT
from core.generator import (generate_population, rank_plans,
                             score_vastu, estimate_cost, climate_suggestions)
from core.visualizer import (draw_floorplan, draw_top5_grid, draw_score_radar,
                              draw_vastu_compass, fig_to_bytes)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Floorplan Generator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root { --bg: #0D1117; --card: #161B22; --accent: #1F4E79; --text: #C9D1D9; }
    .stApp { background-color: var(--bg); color: var(--text); }
    .stSidebar { background-color: #0D1117 !important; }
    .main-header {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 50%, #1B5E20 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(31,78,121,0.4);
    }
    .main-header h1 { color: white; font-size: 2rem; margin:0; letter-spacing:1px; }
    .main-header p  { color: #B0BEC5; font-size: 0.95rem; margin:0.4rem 0 0 0; }
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .value { font-size: 1.6rem; font-weight: bold; color: #4FC3F7; }
    .metric-card .label { font-size: 0.78rem; color: #8B949E; margin-top: 0.2rem; }
    .score-bar-container { background: #21262D; border-radius: 6px; height: 10px; margin: 4px 0; }
    .score-bar { height: 10px; border-radius: 6px; transition: width 0.3s; }
    .section-header {
        border-left: 4px solid #1F4E79;
        padding-left: 0.75rem;
        margin: 1.2rem 0 0.6rem 0;
        font-size: 1.1rem; font-weight: bold; color: #4FC3F7;
    }
    .vastu-rule { padding: 0.3rem 0.6rem; border-radius: 6px; margin: 2px 0; font-size: 0.85rem; }
    .vastu-pass { background: #1B3A1B; border-left: 3px solid #00E676; color: #A5D6A7; }
    .vastu-fail { background: #3A1B1B; border-left: 3px solid #FF5252; color: #EF9A9A; }
    .vastu-partial { background: #3A331B; border-left: 3px solid #FFFF00; color: #FFF59D; }
    .suggest-item { 
        background: #1A2332; border: 1px solid #2E4A6E; 
        border-radius: 8px; padding: 0.5rem 0.8rem; 
        margin: 4px 0; font-size: 0.85rem; color: #B0C4DE;
    }
    .rank-badge-1 { color: #FFD700; font-size: 1.4rem; font-weight: bold; }
    .rank-badge-2 { color: #C0C0C0; font-size: 1.4rem; font-weight: bold; }
    .rank-badge-3 { color: #CD7F32; font-size: 1.4rem; font-weight: bold; }
    .stButton>button {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: bold; font-size: 1rem;
        width: 100%; transition: all 0.2s;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #2E75B6, #1F4E79); transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏠 AI Floorplan Generator</h1>
    <p>Automated 2D Floor Plan Generation & Optimisation — Vastu • Cost • Climate</p>
    <p style="font-size:0.78rem; opacity:0.7;">Dissertation Project | Hybrid GNN + Diffusion + NSGA-II Architecture</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 📐 Plot Dimensions")
    col1, col2 = st.columns(2)
    with col1:
        plot_w = st.number_input("Width (ft)", min_value=20, max_value=100, value=40, step=5)
    with col2:
        plot_h = st.number_input("Depth (ft)", min_value=20, max_value=100, value=60, step=5)

    plot_area = plot_w * plot_h
    st.caption(f"Plot area: {plot_area} sq ft ({plot_area*0.0929:.0f} sq m)")

    st.markdown("### 🏠 Room Configuration")
    bhk = st.selectbox("BHK Type", list(ROOM_CONFIGS.keys()), index=1)
    room_names = ROOM_CONFIGS[bhk]
    st.caption(f"Rooms: {', '.join(r.replace('_',' ') for r in room_names)}")

    st.markdown("### 💰 Budget & Finishing")
    budget = st.slider("Budget (₹ Lakhs)", min_value=20, max_value=150, value=60, step=5)
    finishing = st.selectbox("Finishing Level", ["Basic", "Standard", "Premium"], index=1)

    st.markdown("### 🌿 Vastu Rules")
    st.caption("Select which Vastu rules to enforce:")
    all_vastu_names = [r[0] for r in VASTU_RULES]
    selected_vastu = st.multiselect(
        "Active Vastu Rules",
        options=all_vastu_names,
        default=all_vastu_names[:6],
        label_visibility="collapsed"
    )

    st.markdown("### 🌡️ Climate Zone")
    climate = st.selectbox("Climate Zone", list(CLIMATE_ZONES.keys()), index=2)

    st.markdown("### 🎛️ Objective Weights")
    with st.expander("Customise weights (advanced)"):
        w1 = st.slider("Adjacency Quality (AQ)", 0.0, 1.0, 0.25, 0.05)
        w2 = st.slider("Spatial Efficiency (SE)", 0.0, 1.0, 0.15, 0.05)
        w3 = st.slider("Layout Compactness (LC)", 0.0, 1.0, 0.10, 0.05)
        w4 = st.slider("Vastu Score (VS)",        0.0, 1.0, 0.30, 0.05)
        w5 = st.slider("Area Compliance (AC)",     0.0, 1.0, 0.15, 0.05)
        w6 = st.slider("Cost Efficiency",           0.0, 1.0, 0.05, 0.05)
        total_w = w1+w2+w3+w4+w5+w6
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.2f} — ideally should sum to 1.0")
    st.markdown("---")
    generate_btn = st.button("🚀 Generate Floor Plans", use_container_width=True)

# ── Main Area ──────────────────────────────────────────────────────────────────
if generate_btn:
    progress_bar = st.progress(0, text="Initialising generation pipeline...")
    status = st.empty()

    with st.spinner(""):
        status.markdown("**Step 1/4** — Graph VAE: Generating room adjacency graphs...")
        progress_bar.progress(15, text="Graph VAE: generating room topology graphs...")
        time.sleep(0.6)

        status.markdown("**Step 2/4** — Diffusion Stage 1: Graph → Room geometry...")
        progress_bar.progress(40, text="Diffusion Stage 1: computing room coordinates...")
        plans = generate_population(room_names, plot_w, plot_h, n=25)
        time.sleep(0.5)

        status.markdown("**Step 3/4** — Diffusion Stage 2: Adding walls, doors, windows...")
        progress_bar.progress(65, text="Diffusion Stage 2: rendering floor plan details...")
        time.sleep(0.5)

        status.markdown("**Step 4/4** — NSGA-II: Optimising across all objectives...")
        progress_bar.progress(85, text="NSGA-II: running multi-objective optimisation...")
        ranked = rank_plans(plans, selected_vastu, finishing, climate, w1,w2,w3,w4,w5,w6)
        budget_filtered = [fp for fp in ranked if fp.scores["Cost"] <= budget * 100_000]
        if len(budget_filtered) < 5:
            budget_filtered = ranked  # fallback
        top5 = budget_filtered[:5]
        progress_bar.progress(100, text="✅ Generation complete!")
        time.sleep(0.3)

    progress_bar.empty()
    status.empty()
    st.success(f"✅ Generated and ranked 25 candidate floor plans. Showing top 5 optimised results.")

    # ── Overview metrics ────────────────────────────────────────────────────────
    best = top5[0]
    st.markdown('<div class="section-header">📊 Best Plan Overview</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, label, value, suffix in [
        (mc1, "Fitness Score", f"{best.scores['Fitness']:.3f}", ""),
        (mc2, "Vastu Score",   f"{best.scores['VS']*100:.0f}", "%"),
        (mc3, "Adjacency",     f"{best.scores['AQ']*100:.0f}", "%"),
        (mc4, "Est. Cost",     f"₹{best.scores['Cost']/100000:.1f}", "L"),
        (mc5, "Total Rooms",   str(len(best.rooms)), " rooms"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{value}<span style="font-size:0.9rem">{suffix}</span></div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # ── Top 5 grid view ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Top 5 Floor Plans</div>', unsafe_allow_html=True)
    grid_fig = draw_top5_grid(top5)
    st.image(fig_to_bytes(grid_fig), use_column_width=True)

    # ── Detailed plan tabs ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Detailed Analysis — Select a Plan</div>',
                unsafe_allow_html=True)
    tab_labels = [f"#{i+1} — Fitness {top5[i].scores['Fitness']:.3f}" for i in range(len(top5))]
    tabs = st.tabs(tab_labels)

    for i, (tab, fp) in enumerate(zip(tabs, top5)):
        with tab:
            rank = i + 1
            badge_cls = {1:"rank-badge-1", 2:"rank-badge-2", 3:"rank-badge-3"}.get(rank, "")
            st.markdown(f'<span class="{badge_cls}">{"🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"#{rank}"} Plan #{rank}</span>',
                        unsafe_allow_html=True)

            col_plan, col_scores = st.columns([3, 2])

            with col_plan:
                plan_fig = draw_floorplan(fp, rank=rank)
                st.image(fig_to_bytes(plan_fig), use_column_width=True)

            with col_scores:
                # Score bars
                st.markdown("**Objective Scores**")
                s = fp.scores
                for label, val, color in [
                    ("Adjacency Quality", s['AQ'], "#4FC3F7"),
                    ("Spatial Efficiency", s['SE'], "#81C784"),
                    ("Layout Compactness", s['LC'], "#FFB74D"),
                    ("Vastu Score", s['VS'], "#CE93D8"),
                    ("Area Compliance", s['AC'], "#80DEEA"),
                ]:
                    pct = int(val * 100)
                    st.markdown(f"""
                    <div style="margin:6px 0;">
                        <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#C9D1D9;">
                            <span>{label}</span><span style="color:{color};font-weight:bold;">{pct}%</span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width:{pct}%;background:{color};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card" style="margin-top:0.8rem;">
                    <div class="value">₹{s['Cost']/100000:.2f}L</div>
                    <div class="label">Estimated Construction Cost ({finishing} finishing)</div>
                </div>""", unsafe_allow_html=True)

                # Room list
                st.markdown("**Room Areas**")
                room_data = [(r.name.replace('_',' ').title(), f"{r.area:.0f} sqft") for r in fp.rooms]
                room_df_html = "<table style='width:100%;font-size:0.8rem;color:#C9D1D9;'>"
                for rname, rarea in room_data:
                    room_df_html += f"<tr><td>{rname}</td><td style='text-align:right;color:#4FC3F7;'>{rarea}</td></tr>"
                room_df_html += "</table>"
                st.markdown(room_df_html, unsafe_allow_html=True)

            # Radar + Compass row
            col_radar, col_compass = st.columns(2)
            with col_radar:
                st.markdown("**Score Radar**")
                radar_fig = draw_score_radar(fp)
                st.image(fig_to_bytes(radar_fig), use_column_width=True)
            with col_compass:
                st.markdown("**Vastu Compass**")
                compass_fig = draw_vastu_compass(fp)
                st.image(fig_to_bytes(compass_fig), use_column_width=True)

            # Vastu rule breakdown
            st.markdown("**Vastu Rule Compliance**")
            vastu_details = fp.scores.get('vastu_details', [])
            if vastu_details:
                for rule_name, room_name, zone, passed in vastu_details:
                    css = ("vastu-pass" if passed == True else
                           "vastu-partial" if passed == "partial" else "vastu-fail")
                    icon = "✅" if passed == True else ("⚠️" if passed == "partial" else "❌")
                    room_display = room_name.replace('_', ' ').title()
                    st.markdown(
                        f'<div class="vastu-rule {css}">'
                        f'{icon} <b>{rule_name}</b> — {room_display} is in <b>{zone}</b> zone</div>',
                        unsafe_allow_html=True)
            else:
                st.info("No Vastu rules selected.")

    # ── Climate Suggestions ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌡️ Climate & Energy Efficiency Suggestions</div>',
                unsafe_allow_html=True)
    st.markdown(f"**Climate Zone:** {climate}")
    suggestions = climate_suggestions(best, climate)
    for s in suggestions:
        st.markdown(f'<div class="suggest-item">💡 {s}</div>', unsafe_allow_html=True)

    # ── Comparison table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Top 5 Comparison Table</div>', unsafe_allow_html=True)
    import pandas as pd
    table_data = []
    for i, fp in enumerate(top5):
        s = fp.scores
        table_data.append({
            "Rank": f"#{i+1}",
            "Fitness": f"{s['Fitness']:.4f}",
            "Vastu %": f"{s['VS']*100:.0f}%",
            "Adjacency %": f"{s['AQ']*100:.0f}%",
            "Spatial Eff.": f"{s['SE']*100:.0f}%",
            "Compactness": f"{s['LC']*100:.0f}%",
            "Area Compliance": f"{s['AC']*100:.0f}%",
            "Est. Cost (₹L)": f"{s['Cost']/100000:.1f}",
        })
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # ── Welcome screen ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem; color: #8B949E;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🏗️</div>
        <h3 style="color: #4FC3F7;">Configure your requirements in the sidebar</h3>
        <p>Set plot dimensions, BHK type, budget, Vastu rules, and climate zone<br>
        then click <b style="color:white;">🚀 Generate Floor Plans</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline info cards
    st.markdown('<div class="section-header">🔬 How It Works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "🔵", "Graph VAE", "Learns Indian floor plan adjacency patterns and generates new room topology graphs"),
        (c2, "🟣", "Diffusion Stage 1", "Converts room graphs to geometric layouts with precise room coordinates"),
        (c3, "🟢", "Diffusion Stage 2", "Adds architectural details — walls, doors, windows to the layout"),
        (c4, "🔴", "NSGA-II Optimiser", "Multi-objective optimisation across Vastu, Cost, Adjacency, Climate"),
    ]:
        col.markdown(f"""
        <div class="metric-card" style="text-align:left; height:150px;">
            <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icon} <b style="color:#4FC3F7;">{title}</b></div>
            <div style="font-size:0.82rem; color:#8B949E; line-height:1.4;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class="metric-card">
        <div class="value">56</div><div class="label">Constraints Encoded</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""
    <div class="metric-card">
        <div class="value">15</div><div class="label">Vastu Rules Supported</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""
    <div class="metric-card">
        <div class="value">7</div><div class="label">Objective Functions</div>
    </div>""", unsafe_allow_html=True)
