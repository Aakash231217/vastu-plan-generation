# AI Floorplan Generator
### Dissertation Project — Hybrid GNN + Diffusion + NSGA-II Architecture

---

## What This Does

Generates and optimises 2D residential floor plans for Indian homes with respect to:
- **Vastu Shastra** compliance (15 rules, user-selectable)
- **Construction Cost** estimation (India-specific rates)
- **Climatic feasibility** suggestions (5 Indian climate zones)
- **Spatial quality** — adjacency, efficiency, compactness

Shows **Top 5 optimised floor plans** ranked by a multi-objective fitness score.

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

### Step 3 — Open in browser
```
http://localhost:8501
```

---

## Project Structure

```
floorplan_project/
├── app.py                  ← Main Streamlit GUI
├── requirements.txt
├── core/
│   ├── constants.py        ← Room definitions, Vastu rules, cost rates
│   ├── generator.py        ← Floor plan generation + all scoring functions
│   └── visualizer.py       ← Matplotlib floor plan rendering
└── README.md
```

---

## Architecture Pipeline

```
User Input (GUI)
      ↓
Graph VAE        → Generates room adjacency graphs
      ↓
Diffusion Stg 1  → Graph → Room bounding boxes (x,y,w,h)
      ↓
Diffusion Stg 2  → Boxes → Full floor plan with walls/doors
      ↓
Constraints Check → Validates 7 hard structural constraints
      ↓
NSGA-II          → Optimises: AQ + SE + LC + VS + AC + Cost
      ↓
Top 5 Plans      → Displayed in GUI with full analysis
```

---

## Objective Functions

| Symbol | Name | Goal |
|--------|------|------|
| AQ | Adjacency Quality | MAXIMISE |
| SE | Spatial Efficiency | MAXIMISE |
| LC | Layout Compactness | MAXIMISE |
| VS | Vastu Score | MAXIMISE |
| AC | Area Compliance | MAXIMISE |
| CC | Construction Cost | MINIMISE |

---

## Room Classes (Indian-Specific)

`master_bedroom`, `bedroom_01/02/03`, `kitchen`, `living_room`, `dining_room`,
`common_bathroom`, `attached_bathroom`, `powder_room`, `foyer`,
`internal_corridor`, `external_corridor`, `pooja_room`, `utility_area`,
`balcony`, `study`

---

*Dissertation: AI Framework for Automated Generation and Optimisation of 2D Floorplans
w.r.t. Vastu Orientation, Cost Effectiveness & Climatic Feasibility*
