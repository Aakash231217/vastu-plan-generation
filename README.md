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

### Step 2 — Download the trained model weights
Copy the three checkpoint files from Google Drive into the `models/` folder:

```
models/graph_vae.pth            ← Graph VAE (200 epochs, loss 2.15)
models/geometry_predictor.pth   ← Geometry Predictor (500 epochs, loss 0.38)
models/best.pt                  ← YOLOv8 room detector (optional, mAP50 ≈ 54%)
```

If these files are missing, the app still runs but falls back to a rule-based
rectangle packer instead of the trained AI pipeline.

### Step 3 — Run the app
```bash
streamlit run app.py
```

### Step 4 — Open in browser
```
http://localhost:8501
```

---

## Project Structure

```
vastu-plan-generation/
├── app.py                  ← Main Streamlit GUI
├── requirements.txt
├── models/                 ← Trained weights (downloaded from Drive)
│   ├── graph_vae.pth
│   ├── geometry_predictor.pth
│   └── best.pt
├── core/
│   ├── constants.py        ← Room definitions, Vastu rules, cost rates
│   ├── generator.py        ← Bridges ML pipeline → FloorPlan objects + scoring
│   ├── ml_models.py        ← GraphVAE + GeometryPredictor class definitions
│   ├── ml_pipeline.py      ← Rejection-sampling pipeline (Cells E + F port)
│   └── visualizer.py       ← Matplotlib floor plan rendering
└── README.md
```

---

## Pipeline numbers (training + validation)

| Stage              | Setting                                | Result            |
| ------------------ | -------------------------------------- | ----------------- |
| YOLOv8 detector    | 1.2k images, 11 classes                | mAP50 ≈ 54%       |
| Graph VAE          | 3283 RAG graphs, 200 epochs            | Final loss 2.15   |
| Geometry Predictor | 500 epochs, cosine LR + adjacency loss | Final MSE 0.38    |
| End-to-end         | Rejection sampling (HouseGAN++ style)  | 100/100 valid     |
|                    | Underlying acceptance rate             | 23% (~4 tries)    |

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
