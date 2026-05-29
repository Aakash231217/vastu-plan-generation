"""
End-to-end AI floorplan generation pipeline.

Ports Cells E (rejection sampling + validity) and F (apartment-style layout)
from the Colab notebook to a local CPU/GPU-agnostic module. All heavy lifting
goes through torch; if torch isn't available or the weights aren't on disk
we raise — callers must check `is_ready()` first.
"""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ml_models import load_models, models_available


# ── Domain constants (kept consistent with Cell F) ────────────────────────────

OUTER_MARGIN = 3.0           # ft uniform inner setback (apartment wall)
DEFAULT_PLOT_W = 40.0
DEFAULT_PLOT_H = 60.0

MIN_DIMS = {
    "master_bedroom":  (12, 12),
    "bedroom":         (10, 10),
    "living_room":     (14, 12),
    "kitchen":         (8, 10),
    "dining_room":     (10, 10),
    "common_bathroom": (5, 7),
    "pooja_room":      (5, 5),
    "study":           (8, 9),
    "balcony":         (5, 7),
    "staircase":       (8, 12),
    "utility_area":    (5, 6),
}
MIN_AREA = {k: w * h for k, (w, h) in MIN_DIMS.items()}

# Cap how many of each class show up in a single plan
ROOM_RULES_MAX = {
    "bedroom": 3, "balcony": 2, "common_bathroom": 2, "dining_room": 1,
    "pooja_room": 1, "study": 1, "staircase": 1, "utility_area": 1,
    "living_room": 1, "kitchen": 1, "master_bedroom": 1,
}

CRITICAL_EDGES = [
    ("kitchen", "dining_room"),
    ("dining_room", "living_room"),
    ("kitchen", "living_room"),
    ("living_room", "master_bedroom"),
    ("master_bedroom", "common_bathroom"),
    ("living_room", "bedroom"),
    ("living_room", "balcony"),
    ("living_room", "pooja_room"),
    ("living_room", "staircase"),
]

CRITICAL_ROOMS = ["living_room", "kitchen", "master_bedroom", "common_bathroom"]


# ── Lazy singleton for models ────────────────────────────────────────────────

@dataclass
class _ModelBundle:
    gcn: object
    geo: object
    classes: list
    class_to_id: dict
    device: str


_bundle: Optional[_ModelBundle] = None


def is_ready() -> bool:
    return models_available()


def get_models() -> _ModelBundle:
    global _bundle
    if _bundle is None:
        gcn, geo, classes, device = load_models()
        _bundle = _ModelBundle(
            gcn=gcn,
            geo=geo,
            classes=classes,
            class_to_id={c: i for i, c in enumerate(classes)},
            device=device,
        )
    return _bundle


# ── Graph generation (Cell E v2: enforces criticals + room caps) ──────────────

def generate_graph_v2(target_rooms: int = 8, top_k: int = 1,
                       edge_threshold: float = 0.55):
    import torch
    bundle = get_models()
    gcn = bundle.gcn
    ALL_CLASSES = bundle.classes
    class_to_id = bundle.class_to_id
    device = bundle.device

    with torch.no_grad():
        z = torch.randn(target_rooms * 3, 16, device=device)
        node_logits, _ = gcn.decode(z)
        probs = torch.softmax(node_logits, dim=1).cpu().numpy()

    selected, classes, counts = [], [], {c: 0 for c in ALL_CLASSES}

    # Force the 4 critical rooms first
    for cls in CRITICAL_ROOMS:
        if cls in class_to_id:
            cid = class_to_id[cls]
            nid = int(np.argmax(probs[:, cid]))
            selected.append(nid)
            classes.append(cls)
            counts[cls] = counts.get(cls, 0) + 1
            probs[nid] = -1

    # Fill remaining slots greedily, respecting per-class caps
    while len(classes) < target_rooms:
        flat = probs.copy()
        for c, cnt in counts.items():
            cap = ROOM_RULES_MAX.get(c, 99)
            if cnt >= cap and c in class_to_id:
                flat[:, class_to_id[c]] = -1
        bi = np.unravel_index(np.argmax(flat), flat.shape)
        nid, cid = int(bi[0]), int(bi[1])
        if flat[nid, cid] < 0:
            break
        cls = ALL_CLASSES[cid]
        selected.append(nid)
        classes.append(cls)
        counts[cls] = counts.get(cls, 0) + 1
        probs[nid] = -1

    n = len(classes)
    with torch.no_grad():
        z = torch.randn(n, 16, device=device)
        _, edge_logits = gcn.decode(z)
        ep = torch.sigmoid(edge_logits).cpu().numpy()

    adj = np.zeros((n, n), dtype=int)

    # Critical edges
    for a, b in CRITICAL_EDGES:
        if a in classes and b in classes:
            i, j = classes.index(a), classes.index(b)
            adj[i][j] = adj[j][i] = 1

    # Top-k learned edges
    for i in range(n):
        s = ep[i].copy()
        s[i] = -1
        for j in np.argsort(s)[::-1][:top_k]:
            if s[j] > edge_threshold:
                adj[i][j] = adj[j][i] = 1

    return classes, adj


def graph_is_feasible(adj: np.ndarray) -> bool:
    """Cheap pre-filter: no node with >4 neighbours, total edges ≤12, no isolates."""
    deg = adj.sum(axis=1)
    if (deg > 4).any():
        return False
    if (deg == 0).any():
        return False
    if int(adj.sum() // 2) > 12:
        return False
    return True


# ── Geometry inference + repair ──────────────────────────────────────────────

def _to_input(room_names, adj):
    import torch
    bundle = get_models()
    n = len(room_names)
    x = torch.zeros(n, len(bundle.classes), device=bundle.device)
    for i, r in enumerate(room_names):
        x[i, bundle.class_to_id.get(r, 0)] = 1.0
    A = torch.tensor(adj, dtype=torch.float32, device=bundle.device)
    return x, A


def predict_geometry(room_names, adj, plot_w: float, plot_h: float):
    import torch
    bundle = get_models()
    x, A = _to_input(room_names, adj)
    with torch.no_grad():
        pred = bundle.geo(x, A).cpu().numpy()
    boxes = []
    for i, name in enumerate(room_names):
        cx, cy, w, h = pred[i]
        mw, mh = MIN_DIMS.get(name, (6, 6))
        rw = max(mw, w * plot_w * 0.7)
        rh = max(mh, h * plot_h * 0.6)
        rx = cx * plot_w - rw / 2
        ry = cy * plot_h - rh / 2
        rx = max(0.5, min(rx, plot_w - rw - 0.5))
        ry = max(0.5, min(ry, plot_h - rh - 0.5))
        boxes.append({"name": name, "x": rx, "y": ry, "w": rw, "h": rh})
    return boxes


def _overlap(a, b):
    return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or
                a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])


def graph_guided_repair(boxes, adj, plot_w, plot_h, iters: int = 600):
    """Two-phase force-directed repair: (1) push apart overlaps,
    (2) pull connected rooms together. Mutates boxes in place."""
    n = len(boxes)
    for it in range(iters):
        moved = 0
        # Phase 1: push apart
        for i in range(n):
            for j in range(i + 1, n):
                a, b = boxes[i], boxes[j]
                if _overlap(a, b):
                    ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                    oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
                    if ox < oy:
                        push = ox / 2 + 0.5
                        if a["x"] < b["x"]:
                            a["x"] -= push; b["x"] += push
                        else:
                            a["x"] += push; b["x"] -= push
                    else:
                        push = oy / 2 + 0.5
                        if a["y"] < b["y"]:
                            a["y"] -= push; b["y"] += push
                        else:
                            a["y"] += push; b["y"] -= push
                    for r in (a, b):
                        r["x"] = max(0.3, min(r["x"], plot_w - r["w"] - 0.3))
                        r["y"] = max(0.3, min(r["y"], plot_h - r["h"] - 0.3))
                    moved += 1

        # Phase 2: pull connected pairs that are far apart
        if it % 5 == 0:
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        continue
                    a, b = boxes[i], boxes[j]
                    dx = (b["x"] + b["w"] / 2) - (a["x"] + a["w"] / 2)
                    dy = (b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)
                    dist = math.hypot(dx, dy)
                    if dist > (a["w"] + b["w"]) / 2 + 4:
                        step = 0.3
                        a["x"] += step * dx / max(dist, 1e-3)
                        a["y"] += step * dy / max(dist, 1e-3)
                        b["x"] -= step * dx / max(dist, 1e-3)
                        b["y"] -= step * dy / max(dist, 1e-3)
                        for r in (a, b):
                            r["x"] = max(0.3, min(r["x"], plot_w - r["w"] - 0.3))
                            r["y"] = max(0.3, min(r["y"], plot_h - r["h"] - 0.3))

        # Shrink offenders periodically
        if moved > 0 and it % 30 == 29:
            for r in boxes:
                r["w"] *= 0.95
                r["h"] *= 0.95
                mw, mh = MIN_DIMS.get(r["name"], (5, 5))
                r["w"] = max(mw, r["w"])
                r["h"] = max(mh, r["h"])

        if moved == 0 and it > 5:
            break
    return boxes


def _adjacency_ok_pair(a, b, tol: float = 4.0) -> bool:
    dx = max(0, max(a["x"], b["x"]) - min(a["x"] + a["w"], b["x"] + b["w"]))
    dy = max(0, max(a["y"], b["y"]) - min(a["y"] + a["h"], b["y"] + b["h"]))
    return math.hypot(dx, dy) <= tol


def check_validity(boxes, adj, plot_w: float, plot_h: float) -> dict:
    n = len(boxes)
    overlaps = sum(1 for i in range(n) for j in range(i + 1, n) if _overlap(boxes[i], boxes[j]))
    no_overlap = overlaps == 0
    inside = all(b["x"] >= 0 and b["y"] >= 0 and
                 b["x"] + b["w"] <= plot_w + 0.1 and
                 b["y"] + b["h"] <= plot_h + 0.1 for b in boxes)
    min_size = all(b["w"] * b["h"] >= MIN_AREA.get(b["name"], 25) - 2 for b in boxes)
    aspect_ok = all(max(b["w"] / b["h"], b["h"] / b["w"]) <= 3.5 for b in boxes)

    needed = satisfied = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                needed += 1
                if _adjacency_ok_pair(boxes[i], boxes[j]):
                    satisfied += 1
    adj_ratio = (satisfied / needed) if needed else 1.0
    adj_ok = adj_ratio >= 0.6

    # connectivity
    if n == 0:
        connected = False
    else:
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for k in range(n):
                if adj[u][k] and k not in seen:
                    seen.add(k); stack.append(k)
        connected = len(seen) == n

    valid = no_overlap and inside and min_size and adj_ok and aspect_ok and connected
    return {
        "valid": valid,
        "no_overlap": no_overlap,
        "inside_boundary": inside,
        "min_size": min_size,
        "adjacency_ok": adj_ok,
        "adjacency_ratio": adj_ratio,
        "aspect_ok": aspect_ok,
        "connected": connected,
    }


# ── Apartment fit + corridor detection (Cell F) ───────────────────────────────

def fit_to_inner_box(boxes, plot_w, plot_h, margin: float = OUTER_MARGIN):
    """Non-uniform x/y scaling so room cluster exactly fills the inner box."""
    if not boxes:
        return boxes
    boxes = copy.deepcopy(boxes)
    xs0 = min(b["x"] for b in boxes)
    ys0 = min(b["y"] for b in boxes)
    xs1 = max(b["x"] + b["w"] for b in boxes)
    ys1 = max(b["y"] + b["h"] for b in boxes)
    cur_w = xs1 - xs0
    cur_h = ys1 - ys0
    inner_w = plot_w - 2 * margin
    inner_h = plot_h - 2 * margin
    sx = inner_w / max(cur_w, 1e-6)
    sy = inner_h / max(cur_h, 1e-6)
    for b in boxes:
        b["x"] = margin + (b["x"] - xs0) * sx
        b["y"] = margin + (b["y"] - ys0) * sy
        b["w"] *= sx
        b["h"] *= sy
    return boxes


def detect_corridors(boxes, plot_w, plot_h, margin: float = OUTER_MARGIN,
                     grid_res: float = 1.0, min_corridor_area: float = 8.0):
    """Flood-fill empty interior cells, mark large empty regions as corridor boxes."""
    inner_x0, inner_y0 = margin, margin
    inner_x1, inner_y1 = plot_w - margin, plot_h - margin
    gx_count = max(1, int(round((inner_x1 - inner_x0) / grid_res)))
    gy_count = max(1, int(round((inner_y1 - inner_y0) / grid_res)))

    occupied = np.zeros((gx_count, gy_count), dtype=bool)
    for b in boxes:
        x0 = max(0, int(math.floor((b["x"] - inner_x0) / grid_res)))
        y0 = max(0, int(math.floor((b["y"] - inner_y0) / grid_res)))
        x1 = min(gx_count, int(math.ceil((b["x"] + b["w"] - inner_x0) / grid_res)))
        y1 = min(gy_count, int(math.ceil((b["y"] + b["h"] - inner_y0) / grid_res)))
        occupied[x0:x1, y0:y1] = True

    visited = np.zeros_like(occupied)
    corridors = []
    for i in range(gx_count):
        for j in range(gy_count):
            if occupied[i, j] or visited[i, j]:
                continue
            # BFS
            stack = [(i, j)]
            cells = []
            while stack:
                cx, cy = stack.pop()
                if 0 <= cx < gx_count and 0 <= cy < gy_count and \
                        not occupied[cx, cy] and not visited[cx, cy]:
                    visited[cx, cy] = True
                    cells.append((cx, cy))
                    stack.extend([(cx + 1, cy), (cx - 1, cy),
                                  (cx, cy + 1), (cx, cy - 1)])
            area = len(cells) * grid_res * grid_res
            if area >= min_corridor_area:
                xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
                bx = inner_x0 + min(xs) * grid_res
                by = inner_y0 + min(ys) * grid_res
                bw = (max(xs) - min(xs) + 1) * grid_res
                bh = (max(ys) - min(ys) + 1) * grid_res
                corridors.append({"name": "internal_corridor",
                                  "x": bx, "y": by, "w": bw, "h": bh})
    return corridors


# ── Full pipeline with rejection sampling ─────────────────────────────────────

def generate_one_plan(target_rooms: int, plot_w: float, plot_h: float,
                      max_attempts: int = 30):
    """Rejection sample one valid plan. Returns (boxes_fitted, adj, room_names,
    validity_dict, attempts). Raises RuntimeError if it can't find one."""
    attempts = 0
    last_result = None
    while attempts < max_attempts:
        attempts += 1
        rooms, adj = generate_graph_v2(target_rooms=target_rooms)
        if not graph_is_feasible(adj):
            continue
        boxes = predict_geometry(rooms, adj, plot_w, plot_h)
        boxes = graph_guided_repair(boxes, adj, plot_w, plot_h, iters=600)
        v = check_validity(boxes, adj, plot_w, plot_h)
        last_result = (boxes, adj, rooms, v, attempts)
        if v["valid"]:
            boxes_fitted = fit_to_inner_box(boxes, plot_w, plot_h)
            return boxes_fitted, adj, rooms, v, attempts
    # If 30 attempts didn't yield a strictly valid plan, return the best we have.
    boxes, adj, rooms, v, _ = last_result
    boxes_fitted = fit_to_inner_box(boxes, plot_w, plot_h)
    return boxes_fitted, adj, rooms, v, attempts


def generate_population(target_rooms: int, plot_w: float, plot_h: float, n: int = 10):
    """Generate n valid (or best-effort) plans."""
    out = []
    for _ in range(n):
        boxes, adj, rooms, v, attempts = generate_one_plan(
            target_rooms=target_rooms, plot_w=plot_w, plot_h=plot_h)
        out.append({
            "boxes": boxes,
            "adj": adj.tolist() if hasattr(adj, "tolist") else adj,
            "rooms": rooms,
            "validity": v,
            "attempts": attempts,
        })
    return out
