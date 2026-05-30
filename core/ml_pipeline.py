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
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ml_models import load_models, models_available
from .constants import ROOM_SIZE_RANGES, CORRIDOR_WIDTH_MIN, CORRIDOR_WIDTH_MAX


# ── Domain constants ──────────────────────────────────────────────────────────

# Non-uniform setbacks per professor's spec:
#   2 ft on the road-facing side, 1 ft on the other three sides
# (north = top of the plot; road assumed on the south unless otherwise stated)
SETBACK_ROAD = 2.0
SETBACK_OTHER = 1.0
ROAD_SIDE = "S"   # 'S','N','E','W' — which side faces the road
OUTER_MARGIN = 1.0      # legacy alias kept for any external callers
DEFAULT_PLOT_W = 40.0
DEFAULT_PLOT_H = 60.0


def get_setbacks(road_side: str = ROAD_SIDE):
    """Return (left, right, bottom, top) setback distances in feet."""
    left = SETBACK_ROAD if road_side == "W" else SETBACK_OTHER
    right = SETBACK_ROAD if road_side == "E" else SETBACK_OTHER
    bottom = SETBACK_ROAD if road_side == "S" else SETBACK_OTHER
    top = SETBACK_ROAD if road_side == "N" else SETBACK_OTHER
    return left, right, bottom, top

MIN_DIMS = {
    "external_staircase": (7, 8),
    "foyer":           (6, 6),
    "master_bedroom":  (12, 12),
    "attached_bathroom": (5, 7),
    "bedroom":         (10, 10),
    "bedroom_01":      (10, 10),
    "bedroom_02":      (10, 10),
    "bedroom_03":      (10, 10),
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

"""Vastu-preferred quadrant for each room class.

Coordinates: x grows east, y grows north. The inner buildable rectangle is
split into 4 quadrants and rooms are placed in the quadrant matching their
Vastu requirement before squarify packs each quadrant.
"""
VASTU_QUADRANT = {
    # SW quadrant (x small, y small) — heavy / private
    "master_bedroom":   "SW",
    "bedroom_01":       "SW",
    "staircase":        "SW",
    "external_staircase":"SW",
    # SE quadrant (x large, y small) — fire / cooking
    "kitchen":          "SE",
    "dining_room":      "SE",
    # NE quadrant (x large, y large) — light / sacred
    "pooja_room":       "NE",
    "living_room":      "NE",
    "balcony":          "NE",
    "study":            "NE",
    "foyer":            "NE",      # main entrance prefers N/NE/E
    # NW quadrant (x small, y large) — water / waste
    "common_bathroom":  "NW",
    "attached_bathroom":"NW",
    "utility_area":     "NW",
    "bedroom_02":       "NW",
    "bedroom_03":       "NW",
    # Generic
    "bedroom":          "SW",
}


def fit_to_inner_box(boxes, plot_w, plot_h, road_side: str = ROAD_SIDE,
                      margin: float | None = None):
    """Tile the inner rectangle 100% with rooms — no gaps, no overlaps.

    Rooms are grouped by their Vastu-preferred quadrant (NE/NW/SE/SW), then
    each quadrant is filled by a squarified treemap. This guarantees:
      - the entire inner box is covered (Space Usage = 100%)
      - kitchen lands in SE, master_bedroom in SW, etc. (high Vastu score)

    Setbacks are non-uniform: 2 ft on `road_side`, 1 ft on the other 3 sides.
    Passing `margin=` (legacy) overrides this with a uniform value.
    """
    if not boxes:
        return boxes
    boxes = copy.deepcopy(boxes)

    if margin is not None:
        left = right = bottom = top = margin
    else:
        left, right, bottom, top = get_setbacks(road_side)

    inner_x0 = left
    inner_y0 = bottom
    inner_w = plot_w - left - right
    inner_h = plot_h - bottom - top
    inner_area = inner_w * inner_h

    # 1. Compute target areas proportional to AI predictions
    weights = [max(b["w"] * b["h"], 1.0) for b in boxes]
    total_w = sum(weights)
    target_areas = [w * inner_area / total_w for w in weights]

    # 2. Group rooms by Vastu quadrant
    quadrant_buckets = {"NE": [], "NW": [], "SE": [], "SW": []}
    for i, b in enumerate(boxes):
        q = VASTU_QUADRANT.get(b["name"], "SW")
        quadrant_buckets[q].append((i, b["name"], target_areas[i]))

    # 3. Quadrant rectangles (split inner box 50/50 each way)
    half_w = inner_w / 2
    half_h = inner_h / 2
    quad_rects = {
        "SW": (inner_x0,           inner_y0,           half_w, half_h),
        "SE": (inner_x0 + half_w,  inner_y0,           half_w, half_h),
        "NW": (inner_x0,           inner_y0 + half_h,  half_w, half_h),
        "NE": (inner_x0 + half_w,  inner_y0 + half_h,  half_w, half_h),
    }

    # 4. If a quadrant is empty, donate its area to a non-empty neighbour
    # (otherwise we waste space and fail the 100% fill target).
    empty = [q for q, items in quadrant_buckets.items() if not items]
    for q in empty:
        donor = max(quadrant_buckets, key=lambda k: sum(a for _, _, a in quadrant_buckets[k]))
        if donor == q:
            continue
        qx, qy, qw, qh = quad_rects[q]
        dx, dy, dw, dh = quad_rects[donor]
        # Merge q's rectangle into donor's by extending along the shared edge
        if q[0] == donor[0]:        # same vertical half (e.g. SW+NW or SE+NE)
            quad_rects[donor] = (min(dx, qx), min(dy, qy), dw, dh + qh)
        elif q[1] == donor[1]:      # same horizontal half (e.g. SW+SE or NW+NE)
            quad_rects[donor] = (min(dx, qx), min(dy, qy), dw + qw, dh)
        else:
            # diagonal — just give donor everything (rare)
            quad_rects[donor] = (inner_x0, inner_y0, inner_w, inner_h)
        quad_rects[q] = (0, 0, 0, 0)

    # 5. Rescale each quadrant's room areas to exactly match its rectangle
    out = [None] * len(boxes)
    for q, items in quadrant_buckets.items():
        if not items:
            continue
        qx, qy, qw, qh = quad_rects[q]
        if qw <= 0 or qh <= 0:
            continue
        target_quad_area = qw * qh
        items_area = sum(a for _, _, a in items)
        scale = target_quad_area / max(items_area, 1e-6)
        # squarify wants descending order
        items_sorted = sorted(items, key=lambda t: -t[2])
        sizes = [a * scale for _, _, a in items_sorted]
        names = [n for _, n, _ in items_sorted]
        placed = _squarify(sizes, qx, qy, qw, qh)
        for k, (orig_idx, _, _) in enumerate(items_sorted):
            x, y, w, h = placed[k]
            out[orig_idx] = {"name": names[k], "x": x, "y": y, "w": w, "h": h}

    # Any rooms that somehow ended up unplaced (shouldn't happen) — drop them
    return [b for b in out if b is not None]


def _squarify(sizes, x, y, w, h):
    """Squarified treemap — returns list of (x, y, w, h) in input order."""
    rects = [None] * len(sizes)
    _squarify_rec(list(enumerate(sizes)), x, y, w, h, rects)
    return rects


def _squarify_rec(items, x, y, w, h, rects):
    """items = [(orig_idx, area), ...] sorted descending by area."""
    if not items:
        return
    if len(items) == 1:
        idx, _ = items[0]
        rects[idx] = (x, y, w, h)
        return

    total = sum(a for _, a in items)
    # Pick the row that improves aspect ratio
    row = []
    row_sum = 0.0
    best_score = float("inf")
    split_at = 1
    for k in range(1, len(items) + 1):
        row = items[:k]
        row_sum = sum(a for _, a in row)
        # Available short side
        short = min(w, h)
        long_side = max(w, h)
        if total <= 0:
            break
        row_long = row_sum / total * long_side
        # Worst aspect ratio in this row
        worst = max(
            max((row_long * row_long) / (a * long_side / row_sum + 1e-9),
                (a * long_side / row_sum + 1e-9) / (row_long * row_long + 1e-9))
            for _, a in row
        ) if short > 0 else float("inf")
        if worst < best_score:
            best_score = worst
            split_at = k
        else:
            break

    row = items[:split_at]
    rest = items[split_at:]
    row_sum = sum(a for _, a in row)

    # Lay out this row along the shorter side
    if w >= h:
        # vertical strip on the left of width row_w
        row_w = row_sum / total * w
        cy = y
        for orig_idx, area in row:
            rh = area / row_w if row_w > 0 else 0
            rects[orig_idx] = (x, cy, row_w, rh)
            cy += rh
        _squarify_rec(rest, x + row_w, y, w - row_w, h, rects)
    else:
        # horizontal strip on the bottom of height row_h
        row_h = row_sum / total * h
        cx = x
        for orig_idx, area in row:
            rw = area / row_h if row_h > 0 else 0
            rects[orig_idx] = (cx, y, rw, row_h)
            cx += rw
        _squarify_rec(rest, x, y + row_h, w, h - row_h, rects)


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

def _filter_rooms_to_spec(rooms_pred, adj_pred, required_rooms, optional_rooms):
    """Force the AI's room list to match the user's spec.

    `required_rooms` always go in (even if the VAE didn't pick them — we add
    placeholders). `optional_rooms` are also forced when selected by the user.
    Anything the VAE invented outside both lists is dropped. Returns
    (rooms, adj_matrix).
    """
    import numpy as np
    keep_idx = []
    rooms_out = []
    seen = set()
    # Pass 1: keep VAE-predicted rooms that are in required ∪ optional, no dupes
    allowed = set(required_rooms) | set(optional_rooms)
    for i, r in enumerate(rooms_pred):
        if r in allowed and r not in seen:
            keep_idx.append(i)
            rooms_out.append(r)
            seen.add(r)
    # Pass 2: append any requested rooms still missing (as synthetic nodes)
    requested = list(required_rooms) + list(optional_rooms)
    missing = [r for r in requested if r not in seen]
    rooms_out.extend(missing)

    n_old = len(rooms_pred)
    n_new = len(rooms_out)
    new_adj = np.zeros((n_new, n_new), dtype=int)
    # Copy adjacencies for kept nodes
    for a, oa in enumerate(keep_idx):
        for b, ob in enumerate(keep_idx):
            new_adj[a][b] = adj_pred[oa][ob]
    # Wire each newly-added missing room to the foyer/living_room as default
    n_kept = len(keep_idx)
    for k, r in enumerate(missing):
        idx = n_kept + k
        for hub in ("foyer", "living_room"):
            if hub in rooms_out:
                h = rooms_out.index(hub)
                new_adj[idx][h] = new_adj[h][idx] = 1
                break
    return rooms_out, new_adj


def _resolve_residual_overlaps(boxes, plot_w, plot_h, max_passes: int = 80):
    """Final guarantee that the RAW boxes don't visually overlap.

    Force-directed repair sometimes oscillates around minimum sizes. This
    pass shrinks the smaller of each overlapping pair on the dominant axis
    just enough to remove the overlap, then nudges it inside the plot.
    """
    n = len(boxes)
    for _ in range(max_passes):
        any_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = boxes[i], boxes[j]
                if not _overlap(a, b):
                    continue
                any_overlap = True
                ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
                # shrink the smaller-area box along the cheaper axis
                target = a if a["w"] * a["h"] <= b["w"] * b["h"] else b
                if ox <= oy:
                    target["w"] = max(2.0, target["w"] - ox - 0.2)
                else:
                    target["h"] = max(2.0, target["h"] - oy - 0.2)
                target["x"] = max(0.3, min(target["x"], plot_w - target["w"] - 0.3))
                target["y"] = max(0.3, min(target["y"], plot_h - target["h"] - 0.3))
        if not any_overlap:
            break
    return boxes


def _pin_staircase_to_corner(boxes, plot_w, plot_h):
    """Make sure the staircase sits on a plot edge (SW corner preferred).

    If the squarified layout placed `external_staircase`/`staircase` in the
    interior, swap its rectangle with the room currently nearest the SW
    corner. The swap preserves the 100% tiling.
    """
    if not boxes:
        return boxes
    stair_idx = next((i for i, b in enumerate(boxes)
                      if b["name"] in ("external_staircase", "staircase")), None)
    if stair_idx is None:
        return boxes
    s = boxes[stair_idx]
    touches_edge = (s["x"] <= 1.5 or s["y"] <= 1.5 or
                    s["x"] + s["w"] >= plot_w - 1.5 or
                    s["y"] + s["h"] >= plot_h - 1.5)
    if touches_edge:
        return boxes
    # Find the room closest to the SW corner (0,0) by its own corner
    best_j, best_d = None, float("inf")
    for j, b in enumerate(boxes):
        if j == stair_idx:
            continue
        d = b["x"] * b["x"] + b["y"] * b["y"]
        if d < best_d:
            best_d, best_j = d, j
    if best_j is None:
        return boxes
    # Swap rectangles (keep names with their original positions in the list)
    a, b = boxes[stair_idx], boxes[best_j]
    a["x"], b["x"] = b["x"], a["x"]
    a["y"], b["y"] = b["y"], a["y"]
    a["w"], b["w"] = b["w"], a["w"]
    a["h"], b["h"] = b["h"], a["h"]
    return boxes


# ── Perimeter-ring rule-based layout ─────────────────────────────────────────
# Rule: every room must sit on an outer wall. The plan is laid out as four
# perimeter strips (S=road, W, N, E) wrapped around a central corridor
# rectangle. Rooms are routed to a strip by Vastu zone, then each strip is
# filled proportionally to ROOM_SIZE_RANGES. Output is non-overlapping and
# tiles the inner buildable rectangle 100%.

# Three structurally different perimeter-ring variants. Each is a
# (room_side_map, side_order_map) pair, all still Vastu-correct on the
# critical corners (kitchen SE, master SW, pooja NE, staircase road-edge).
#
# Variant 0 — Living on East, all bedrooms on West (compact private wing)
# Variant 1 — Living on North (north-facing windows), bedrooms split E/W
# Variant 2 — Dining on South (open dining near entry), living on East middle,
#             bedrooms compacted to W and N
_LAYOUT_VARIANTS = [
    # ── Variant 0 ──
    {
        "side": {
            "external_staircase": "S", "foyer": "S", "balcony": "S",
            "kitchen": "E", "dining_room": "E", "living_room": "E",
            "pooja_room": "N", "study": "N", "bedroom_03": "N",
            "master_bedroom": "W", "attached_bathroom": "W",
            "bedroom_01": "W", "bedroom_02": "W",
            "common_bathroom": "W", "utility_area": "W",
        },
        "order": {
            "S": ["external_staircase", "foyer", "balcony"],
            "E": ["kitchen", "dining_room", "living_room"],
            "N": ["bedroom_03", "study", "pooja_room"],
            "W": ["master_bedroom", "attached_bathroom", "bedroom_01",
                  "bedroom_02", "common_bathroom", "utility_area"],
        },
        "south_h":  (7.5, 9.0),
        "north_h":  (8.0, 10.0),
        "west_frac":(0.46, 0.54),
    },
    # ── Variant 1 ──   Living moves North, bedrooms split E/W
    {
        "side": {
            "external_staircase": "S", "foyer": "S", "balcony": "S",
            "kitchen": "E", "dining_room": "E",
            "bedroom_02": "E", "bedroom_03": "E", "study": "E",
            "living_room": "N", "pooja_room": "N",
            "master_bedroom": "W", "attached_bathroom": "W",
            "bedroom_01": "W",
            "common_bathroom": "W", "utility_area": "W",
        },
        "order": {
            "S": ["external_staircase", "foyer", "balcony"],
            "E": ["kitchen", "dining_room", "bedroom_02",
                  "study", "bedroom_03"],
            "N": ["living_room", "pooja_room"],
            "W": ["master_bedroom", "attached_bathroom", "bedroom_01",
                  "common_bathroom", "utility_area"],
        },
        "south_h":  (7.0, 8.5),
        "north_h":  (11.0, 13.5),    # taller north strip so living fits
        "west_frac":(0.38, 0.44),    # west thinner, east wider
    },
    # ── Variant 2 ──   Dining + kitchen south near entry, living mid-east
    {
        "side": {
            "external_staircase": "S", "foyer": "S",
            "dining_room": "S", "utility_area": "S",
            "kitchen": "E", "living_room": "E", "balcony": "E",
            "pooja_room": "N", "study": "N", "bedroom_03": "N",
            "master_bedroom": "W", "attached_bathroom": "W",
            "bedroom_01": "W", "bedroom_02": "W",
            "common_bathroom": "W",
        },
        "order": {
            "S": ["external_staircase", "foyer", "dining_room", "utility_area"],
            "E": ["kitchen", "living_room", "balcony"],
            "N": ["bedroom_03", "study", "pooja_room"],
            "W": ["master_bedroom", "attached_bathroom", "bedroom_01",
                  "bedroom_02", "common_bathroom"],
        },
        "south_h":  (9.5, 11.5),     # deeper south strip — fits dining
        "north_h":  (7.5, 9.5),
        "west_frac":(0.50, 0.58),    # west slightly wider
    },
]


def _room_target_size(name: str, rng: random.Random):
    (wlo, whi), (hlo, hhi) = ROOM_SIZE_RANGES.get(name, ((8, 11), (8, 11)))
    return rng.uniform(wlo, whi), rng.uniform(hlo, hhi)


def perimeter_ring_layout(rooms, plot_w, plot_h, road_side: str = ROAD_SIDE,
                          seed: int = 0):
    """Rule-based layout — every room sits on an external wall, central
    corridor rectangle. Built for road on the south; the result is rotated
    afterwards if the road is on a different side.

    Layout is a four-strip ring:
        +------------------------+
        | N strip                |
        +----+--------------+----+
        |    |              |    |
        | W  |   CORRIDOR   |  E |
        |    |              |    |
        +----+--------------+----+
        | S strip (road)         |
        +------------------------+
    """
    rng = random.Random(seed)
    rooms = list(rooms)
    if not rooms:
        return []

    # Pick one of the structural variants based on seed
    variant = _LAYOUT_VARIANTS[seed % len(_LAYOUT_VARIANTS)]
    room_side = variant["side"]
    side_order = variant["order"]

    left, right, bottom, top = get_setbacks("S")
    ix0, iy0 = left, bottom
    iw = plot_w - left - right
    ih = plot_h - bottom - top

    # 1. Bucket rooms by side (fallback to W for anything not mapped)
    side_rooms = {"S": [], "E": [], "N": [], "W": []}
    for r in rooms:
        side_rooms[room_side.get(r, "W")].append(r)

    # Sort each bucket by the variant's flow order
    for s in side_rooms:
        ordering = side_order.get(s, [])
        side_rooms[s].sort(key=lambda n: ordering.index(n) if n in ordering else 99)

    # 2. Pick strip thicknesses from the variant's ranges
    sh_lo, sh_hi = variant["south_h"]
    nh_lo, nh_hi = variant["north_h"]
    south_h = rng.uniform(sh_lo, sh_hi)
    north_h = rng.uniform(nh_lo, nh_hi) if side_rooms["N"] else 0.0
    corridor_w = rng.uniform(CORRIDOR_WIDTH_MIN, CORRIDOR_WIDTH_MAX)

    middle_h = ih - south_h - north_h
    if middle_h < 14.0:
        south_h = max(6.5, south_h * 0.85)
        north_h = max(0.0, north_h * 0.7)
        middle_h = ih - south_h - north_h

    # West / East column widths
    side_total_w = iw - corridor_w
    if not side_rooms["W"]:
        west_w, east_w = 0.0, side_total_w
    elif not side_rooms["E"]:
        west_w, east_w = side_total_w, 0.0
    else:
        wf_lo, wf_hi = variant["west_frac"]
        west_w = side_total_w * rng.uniform(wf_lo, wf_hi)
        east_w = side_total_w - west_w

    west_x = ix0
    corr_x = ix0 + west_w
    east_x = corr_x + corridor_w
    middle_y = iy0 + south_h
    north_y = middle_y + middle_h

    placed = []

    # 3. SOUTH strip — full inner width, room widths proportional to targets
    if side_rooms["S"]:
        _fill_horizontal(side_rooms["S"], ix0, iy0, iw, south_h, rng, placed)

    # 4. NORTH strip
    if side_rooms["N"] and north_h > 0:
        _fill_horizontal(side_rooms["N"], ix0, north_y, iw, north_h, rng, placed)

    # 5. WEST strip (vertical, bottom→top)
    if side_rooms["W"]:
        _fill_vertical(side_rooms["W"], west_x, middle_y, west_w, middle_h,
                       rng, placed)

    # 6. EAST strip
    if side_rooms["E"]:
        _fill_vertical(side_rooms["E"], east_x, middle_y, east_w, middle_h,
                       rng, placed)

    # 7. Central corridor (only piece NOT on an external wall by design)
    if corridor_w > 0 and middle_h > 1:
        placed.append({"name": "internal_corridor",
                       "x": corr_x, "y": middle_y,
                       "w": corridor_w, "h": middle_h})

    # 8. Rotate the layout for non-south road sides
    if road_side != "S":
        placed = _rotate_layout(placed, plot_w, plot_h, road_side)

    return placed


def _fill_horizontal(names, x0, y0, total_w, strip_h, rng, placed):
    """Tile a horizontal strip with rooms side-by-side, exactly filling total_w."""
    targets = []
    for n in names:
        w, _ = _room_target_size(n, rng)
        targets.append(max(3.0, w))
    s = sum(targets)
    if s <= 0:
        return
    scale = total_w / s
    cx = x0
    for n, t in zip(names, targets):
        rw = t * scale
        placed.append({"name": n, "x": cx, "y": y0, "w": rw, "h": strip_h})
        cx += rw
    if placed:
        # snap last room's right edge so we don't drift due to float
        placed[-1]["w"] = (x0 + total_w) - placed[-1]["x"]


def _fill_vertical(names, x0, y0, strip_w, total_h, rng, placed):
    """Tile a vertical strip with rooms stacked bottom→top, filling total_h."""
    targets = []
    for n in names:
        _, h = _room_target_size(n, rng)
        # services / bathrooms / pooja stay short
        if n in ("attached_bathroom", "common_bathroom", "utility_area",
                 "pooja_room"):
            h *= 0.75
        targets.append(max(3.0, h))
    s = sum(targets)
    if s <= 0:
        return
    scale = total_h / s
    cy = y0
    for n, t in zip(names, targets):
        rh = t * scale
        placed.append({"name": n, "x": x0, "y": cy, "w": strip_w, "h": rh})
        cy += rh
    if placed:
        placed[-1]["h"] = (y0 + total_h) - placed[-1]["y"]


def _rotate_layout(boxes, plot_w, plot_h, road_side):
    """Rotate a south-road layout so the entry strip ends up on `road_side`."""
    out = []
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        if road_side == "N":
            nx, ny, nw, nh = plot_w - x - w, plot_h - y - h, w, h
        elif road_side == "E":
            nx, ny, nw, nh = plot_w - y - h, x, h, w
        elif road_side == "W":
            nx, ny, nw, nh = y, plot_h - x - w, h, w
        else:
            nx, ny, nw, nh = x, y, w, h
        out.append({"name": b["name"], "x": nx, "y": ny, "w": nw, "h": nh})
    return out


def generate_one_plan(plot_w: float, plot_h: float,
                       required_rooms=None, optional_rooms=None,
                       road_side: str = ROAD_SIDE,
                       max_attempts: int = 30,
                       seed: int = 0):
    """Returns one (raw, optimised) plan pair.

    * RAW       — AI's predicted geometry, repaired to be non-overlapping.
    * OPTIMISED — perimeter-ring rule-based layout: every room touches an
                  outer wall, a central corridor rectangle is inserted,
                  staircase/foyer/balcony are anchored to the road edge.
    """
    required_rooms = list(required_rooms or [])
    optional_rooms = list(optional_rooms or [])
    target_rooms = max(4, min(12, len(required_rooms) + len(optional_rooms)))

    attempts = 0
    last_result = None
    while attempts < max_attempts:
        attempts += 1
        rooms_pred, adj_pred = generate_graph_v2(target_rooms=target_rooms)
        rooms, adj = _filter_rooms_to_spec(
            rooms_pred, adj_pred, required_rooms, optional_rooms)
        boxes_raw = predict_geometry(rooms, adj, plot_w, plot_h)
        boxes_repaired = graph_guided_repair(
            copy.deepcopy(boxes_raw), adj, plot_w, plot_h, iters=600)
        v = check_validity(boxes_repaired, adj, plot_w, plot_h)
        last_result = (boxes_raw, boxes_repaired, adj, rooms, v, attempts)
        if v["valid"]:
            break
    boxes_raw, boxes_repaired, adj, rooms, v, attempts = last_result
    boxes_repaired = _resolve_residual_overlaps(
        boxes_repaired, plot_w, plot_h)
    boxes_optimised = perimeter_ring_layout(
        rooms, plot_w, plot_h, road_side=road_side, seed=seed)
    return {
        "raw_boxes": boxes_repaired,       # repaired (non-overlapping) AI output
        "boxes": boxes_optimised,          # perimeter ring + corridor
        "adj": adj.tolist() if hasattr(adj, "tolist") else adj,
        "rooms": rooms,
        "validity": v,
        "attempts": attempts,
    }


def generate_population(plot_w: float, plot_h: float, n: int = 10,
                         required_rooms=None, optional_rooms=None,
                         road_side: str = ROAD_SIDE):
    """Generate n valid (or best-effort) plans with distinct seeds."""
    return [generate_one_plan(
                plot_w=plot_w, plot_h=plot_h,
                required_rooms=required_rooms,
                optional_rooms=optional_rooms,
                road_side=road_side,
                seed=i + 1) for i in range(n)]
