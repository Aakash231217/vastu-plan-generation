import numpy as np
import random
import math
from core.constants import (ROOM_MIN_AREA, ROOM_MAX_AREA, REQUIRED_ADJACENCIES,
                             VASTU_RULES, COST_PER_SQFT, PLUMBING_COST_PER_METER,
                             WALL_COST_PER_METER, CLIMATE_ZONES)


class Room:
    def __init__(self, name, x, y, w, h):
        self.name = name
        self.x = x      # left edge (ft)
        self.y = y      # bottom edge (ft)
        self.w = w      # width (ft)
        self.h = h      # height (ft)
        self.area = w * h

    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2
    @property
    def right(self): return self.x + self.w
    @property
    def top(self): return self.y + self.h

    def zone(self, plot_w, plot_h):
        """Return compass zone of room centre relative to plot."""
        cx_n = self.cx / plot_w   # 0-1
        cy_n = self.cy / plot_h   # 0-1 (0=south, 1=north)
        if cx_n < 0.33:
            h = "W"
        elif cx_n > 0.66:
            h = "E"
        else:
            h = "C"
        if cy_n > 0.66:
            v = "N"
        elif cy_n < 0.33:
            v = "S"
        else:
            v = "M"
        zone_map = {
            ("W","N"): "NW", ("C","N"): "N", ("E","N"): "NE",
            ("W","M"): "W",  ("C","M"): "Centre", ("E","M"): "E",
            ("W","S"): "SW", ("C","S"): "S", ("E","S"): "SE",
        }
        return zone_map.get((h, v), "Centre")

    def shares_wall(self, other, tol=1.5):
        """Check if two rooms share a wall."""
        h_overlap = (self.x < other.right + tol) and (other.x < self.right + tol)
        v_overlap = (self.y < other.top + tol) and (other.y < self.top + tol)
        touching_h = abs(self.right - other.x) < tol or abs(other.right - self.x) < tol
        touching_v = abs(self.top - other.y) < tol or abs(other.top - self.y) < tol
        return (touching_h and v_overlap) or (touching_v and h_overlap)

    def distance_to(self, other):
        return math.sqrt((self.cx - other.cx)**2 + (self.cy - other.cy)**2)

    def overlaps(self, other, tol=0.5):
        return (self.x + tol < other.right and other.x + tol < self.right and
                self.y + tol < other.top  and other.y + tol < self.top)


class FloorPlan:
    def __init__(self, rooms, plot_w, plot_h):
        self.rooms = rooms          # list of Room objects
        self.plot_w = plot_w
        self.plot_h = plot_h
        self.scores = {}

    def get_room(self, name):
        for r in self.rooms:
            if r.name == name:
                return r
        return None


def generate_floorplan(room_names, plot_w, plot_h, seed=None):
    """
    Rule-guided rectangle packing floor plan generator.
    Places rooms using priority order and adjacency hints.
    Returns a FloorPlan object.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Sort by size priority: living first, then bedrooms, kitchen, others
    priority_order = [
        "living_room", "dining_room", "kitchen", "master_bedroom",
        "bedroom_01", "bedroom_02", "bedroom_03", "foyer",
        "attached_bathroom", "common_bathroom", "pooja_room",
        "balcony", "utility_area", "study", "powder_room", "internal_corridor"
    ]
    sorted_rooms = sorted(room_names,
        key=lambda r: priority_order.index(r) if r in priority_order else 99)

    placed = []
    MARGIN = 1.0  # ft wall thickness

    # Compute target area per room (scaled to fit plot)
    plot_area = plot_w * plot_h
    usable_area = plot_area * 0.85  # 85% usable
    total_min = sum(ROOM_MIN_AREA.get(r, 80) for r in sorted_rooms)
    scale = min(usable_area / total_min, 1.6)

    def target_dims(room_name):
        min_a = ROOM_MIN_AREA.get(room_name, 80) * scale
        # Add randomness for variety
        area = min_a * random.uniform(1.0, 1.3)
        # Prefer aspect ratios between 1:1 and 1:2
        aspect = random.uniform(1.0, 1.8)
        w = math.sqrt(area * aspect)
        h = area / w
        w = max(w, 8.0)   # min 8ft
        h = max(h, 8.0)
        w = min(w, plot_w - 2 * MARGIN)
        h = min(h, plot_h - 2 * MARGIN)
        return round(w, 1), round(h, 1)

    def find_position(room_name, w, h):
        """Try to place room adjacent to a preferred neighbour, else grid pack."""
        preferred_neighbours = [a[1] if a[0]==room_name else
                                 (a[0] if a[1]==room_name else None)
                                 for a in REQUIRED_ADJACENCIES]
        preferred_neighbours = [n for n in preferred_neighbours if n]

        # Try placing next to placed preferred neighbours
        candidates = []
        for pn in preferred_neighbours:
            pr = next((r for r in placed if r.name == pn), None)
            if pr:
                # Try right of pr
                for dx, dy in [(pr.w+MARGIN, 0), (0, pr.h+MARGIN),
                                (-w-MARGIN, 0), (0, -h-MARGIN)]:
                    nx = pr.x + dx
                    ny = pr.y + dy
                    nx = max(MARGIN, min(nx, plot_w - w - MARGIN))
                    ny = max(MARGIN, min(ny, plot_h - h - MARGIN))
                    trial = Room(room_name, nx, ny, w, h)
                    if all(not trial.overlaps(p) for p in placed):
                        candidates.append((nx, ny, 0))  # 0 = preferred

        if candidates:
            candidates.sort(key=lambda c: c[2])
            return candidates[0][0], candidates[0][1]

        # Fallback: grid scan
        step = 2.0
        best = None
        best_score = float('inf')
        for gx in np.arange(MARGIN, plot_w - w - MARGIN + 0.1, step):
            for gy in np.arange(MARGIN, plot_h - h - MARGIN + 0.1, step):
                trial = Room(room_name, gx, gy, w, h)
                if all(not trial.overlaps(p) for p in placed):
                    # Score by distance to plot centre (pack centrally)
                    cx_dist = abs(gx + w/2 - plot_w/2)
                    cy_dist = abs(gy + h/2 - plot_h/2)
                    score = cx_dist + cy_dist + random.uniform(0, 3)
                    if score < best_score:
                        best_score = score
                        best = (gx, gy)
        if best:
            return best
        # Last resort: random
        return (random.uniform(MARGIN, max(MARGIN, plot_w-w-MARGIN)),
                random.uniform(MARGIN, max(MARGIN, plot_h-h-MARGIN)))

    for room_name in sorted_rooms:
        w, h = target_dims(room_name)
        x, y = find_position(room_name, w, h)
        placed.append(Room(room_name, round(x,1), round(y,1), round(w,1), round(h,1)))

    return FloorPlan(placed, plot_w, plot_h)


# ── Scoring Functions ──────────────────────────────────────────────────────────

def score_adjacency_quality(fp):
    """AQ — 0 to 1. Higher is better."""
    room_map = {r.name: r for r in fp.rooms}
    total_weight = 0
    satisfied_weight = 0
    for r1, r2, w in REQUIRED_ADJACENCIES:
        if r1 in room_map and r2 in room_map:
            total_weight += w
            room_a, room_b = room_map[r1], room_map[r2]
            if room_a.shares_wall(room_b):
                satisfied_weight += w
            else:
                d = room_a.distance_to(room_b)
                satisfied_weight += w * max(0, 1 - d / (fp.plot_w * 0.5))
    return round(satisfied_weight / total_weight, 3) if total_weight > 0 else 0.0


def score_vastu(fp, selected_rules=None):
    """VS — 0 to 1. Higher is better."""
    room_map = {r.name: r for r in fp.rooms}
    total_w = 0
    satisfied_w = 0
    details = []
    for rule_name, room_name, good_zones, weight in VASTU_RULES:
        if selected_rules and rule_name not in selected_rules:
            continue
        if room_name not in room_map:
            continue
        total_w += weight
        zone = room_map[room_name].zone(fp.plot_w, fp.plot_h)
        if zone in good_zones:
            satisfied_w += weight
            details.append((rule_name, room_name, zone, True))
        elif any(z in zone for z in good_zones):
            satisfied_w += weight * 0.5
            details.append((rule_name, room_name, zone, "partial"))
        else:
            details.append((rule_name, room_name, zone, False))
    score = round(satisfied_w / total_w, 3) if total_w > 0 else 0.0
    return score, details


def score_spatial_efficiency(fp):
    """SE — 0 to 1."""
    total_room_area = sum(r.area for r in fp.rooms)
    plot_area = fp.plot_w * fp.plot_h
    area_util = min(total_room_area / plot_area, 1.0)
    # Graph connectivity — all rooms reachable via shared walls?
    connected = 0
    for r in fp.rooms:
        for other in fp.rooms:
            if r != other and r.shares_wall(other):
                connected += 1
                break
    connectivity = connected / len(fp.rooms) if fp.rooms else 0
    se = 0.5 * area_util + 0.5 * connectivity
    return round(se, 3)


def score_layout_compactness(fp):
    """LC — 0 to 1."""
    if not fp.rooms:
        return 0.0
    total_area = sum(r.area for r in fp.rooms)
    # Perimeter of bounding box of all rooms
    min_x = min(r.x for r in fp.rooms)
    max_x = max(r.right for r in fp.rooms)
    min_y = min(r.y for r in fp.rooms)
    max_y = max(r.top for r in fp.rooms)
    bbox_area = (max_x - min_x) * (max_y - min_y)
    bbox_perim = 2 * ((max_x - min_x) + (max_y - min_y))
    compactness = (4 * math.pi * bbox_area) / (bbox_perim ** 2) if bbox_perim > 0 else 0
    fill = total_area / bbox_area if bbox_area > 0 else 0
    lc = 0.4 * compactness + 0.6 * fill
    return round(min(lc, 1.0), 3)


def estimate_cost(fp, finishing="Standard"):
    """Returns estimated construction cost in ₹."""
    total_area_sqft = sum(r.area for r in fp.rooms)
    base_cost = total_area_sqft * COST_PER_SQFT[finishing]

    # Wall length cost
    total_wall = 0
    for r in fp.rooms:
        total_wall += 2 * (r.w + r.h)
    # Subtract shared walls (save ~30% for shared)
    shared = 0
    for i, r1 in enumerate(fp.rooms):
        for r2 in fp.rooms[i+1:]:
            if r1.shares_wall(r2):
                shared += min(r1.w, r2.w, r1.h, r2.h)
    wall_cost = (total_wall - shared * 0.6) * WALL_COST_PER_METER * 0.3048  # ft to m

    # Plumbing cost — distance between wet areas
    wet_rooms = [r for r in fp.rooms if any(w in r.name for w in ["bathroom","kitchen","utility"])]
    plumbing_cost = 0
    if len(wet_rooms) > 1:
        for i in range(len(wet_rooms)-1):
            d = wet_rooms[i].distance_to(wet_rooms[i+1]) * 0.3048  # ft to m
            plumbing_cost += d * PLUMBING_COST_PER_METER

    total = base_cost + wall_cost + plumbing_cost
    return round(total)


def score_area_compliance(fp, room_names):
    """AC — 0 to 1."""
    scores = []
    for r in fp.rooms:
        min_a = ROOM_MIN_AREA.get(r.name, 80)
        max_a = ROOM_MAX_AREA.get(r.name, 200)
        if r.area < min_a:
            dev = (min_a - r.area) / min_a
        elif r.area > max_a:
            dev = (r.area - max_a) / max_a
        else:
            dev = 0
        scores.append(max(0, 1 - dev))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def climate_suggestions(fp, climate_zone):
    """Returns list of climate suggestion strings."""
    zone_data = CLIMATE_ZONES.get(climate_zone, {})
    suggestions = []
    wwr = zone_data.get("wwr", 0.20)
    buf = zone_data.get("buffer", "W")

    suggestions.append(f"Target Window-to-Wall Ratio: {int(wwr*100)}% "
                       f"({'smaller windows reduce heat gain' if wwr < 0.25 else 'larger windows improve daylight'})")
    suggestions.append(f"Place utility/store rooms on {buf}-facing walls as thermal buffer zones")
    suggestions.append("Ensure cross-ventilation: living room and kitchen should have openings on opposite walls")
    suggestions.append("Orient long axis of building East-West to minimise afternoon sun exposure")
    if "Hot" in climate_zone:
        suggestions.append("Use high thermal mass materials (brick/concrete) for South and West walls")
        suggestions.append("Consider roof overhang depth of 0.6–0.9m on South-facing windows")
    elif "Cold" in climate_zone:
        suggestions.append("Maximise South-facing glazing for passive solar heat gain")
        suggestions.append("Use double-glazed windows on North-facing rooms")
    return suggestions


def _ml_boxes_to_floorplan(boxes, plot_w, plot_h):
    """Convert a list of ML pipeline boxes into a FloorPlan with Room objects."""
    rooms = []
    for b in boxes:
        rooms.append(Room(b["name"],
                          round(float(b["x"]), 2),
                          round(float(b["y"]), 2),
                          round(float(b["w"]), 2),
                          round(float(b["h"]), 2)))
    return FloorPlan(rooms, plot_w, plot_h)


def generate_population(room_names, plot_w, plot_h, n=20,
                          required_rooms=None, optional_rooms=None,
                          road_side="S"):
    """Generate n candidate plans.

    Returns a list of (raw_fp, optimised_fp) tuples so the UI can compare the
    AI's untouched output against the Vastu-optimised + space-tiled version.
    Falls back to the rule-based packer if the trained weights aren't present.
    """
    # Try the trained-model pipeline
    try:
        from core import ml_pipeline
        if ml_pipeline.is_ready():
            ml_plans = ml_pipeline.generate_population(
                plot_w=plot_w, plot_h=plot_h, n=n,
                required_rooms=required_rooms,
                optional_rooms=optional_rooms,
                road_side=road_side)
            return [(_ml_boxes_to_floorplan(p["raw_boxes"], plot_w, plot_h),
                     _ml_boxes_to_floorplan(p["boxes"], plot_w, plot_h))
                    for p in ml_plans]
    except Exception as e:
        print(f"[generator] ML pipeline unavailable, falling back to rules: {e}")

    plans = []
    for i in range(n):
        fp = generate_floorplan(room_names, plot_w, plot_h, seed=i * 7 + 42)
        plans.append((fp, fp))   # no optimisation in fallback
    return plans


def score_space_usage(fp, road_side: str = "S") -> float:
    """SU — fraction of the buildable (inner-setback) area actually covered
    by rooms. 1.0 means zero wasted space, which is the design objective.
    Setbacks: 2 ft on road_side, 1 ft on the other three sides."""
    sb_road, sb_other = 2.0, 1.0
    left   = sb_road if road_side == "W" else sb_other
    right  = sb_road if road_side == "E" else sb_other
    bottom = sb_road if road_side == "S" else sb_other
    top    = sb_road if road_side == "N" else sb_other
    inner_w = max(fp.plot_w - left - right, 1e-6)
    inner_h = max(fp.plot_h - bottom - top, 1e-6)
    inner_area = inner_w * inner_h
    used = sum(r.area for r in fp.rooms)
    return round(min(used / inner_area, 1.0), 3)


def _score_one(fp, selected_vastu_rules, weights):
    w_aq, w_vs, w_su, w_ac = weights
    aq = score_adjacency_quality(fp)
    se = score_spatial_efficiency(fp)
    lc = score_layout_compactness(fp)
    vs, vastu_details = score_vastu(fp, selected_vastu_rules)
    ac = score_area_compliance(fp, [r.name for r in fp.rooms])
    su = score_space_usage(fp)
    fitness = w_aq * aq + w_vs * vs + w_su * su + w_ac * ac
    fp.scores = {
        "AQ": aq, "SE": se, "LC": lc, "VS": vs, "AC": ac, "SU": su,
        "Fitness": round(fitness, 4),
        "vastu_details": vastu_details,
    }
    return fp


def rank_plans(plans, selected_vastu_rules,
               w_aq=0.25, w_vs=0.35, w_su=0.30, w_ac=0.10):
    """Score and rank plans. Accepts either a list of FloorPlan (legacy) or a
    list of (raw, optimised) tuples. Returns a list of dicts:
        {"raw": raw_fp, "optimised": opt_fp, "fitness_gain": float}
    sorted by the OPTIMISED plan's fitness, descending.

    Objectives (cost & climate intentionally removed):
        AQ (adjacency) + VS (Vastu) + SU (space usage) + AC (area compliance)
    """
    weights = (w_aq, w_vs, w_su, w_ac)
    pairs = []
    for item in plans:
        if isinstance(item, tuple):
            raw_fp, opt_fp = item
        else:
            raw_fp = opt_fp = item
        _score_one(raw_fp, selected_vastu_rules, weights)
        _score_one(opt_fp, selected_vastu_rules, weights)
        pairs.append({
            "raw": raw_fp,
            "optimised": opt_fp,
            "fitness_gain": round(
                opt_fp.scores["Fitness"] - raw_fp.scores["Fitness"], 4),
        })
    pairs.sort(key=lambda p: p["optimised"].scores["Fitness"], reverse=True)
    return pairs
