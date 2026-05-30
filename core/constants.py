# ── Room Definitions ──────────────────────────────────────────────────────────
# Mandatory rooms in every plan (per professor's spec)
COMMON_ROOMS = [
    "external_staircase",
    "foyer",
    "living_room",
    "kitchen",
    "dining_room",
    "utility_area",
    "common_bathroom",
    "master_bedroom",
    "attached_bathroom",
]

# Optional rooms the user can opt-in via checkboxes
OPTIONAL_ROOMS = [
    "bedroom_01",
    "bedroom_02",
    "bedroom_03",
    "balcony",
    "study",
    "pooja_room",
]
ROOM_CONFIGS = {
    "1BHK": ["master_bedroom", "kitchen", "living_room", "bathroom", "foyer"],
    "2BHK": ["master_bedroom", "bedroom_01", "kitchen", "living_room",
             "dining_room", "common_bathroom", "attached_bathroom", "foyer", "balcony"],
    "3BHK": ["master_bedroom", "bedroom_01", "bedroom_02", "kitchen",
             "living_room", "dining_room", "common_bathroom", "attached_bathroom",
             "pooja_room", "foyer", "balcony", "utility_area"],
    "3BHK+Study": ["master_bedroom", "bedroom_01", "bedroom_02", "kitchen",
                   "living_room", "dining_room", "common_bathroom", "attached_bathroom",
                   "pooja_room", "foyer", "balcony", "utility_area", "study"],
}

# Min area in sq ft per room type
ROOM_MIN_AREA = {
    "master_bedroom":   150,
    "bedroom_01":       120,
    "bedroom_02":       100,
    "bedroom_03":       100,
    "kitchen":           80,
    "living_room":      180,
    "dining_room":      100,
    "common_bathroom":   45,
    "attached_bathroom": 45,
    "powder_room":       25,
    "foyer":             40,
    "internal_corridor": 30,
    "balcony":           60,
    "pooja_room":        30,
    "utility_area":      35,
    "study":             80,
    "external_staircase":40,
}

ROOM_MAX_AREA = {k: v * 2.2 for k, v in ROOM_MIN_AREA.items()}

# Realistic (width, height) ranges in feet for a 30x40 / 40x60 Indian plot.
# Used by the rule-based perimeter layout so rooms aren't 19' x 24' bedrooms.
ROOM_SIZE_RANGES = {
    "external_staircase": ((6, 8),   (8, 11)),
    "foyer":              ((5, 7),   (6, 8)),
    "living_room":        ((12, 16), (14, 20)),
    "dining_room":        ((9, 12),  (10, 14)),
    "kitchen":            ((8, 11),  (10, 14)),
    "utility_area":       ((4, 6),   (6, 9)),
    "master_bedroom":     ((11, 14), (12, 15)),
    "attached_bathroom":  ((5, 6),   (6, 8)),
    "bedroom_01":         ((10, 12), (10, 13)),
    "bedroom_02":         ((9, 11),  (10, 12)),
    "bedroom_03":         ((9, 11),  (9, 11)),
    "common_bathroom":    ((5, 6),   (6, 8)),
    "pooja_room":         ((4, 6),   (4, 6)),
    "balcony":            ((4, 6),   (8, 12)),
    "study":              ((8, 10),  (9, 11)),
    "internal_corridor":  ((3, 4),   (10, 30)),
}

CORRIDOR_WIDTH_MIN = 3.0
CORRIDOR_WIDTH_MAX = 4.0

# Room display colors
ROOM_COLORS = {
    "master_bedroom":   "#1565C0",
    "bedroom_01":       "#1976D2",
    "bedroom_02":       "#1E88E5",
    "bedroom_03":       "#2196F3",
    "kitchen":          "#B71C1C",
    "living_room":      "#2E7D32",
    "dining_room":      "#388E3C",
    "common_bathroom":  "#F9A825",
    "attached_bathroom":"#F57F17",
    "powder_room":      "#FF8F00",
    "foyer":            "#6A1B9A",
    "internal_corridor":"#78909C",
    "balcony":          "#00695C",
    "pooja_room":       "#E65100",
    "utility_area":     "#4E342E",
    "study":            "#283593",
    "external_staircase":"#455A64",
}

# Required adjacencies (room_a, room_b, priority)
REQUIRED_ADJACENCIES = [
    ("kitchen",        "dining_room",      1.0),
    ("dining_room",    "living_room",      0.9),
    ("living_room",    "foyer",            1.0),
    ("master_bedroom", "attached_bathroom",1.0),
    ("bedroom_01",     "common_bathroom",  0.8),
    ("bedroom_02",     "common_bathroom",  0.8),
    ("kitchen",        "utility_area",     0.7),
    ("pooja_room",     "foyer",            0.6),
    ("living_room",    "balcony",          0.7),
    ("dining_room",    "kitchen",          1.0),
    ("foyer",          "living_room",      0.9),
]

# Vastu rules: (room, preferred_zones, weight)
# Zones: NE, N, NW, W, SW, S, SE, E, Centre
VASTU_RULES = [
    ("Main Entrance",   "foyer",           ["N","NE","E"],          5),
    ("Kitchen",         "kitchen",         ["SE"],                  5),
    ("Master Bedroom",  "master_bedroom",  ["SW"],                  5),
    ("Toilet Avoid NE", "attached_bathroom",["W","NW","S"],         5),
    ("Living Room",     "living_room",     ["N","E","NE"],          4),
    ("Pooja Room",      "pooja_room",      ["NE"],                  5),
    ("Staircase",       "external_staircase",["S","SW","W"],       4),
    ("Common Bathroom", "common_bathroom", ["W","NW","S"],          3),
    ("Balcony",         "balcony",         ["N","E","NE"],          3),
    ("Study",           "study",           ["N","NE"],              3),
    ("Utility Area",    "utility_area",    ["NW","W"],              2),
    ("Dining Room",     "dining_room",     ["E","W"],               2),
]

# Climate zones
CLIMATE_ZONES = {
    "Hot-Dry (Rajasthan, Gujarat)":      {"wwr": 0.15, "orientation": "EW", "buffer": "W"},
    "Hot-Humid (Mumbai, Chennai, Kochi)":{"wwr": 0.25, "orientation": "EW", "buffer": "W"},
    "Composite (Delhi, Bengaluru, Pune)":{"wwr": 0.20, "orientation": "EW", "buffer": "SW"},
    "Cold (Shimla, Dehradun)":           {"wwr": 0.35, "orientation": "NS", "buffer": "N"},
    "Warm-Humid (Kolkata, Bhubaneswar)": {"wwr": 0.22, "orientation": "EW", "buffer": "W"},
}

# India construction cost per sqft (₹) by finishing level
COST_PER_SQFT = {
    "Basic":    1800,
    "Standard": 2400,
    "Premium":  3200,
}

PLUMBING_COST_PER_METER = 2500   # ₹ per metre of pipe run
WALL_COST_PER_METER     = 1200   # ₹ per metre of wall (material only)
