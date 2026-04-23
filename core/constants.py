# ── Room Definitions ──────────────────────────────────────────────────────────

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
}

ROOM_MAX_AREA = {k: v * 2.2 for k, v in ROOM_MIN_AREA.items()}

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
    ("Staircase",       "internal_corridor",["S","SW"],             4),
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
