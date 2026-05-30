import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import numpy as np
import io
from core.constants import ROOM_COLORS

# Architectural-drawing colour scheme
PLOT_OUTLINE   = "#00C853"   # green — outer plot boundary
SETBACK_LINE   = "#E53935"   # red   — inner setback line
ROOM_OUTLINE   = "#1E88E5"   # blue  — room walls
ROAD_COLOR     = "#37474F"   # dark slate for the road strip
BG_PAGE        = "#0D1117"
BG_PLOT        = "#FFFFFF"   # white interior so blue/red read crisply
SETBACK_ROAD   = 2.0         # ft, road-facing side
SETBACK_OTHER  = 1.0         # ft, other three sides


def _setbacks(road_side="S"):
    left   = SETBACK_ROAD if road_side == "W" else SETBACK_OTHER
    right  = SETBACK_ROAD if road_side == "E" else SETBACK_OTHER
    bottom = SETBACK_ROAD if road_side == "S" else SETBACK_OTHER
    top    = SETBACK_ROAD if road_side == "N" else SETBACK_OTHER
    return left, right, bottom, top


def format_room_label(name):
    labels = {
        "external_staircase": "External\nStaircase",
        "foyer": "Foyer",
        "living_room": "Living",
        "kitchen": "Kitchen",
        "dining_room": "Dining",
        "utility_area": "Utility",
        "common_bathroom": "Common\nWash",
        "attached_bathroom": "Attached\nWash",
        "master_bedroom": "Master\nBedroom",
        "bedroom_01": "Bedroom 1",
        "bedroom_02": "Bedroom 2",
        "bedroom_03": "Bedroom 3",
        "pooja_room": "Pooja",
        "study": "Study",
        "balcony": "Balcony",
    }
    return labels.get(name, name.replace("_", "\n").title())


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _label_color(fill_color):
    r, g, b = _hex_to_rgb(fill_color)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#111111" if luminance > 0.58 else "#FFFFFF"


def _label_box_and_stroke(fill_color):
    """Return (bbox_dict, stroke_color) tuned to the room fill color.

    Dark rooms get a soft light pill with dark text.
    Light rooms get a soft dark pill with white text.
    This avoids the muddy black halo you were seeing on small labels.
    """
    r, g, b = _hex_to_rgb(fill_color)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    if luminance > 0.58:
        return (
            dict(boxstyle='round,pad=0.16', facecolor=(1, 1, 1, 0.60), edgecolor='none'),
            '#FFFFFF',
        )
    return (
        dict(boxstyle='round,pad=0.16', facecolor=(0, 0, 0, 0.16), edgecolor='none'),
        '#000000',
    )


def _draw_plot_skeleton(ax, pw, ph, road_side="S"):
    """Draw white interior + green plot outline + red dashed setbacks +
    grey road strip outside the road edge."""
    left, right, bottom, top = _setbacks(road_side)

    # Grey road strip outside the plot on the road side
    rw = 4.0
    road_rect = {
        "S": (-1, -rw - 0.6, pw + 2, rw),
        "N": (-1, ph + 0.6, pw + 2, rw),
        "E": (pw + 0.6, -1, rw, ph + 2),
        "W": (-rw - 0.6, -1, rw, ph + 2),
    }.get(road_side)
    if road_rect:
        ax.add_patch(Rectangle((road_rect[0], road_rect[1]),
                               road_rect[2], road_rect[3],
                               facecolor=ROAD_COLOR, edgecolor='none',
                               alpha=0.9, zorder=1))

    # White interior (paper)
    ax.add_patch(Rectangle((0, 0), pw, ph,
                           facecolor=BG_PLOT, edgecolor='none', zorder=0))
    # Green plot outline
    ax.add_patch(Rectangle((0, 0), pw, ph,
                           facecolor='none', edgecolor=PLOT_OUTLINE,
                           linewidth=3.0, zorder=4))
    # Red dashed setback (inner buildable area)
    ax.add_patch(Rectangle((left, bottom),
                           pw - left - right, ph - bottom - top,
                           facecolor='none', edgecolor=SETBACK_LINE,
                           linewidth=1.5, linestyle=(0, (6, 4)), zorder=3))
    # "ROAD" label on the grey strip
    road_label = {"S": (pw / 2, -2.6, 0),
                  "N": (pw / 2, ph + 2.6, 0),
                  "E": (pw + 2.6, ph / 2, 90),
                  "W": (-2.6, ph / 2, 90)}.get(road_side, (pw / 2, -2.6, 0))
    ax.text(road_label[0], road_label[1], "ROAD",
            ha='center', va='center', fontsize=8,
            color="#FFFFFF", fontweight='bold', rotation=road_label[2], zorder=6)


def _draw_entrance_arrow(ax, pw, ph, road_side="S"):
    """Big green arrow from the road into the plot, labelled ENTRANCE."""
    if road_side == "S":
        xs, ys, xe, ye, rot = pw * 0.5, -3.2, pw * 0.5, 1.5, 0
    elif road_side == "N":
        xs, ys, xe, ye, rot = pw * 0.5, ph + 3.2, pw * 0.5, ph - 1.5, 0
    elif road_side == "E":
        xs, ys, xe, ye, rot = pw + 3.2, ph * 0.5, pw - 1.5, ph * 0.5, 90
    else:
        xs, ys, xe, ye, rot = -3.2, ph * 0.5, 1.5, ph * 0.5, 90
    ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
                arrowprops=dict(arrowstyle="-|>", color="#00C853",
                                lw=2.2, mutation_scale=18), zorder=8)
    ax.text((xs + xe) / 2, (ys + ye) / 2,
            "ENTRANCE", color="#00C853", fontsize=7.5, fontweight='bold',
            rotation=rot, ha='center', va='center', zorder=8,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])


def _draw_plot_size(ax, pw, ph):
    ax.text(0.2, ph + 1.0, f"Plot: {int(pw)}' x {int(ph)}'",
            fontsize=8.5, fontweight='bold', color="#0D47A1",
            ha='left', va='bottom', zorder=7,
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='#FFFFFF', edgecolor='#0D47A1', linewidth=0.8))


def _draw_rooms(ax, rooms, label_fs=7.0, area_fs=5.5, show_dims=True):
    """Fill rooms edge-to-edge with blue outlines. No gaps, no rounded corners."""
    for room in rooms:
        col = ROOM_COLORS.get(room.name, "#546E7A")
        ax.add_patch(Rectangle((room.x, room.y), room.w, room.h,
                               facecolor=col, alpha=0.85,
                               edgecolor=ROOM_OUTLINE, linewidth=1.4, zorder=2))
        cx = room.x + room.w / 2
        cy = room.y + room.h / 2
        label = format_room_label(room.name)
        text_col = _label_color(col)
        label_box, stroke_col = _label_box_and_stroke(col)

        room_short_side = min(room.w, room.h)
        fs = max(5.2, min(label_fs + 0.2, room_short_side * 0.62))
        afs = max(4.2, min(area_fs, room_short_side * 0.44))
        name_only = room.w * room.h < 95 or room_short_side < 6.3
        show_area = room.w * room.h >= 120 and room_short_side >= 7.0
        show_dim_line = show_dims and room.w * room.h >= 170 and room_short_side >= 8.5

        lines = [label]
        if show_area:
            lines.append(f"{room.area:.0f} sqft")
        if show_dim_line:
            lines.append(f"{room.w:.1f}' x {room.h:.1f}'")

        if name_only:
            lines = [label]

        text = "\n".join(lines)
        ax.text(
            cx,
            cy,
            text,
            ha='center',
            va='center',
            fontsize=fs if len(lines) == 1 else afs + 0.5,
            color=text_col,
            fontweight='bold',
            zorder=5,
            linespacing=1.05,
            bbox=label_box,
            path_effects=[pe.withStroke(linewidth=0.65, foreground=stroke_col)],
        )


def _draw_north_arrow(ax, pw, ph):
    """North arrow drawn OUTSIDE the plot so it never overlaps rooms."""
    nx = pw + 1.8
    y_top = ph + 0.6
    ax.annotate("", xy=(nx, y_top), xytext=(nx, y_top - 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#FFFFFF", lw=1.6),
                zorder=8)
    ax.text(nx, y_top + 0.5, "N", fontsize=10, color="#FFFFFF",
            fontweight='bold', ha='center', va='bottom', zorder=8,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])


def draw_floorplan(fp, rank=1, show_grid=False, show_compass=True,
                    road_side="S", title_prefix="Floor Plan"):
    """Draw a single floor plan and return matplotlib figure."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG_PAGE)
    ax.set_facecolor(BG_PAGE)

    pw, ph = fp.plot_w, fp.plot_h
    _draw_plot_skeleton(ax, pw, ph, road_side=road_side)
    _draw_rooms(ax, fp.rooms, label_fs=7.5, area_fs=6.0)
    _draw_entrance_arrow(ax, pw, ph, road_side=road_side)
    _draw_plot_size(ax, pw, ph)
    if show_compass:
        _draw_north_arrow(ax, pw, ph)

    # Rank badge
    badge_color = {1: '#FFD700', 2: '#C0C0C0', 3: '#CD7F32'}.get(rank, '#4FC3F7')
    ax.text(1.2, ph - 1.8, f"#{rank}", fontsize=18, color=badge_color,
            fontweight='bold', zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground='black')])

    # Score strip below plan
    s = fp.scores
    score_txt = (f"Fitness {s['Fitness']:.3f}   "
                 f"Vastu {s['VS']*100:.0f}%   "
                 f"Adjacency {s['AQ']*100:.0f}%   "
                 f"Space {s.get('SU', 0) * 100:.0f}%")
    ax.text(pw / 2, -2.0, score_txt,
            ha='center', va='center', fontsize=8,
            color='#B0BEC5', style='italic', zorder=5)

    ax.set_xlim(-6, pw + 6)
    ax.set_ylim(-7, ph + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{title_prefix} #{rank}", color='#4FC3F7',
                 fontsize=11, fontweight='bold', pad=8)
    plt.tight_layout(pad=0.5)
    return fig


def draw_top5_grid(plans):
    """Draw top 5 plans in a 2+2+1 grid. Returns figure."""
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#0D1117')

    positions = [
        (0.02, 0.52, 0.46, 0.44),   # Plan 1 — top left
        (0.52, 0.52, 0.46, 0.44),   # Plan 2 — top right
        (0.02, 0.04, 0.30, 0.44),   # Plan 3 — mid left
        (0.35, 0.04, 0.30, 0.44),   # Plan 4 — mid centre
        (0.68, 0.04, 0.30, 0.44),   # Plan 5 — mid right
    ]

    for i, (fp, (left, bottom, width, height)) in enumerate(zip(plans[:5], positions)):
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_facecolor(BG_PAGE)

        pw, ph = fp.plot_w, fp.plot_h
        rank = i + 1

        _draw_plot_skeleton(ax, pw, ph, road_side="S")
        _draw_rooms(ax, fp.rooms,
                    label_fs=(6.5 if i < 2 else 5.5),
                    area_fs=(5.0 if i < 2 else 4.5),
                    show_dims=False)

        # Rank + scores
        s = fp.scores
        badge_color = {1: '#FFD700', 2: '#C0C0C0', 3: '#CD7F32'}.get(rank, '#4FC3F7')
        ax.text(1.0, ph - 1.4, f"#{rank}",
                fontsize=14 if i < 2 else 11,
                color=badge_color, fontweight='bold', zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

        score_line = (f"Vastu {s['VS']*100:.0f}%  "
                      f"Adj {s['AQ']*100:.0f}%  "
                      f"Space {s.get('SU', 0) * 100:.0f}%  "
                      f"Fit {s['Fitness']:.3f}")
        ax.set_title(f"Plan #{rank}\n{score_line}",
                     color='#4FC3F7' if i == 0 else '#B0BEC5',
                     fontsize=7.5, fontweight='bold', pad=4)

        ax.set_xlim(-1, pw + 1)
        ax.set_ylim(-1, ph + 1)
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle("Top 5 Optimised Floor Plans",
                 fontsize=16, fontweight='bold', color='white', y=0.99)
    return fig


def draw_comparison(raw_fp, opt_fp, rank=1, road_side="S"):
    """Side-by-side: AI-generated (raw) vs Optimised, with a score delta strip."""
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG_PAGE)

    titles = ("AI Generated", "Optimised")
    subtitles = ("Raw VAE + Geometry Predictor", "Vastu + Space-Usage Squarify")
    plans = (raw_fp, opt_fp)
    axes_positions = [(0.055, 0.075, 0.39, 0.78), (0.555, 0.075, 0.39, 0.78)]

    for ax_pos, fp, title, subtitle in zip(axes_positions, plans, titles, subtitles):
        ax = fig.add_axes(ax_pos)
        ax.set_facecolor(BG_PAGE)
        pw, ph = fp.plot_w, fp.plot_h
        _draw_plot_skeleton(ax, pw, ph, road_side=road_side)
        _draw_rooms(ax, fp.rooms, label_fs=6.2, area_fs=4.9, show_dims=True)
        _draw_entrance_arrow(ax, pw, ph, road_side=road_side)
        _draw_plot_size(ax, pw, ph)
        _draw_north_arrow(ax, pw, ph)
        s = fp.scores
        info = (f"Fit {s['Fitness']:.3f}   "
                f"Vastu {s['VS']*100:.0f}%   "
                f"Adj {s['AQ']*100:.0f}%   "
                f"Space {s.get('SU', 0)*100:.0f}%")

        center_x = ax_pos[0] + ax_pos[2] / 2
        fig.text(center_x, 0.91, title,
                 ha='center', va='center', color='#4FC3F7',
                 fontsize=15, fontweight='bold')
        fig.text(center_x, 0.885, subtitle,
                 ha='center', va='center', color='#B0BEC5',
                 fontsize=10, fontweight='bold')
        fig.text(center_x, 0.86, info,
                 ha='center', va='center', color='#4FC3F7',
                 fontsize=11, fontweight='bold')
        ax.set_xlim(-6, pw + 6)
        ax.set_ylim(-7, ph + 5)
        ax.set_aspect('equal')
        ax.axis('off')

    gain = opt_fp.scores["Fitness"] - raw_fp.scores["Fitness"]
    direction = "+" if gain >= 0 else ""
    fig.suptitle(
        f"Plan #{rank}   |   Optimisation gain: {direction}{gain*100:.1f}% fitness "
        f"({raw_fp.scores['Fitness']:.3f} -> {opt_fp.scores['Fitness']:.3f})",
        fontsize=15, fontweight='bold',
        color='#00E676' if gain >= 0 else '#FF5252',
        y=0.965)
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def draw_score_radar(fp):
    """Draw a radar chart of scores for a plan."""
    categories = ['Adjacency\nQuality', 'Vastu\nScore',
                  'Space\nUsage', 'Area\nCompliance']
    s = fp.scores
    values = [s['AQ'], s['VS'], s.get('SU', 0), s['AC']]
    values_plot = values + [values[0]]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#4FC3F7')
    ax.fill(angles, values_plot, alpha=0.3, color='#4FC3F7')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], color='#B0BEC5', fontsize=6)
    ax.grid(color='#2A2A4A', linewidth=0.8)
    ax.spines['polar'].set_color('#2A2A4A')

    ax.set_title(f"Score Profile\nFitness: {s['Fitness']:.3f}",
                 color='#4FC3F7', fontsize=8, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def draw_vastu_compass(fp):
    """Draw a compass showing room placements vs Vastu zones."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw compass zones
    zone_positions = {
        "NW":(-0.7, 0.7), "N":(0, 1.0), "NE":(0.7, 0.7),
        "W": (-1.0, 0),   "Centre":(0,0),"E": (1.0, 0),
        "SW":(-0.7,-0.7), "S":(0,-1.0), "SE":(0.7,-0.7),
    }
    zone_colors = {
        "NE":"#1B5E20","N":"#2E7D32","NW":"#388E3C",
        "E": "#1565C0","Centre":"#37474F","W":"#5D4037",
        "SE":"#B71C1C","S":"#C62828","SW":"#6A1B9A",
    }
    for zone, (zx, zy) in zone_positions.items():
        circle = plt.Circle((zx*0.75, zy*0.75), 0.28,
                             color=zone_colors.get(zone,'#37474F'),
                             alpha=0.5, zorder=2)
        ax.add_patch(circle)
        ax.text(zx*0.75, zy*0.75, zone, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=3)

    # Place room dots
    s = fp.scores.get('vastu_details', [])
    for rule_name, room_name, zone, passed in s:
        if zone in zone_positions:
            zx, zy = zone_positions[zone]
            offset = 0.12
            color = '#00E676' if passed == True else ('#FFFF00' if passed == 'partial' else '#FF5252')
            ax.plot(zx*0.75 + offset, zy*0.75 + offset, 'o',
                    color=color, markersize=8, zorder=5)
            short = room_name.replace('_', ' ')[:8]
            ax.text(zx*0.75 + offset, zy*0.75 - 0.1, short,
                    ha='center', va='top', fontsize=5, color=color, zorder=6)

    ax.set_title("Vastu Compass", color='#4FC3F7', fontsize=9, fontweight='bold')

    # Legend
    for label, col in [("✓ Correct", '#00E676'), ("~ Partial", '#FFFF00'), ("✗ Wrong", '#FF5252')]:
        ax.plot([], [], 'o', color=col, label=label, markersize=6)
    ax.legend(loc='lower center', fontsize=6.5, facecolor='#1A1A2E',
              edgecolor='#4FC3F7', labelcolor='white', ncol=3,
              bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    return fig
