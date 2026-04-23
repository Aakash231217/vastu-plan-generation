import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
import io
from core.constants import ROOM_COLORS


COMPASS = {
    "N": (0.5, 0.97), "NE": (0.97, 0.97), "E": (0.97, 0.5),
    "SE": (0.97, 0.03), "S": (0.5, 0.03), "SW": (0.03, 0.03),
    "W": (0.03, 0.5), "NW": (0.03, 0.97), "Centre": (0.5, 0.5),
}


def format_room_label(name):
    return name.replace("_", "\n").replace("01","").replace("02","").replace("03","").strip()


def draw_floorplan(fp, rank=1, show_grid=True, show_compass=True):
    """Draw a single floor plan and return matplotlib figure."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#0F0F23')

    pw, ph = fp.plot_w, fp.plot_h

    # Plot boundary
    boundary = FancyBboxPatch((0, 0), pw, ph,
        boxstyle="round,pad=0.3",
        facecolor='#1A1A2E', edgecolor='#4FC3F7',
        linewidth=2.5, zorder=1)
    ax.add_patch(boundary)

    # Grid
    if show_grid:
        for x in np.arange(0, pw+1, 5):
            ax.axvline(x, color='#2A2A4A', linewidth=0.4, zorder=0)
        for y in np.arange(0, ph+1, 5):
            ax.axhline(y, color='#2A2A4A', linewidth=0.4, zorder=0)

    # Draw rooms
    for room in fp.rooms:
        col = ROOM_COLORS.get(room.name, "#546E7A")

        # Room fill
        rect = FancyBboxPatch(
            (room.x, room.y), room.w, room.h,
            boxstyle="round,pad=0.2",
            facecolor=col, alpha=0.80,
            edgecolor='white', linewidth=1.2, zorder=2)
        ax.add_patch(rect)

        # Room label
        label = format_room_label(room.name)
        area_txt = f"{room.area:.0f} sqft"
        ax.text(room.cx, room.cy + 0.6, label,
                ha='center', va='center', fontsize=6.5,
                color='white', fontweight='bold', zorder=4,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
        ax.text(room.cx, room.cy - 0.8, area_txt,
                ha='center', va='center', fontsize=5.5,
                color='#E0E0E0', zorder=4, alpha=0.9)

        # Shared wall indicators (doors)
        for other in fp.rooms:
            if other.name != room.name and room.shares_wall(other):
                mid_x = (max(room.x, other.x) + min(room.right, other.right)) / 2
                mid_y = (max(room.y, other.y) + min(room.top, other.top)) / 2
                ax.plot(mid_x, mid_y, 's', color='#FFD54F',
                        markersize=4, zorder=5, alpha=0.7)

    # North arrow
    if show_compass:
        ax.annotate("N", xy=(pw * 0.93, ph * 0.93),
                    fontsize=10, color='#4FC3F7', fontweight='bold',
                    ha='center', va='center', zorder=6)
        ax.annotate("", xy=(pw * 0.93, ph * 0.97),
                    xytext=(pw * 0.93, ph * 0.90),
                    arrowprops=dict(arrowstyle="-|>", color='#4FC3F7', lw=1.5),
                    zorder=6)

    # Rank badge
    badge_color = {1: '#FFD700', 2: '#C0C0C0', 3: '#CD7F32'}.get(rank, '#4FC3F7')
    ax.text(1.5, ph - 2, f"#{rank}", fontsize=18, color=badge_color,
            fontweight='bold', zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground='black')])

    # Score panel
    s = fp.scores
    score_txt = (f"Fitness: {s['Fitness']:.3f}  |  "
                 f"Vastu: {s['VS']*100:.0f}%  |  "
                 f"AQ: {s['AQ']*100:.0f}%  |  "
                 f"Cost: ₹{s['Cost']/100000:.1f}L")
    ax.text(pw/2, -1.8, score_txt,
            ha='center', va='center', fontsize=7,
            color='#B0BEC5', style='italic', zorder=5)

    ax.set_xlim(-2, pw + 2)
    ax.set_ylim(-3, ph + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Floor Plan #{rank}", color='#4FC3F7',
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
        ax.set_facecolor('#0F0F23')

        pw, ph = fp.plot_w, fp.plot_h
        rank = i + 1

        # Boundary
        boundary = FancyBboxPatch((0, 0), pw, ph,
            boxstyle="round,pad=0.3",
            facecolor='#1A1A2E', edgecolor='#4FC3F7', linewidth=2, zorder=1)
        ax.add_patch(boundary)

        for room in fp.rooms:
            col = ROOM_COLORS.get(room.name, "#546E7A")
            rect = FancyBboxPatch(
                (room.x, room.y), room.w, room.h,
                boxstyle="round,pad=0.2",
                facecolor=col, alpha=0.82,
                edgecolor='white', linewidth=0.8, zorder=2)
            ax.add_patch(rect)
            fs = 5.5 if i >= 2 else 6.5
            ax.text(room.cx, room.cy, format_room_label(room.name),
                    ha='center', va='center', fontsize=fs,
                    color='white', fontweight='bold', zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])

        # Rank + scores
        s = fp.scores
        badge_color = {1:'#FFD700', 2:'#C0C0C0', 3:'#CD7F32'}.get(rank, '#4FC3F7')
        ax.text(1, ph-1.5, f"#{rank}", fontsize=14 if i < 2 else 11,
                color=badge_color, fontweight='bold', zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

        score_line = (f"Fitness:{s['Fitness']:.3f}  Vastu:{s['VS']*100:.0f}%  "
                      f"AQ:{s['AQ']*100:.0f}%  ₹{s['Cost']/100000:.1f}L")
        ax.set_title(f"Plan #{rank}\n{score_line}",
                     color='#4FC3F7' if i == 0 else '#B0BEC5',
                     fontsize=7.5, fontweight='bold', pad=4)

        ax.set_xlim(-1, pw+1)
        ax.set_ylim(-1, ph+1)
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle("Top 5 Optimised Floor Plans",
                 fontsize=16, fontweight='bold', color='white', y=0.99)
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
    categories = ['Adjacency\nQuality', 'Spatial\nEfficiency',
                  'Layout\nCompact', 'Vastu\nScore', 'Area\nCompliance']
    s = fp.scores
    values = [s['AQ'], s['SE'], s['LC'], s['VS'], s['AC']]
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
