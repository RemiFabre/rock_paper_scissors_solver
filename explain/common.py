"""Shared definitions for the figures and animations."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rps_solver import solve_exact  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "images")

BALANCED = [[1, 0, -1], [-1, 1, 0], [0, -1, 1]]
UNBALANCED = [[1, 0, -1], [-1, 1, 1], [0, -1, 1]]

# Categorical palette (same hue for A_i and D_i)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

OPTIMAL = [4 / 9, 3 / 9, 2 / 9]
INTUITIVE = [0.25, 0.5, 0.25]  # "play the strongest attack most"
UNIFORM = [1 / 3, 1 / 3, 1 / 3]


def payoffs_vs_defenses(m, x):
    """Attacker's expected payoff against each pure defence."""
    return np.asarray(x) @ np.asarray(m, float)


def worst_case(m, x):
    return payoffs_vs_defenses(m, x).min()


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Ternary helpers: a mix (a, b, c) maps to a point in an equilateral triangle.
_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def tern(x):
    x = np.asarray(x, float)
    return x @ _CORNERS


def ternary_grid(n=300):
    """Barycentric coordinates covering the triangle on a regular grid."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return np.array(pts)


def draw_triangle(ax, labels, colors=COLORS, pad=0.06):
    tri = np.vstack([_CORNERS, _CORNERS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color=INK2, lw=1)
    offsets = [(-pad, -pad * 0.6), (pad, -pad * 0.6), (0, pad * 0.9)]
    for c, lab, col, off in zip(_CORNERS, labels, colors, offsets):
        ax.text(c[0] + off[0], c[1] + off[1], lab, ha="center", va="center", color=col, fontsize=12, weight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
