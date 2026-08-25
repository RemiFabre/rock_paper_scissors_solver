"""Static figures for the README. Run: python explain/figures.py"""
import os
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from common import (
    BALANCED,
    COLORS,
    GRID,
    INK,
    INK2,
    INTUITIVE,
    OPTIMAL,
    OUT,
    SURFACE,
    UNBALANCED,
    UNIFORM,
    draw_triangle,
    payoffs_vs_defenses,
    solve_exact,
    style,
    tern,
    ternary_grid,
)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.facecolor": SURFACE, "savefig.dpi": 160})


def game_graph(ax, m, title):
    """Arrows from winner to loser, next to the payoff matrix."""
    n = len(m)
    ya = np.linspace(0.85, 0.15, n)
    for i in range(n):
        ax.text(0.12, ya[i], f"A{i}", color=COLORS[i], fontsize=15, weight="bold", ha="center", va="center")
        ax.text(0.58, ya[i], f"D{i}", color=COLORS[i], fontsize=15, weight="bold", ha="center", va="center")
    for i in range(n):
        for j in range(n):
            v = m[i][j]
            if v == 0:
                continue
            src, dst = ((0.17, ya[i]), (0.53, ya[j])) if v > 0 else ((0.53, ya[j]), (0.17, ya[i]))
            col = COLORS[i] if v > 0 else COLORS[j]
            ax.add_patch(
                FancyArrowPatch(src, dst, arrowstyle="-|>", mutation_scale=16, lw=2, color=col, alpha=0.85, shrinkA=6, shrinkB=6)
            )
    # payoff matrix
    x0, cw = 0.86, 0.1
    for j in range(n):
        ax.text(x0 + cw * j, ya[0] + 0.16, f"D{j}", color=COLORS[j], ha="center", va="center", fontsize=9, weight="bold")
    for i in range(n):
        ax.text(x0 - cw * 0.9, ya[i], f"A{i}", color=COLORS[i], ha="center", va="center", fontsize=9, weight="bold")
        for j in range(n):
            v = m[i][j]
            txt = f"{v:+d}" if v else "0"
            ax.text(x0 + cw * j, ya[i], txt, ha="center", va="center", fontsize=11, color=INK if v else INK2)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.08, 1.2)
    ax.axis("off")
    s = solve_exact(m)
    ax.text(0.0, 1.15, title, fontsize=12, color=INK, ha="left", va="center")
    ax.text(0.5, -0.06, "optimal attack " + "[" + ", ".join(str(v) for v in s.attack) + "]" + f"    value {s.value}",
            ha="center", va="bottom", fontsize=10, color=INK2)


def fig_games():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    game_graph(axes[0], BALANCED, "Balanced: rock paper scissors")
    game_graph(axes[1], UNBALANCED, "Unbalanced: A1 also beats D2")
    fig.text(0.5, 1.0, "Arrow from winner to loser. Rows are attacks, columns defences, entries the attacker's payoff.",
             ha="center", fontsize=9, color=INK2)
    fig.savefig(os.path.join(OUT, "games.png"), bbox_inches="tight")
    plt.close(fig)


def fig_indifference():
    """What each defence earns against several attack mixes."""
    mixes = [
        ("Uniform 1/3 each", UNIFORM),
        ("Intuitive: A1 half the time", INTUITIVE),
        ("Overcorrected: A0 two thirds", [2 / 3, 1 / 6, 1 / 6]),
        ("Optimal [4/9, 3/9, 2/9]", OPTIMAL),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, (name, x) in zip(axes, mixes):
        style(ax)
        p = payoffs_vs_defenses(UNBALANCED, x)
        worst = p.argmin()
        bars = ax.bar(range(3), p, color=COLORS[:3], width=0.6)
        for j, b in enumerate(bars):
            b.set_alpha(1.0 if j == worst else 0.35)
            ax.text(j, p[j] + (0.02 if p[j] >= 0 else -0.02), f"{p[j]:+.2f}", ha="center",
                    va="bottom" if p[j] >= 0 else "top", fontsize=9, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axhline(1 / 9, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(range(3), ["D0", "D1", "D2"])
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.text(1, -0.62, "attack mix " + "[" + ", ".join(f"{v:.2f}" for v in x) + "]", ha="center", fontsize=8.5, color=INK2)
        ax.text(1, -0.75, f"guaranteed: {p.min():+.2f} per round", ha="center", fontsize=8.5, color=INK, weight="bold")
        ax.set_ylim(-0.85, 0.75)
    axes[0].set_ylabel("attacker's payoff per round")
    fig.suptitle("Against a given attack mix, the defender picks the defence with the lowest bar (solid). "
                 "The attacker keeps only that. Dotted line: game value 1/9.", fontsize=9.5, color=INK2, y=1.01)
    fig.savefig(os.path.join(OUT, "indifference.png"), bbox_inches="tight")
    plt.close(fig)


def fig_landscape():
    """Worst-case payoff over every attack mix, with best-response regions."""
    pts = ternary_grid(260)
    P = pts @ np.asarray(UNBALANCED, float)
    worst = P.min(axis=1)
    region = P.argmin(axis=1)
    xy = tern(pts)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_ylim(-0.12, 1.02)
    ax = axes[0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=worst, cmap="Blues", s=4, vmin=-1, vmax=1 / 9, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, orientation="horizontal", location="bottom")
    cb.set_label("guaranteed payoff (worst defence)", color=INK2)
    cb.ax.tick_params(colors=INK2)
    draw_triangle(ax, ["A0 always", "A1 always", "A2 always"])
    ax.set_title("What each attack mix guarantees", loc="left", color=INK, y=1.02)
    ax = axes[1]
    ax.scatter(xy[:, 0], xy[:, 1], c=[COLORS[r] for r in region], s=4, alpha=0.35, linewidths=0)
    draw_triangle(ax, ["A0 always", "A1 always", "A2 always"])
    ax.set_title("Which defence punishes that mix", loc="left", color=INK, y=1.02)
    for j in range(3):
        ax.scatter([], [], color=COLORS[j], label=f"D{j} is the defender's best reply")
    ax.legend(loc="lower center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.2), ncol=3)
    for ax in axes:
        for name, x, mk, off in [("optimal", OPTIMAL, "*", (-70, -22)), ("uniform", UNIFORM, "o", (10, 6)), ("intuitive", INTUITIVE, "s", (10, -10))]:
            p = tern(x)
            ax.scatter(p[0], p[1], marker=mk, s=200 if mk == "*" else 60, color=INK, zorder=5, edgecolor=SURFACE)
            ax.annotate(f"{name} ({min(payoffs_vs_defenses(UNBALANCED, x)):+.2f})", p, xytext=off,
                        textcoords="offset points", fontsize=8.5, color=INK, weight="bold" if mk == "*" else "normal")
    fig.suptitle("Every point of the triangle is an attack mix. The optimum is the peak on the left, "
                 "and the meeting point of the three regions on the right.", fontsize=9.5, color=INK2, y=1.0)
    fig.savefig(os.path.join(OUT, "landscape.png"), bbox_inches="tight")
    plt.close(fig)


def fig_buff():
    """Optimal frequencies as A1 gets stronger against D2."""
    ts = np.linspace(0, 3, 61)
    att, dfn, val = [], [], []
    for t in ts:
        s = solve_exact([[1, 0, -1], [-1, 1, Fraction(t).limit_denominator(1000)], [0, -1, 1]])
        att.append([float(v) for v in s.attack])
        dfn.append([float(v) for v in s.defense])
        val.append(float(s.value))
    att, dfn = np.array(att), np.array(dfn)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, data, who in zip(axes, [att, dfn], ["attack", "defence"]):
        style(ax)
        for i in range(3):
            ax.plot(ts, data[:, i], color=COLORS[i], lw=2.2, label=f"{who[0].upper()}{i}")
            ax.text(ts[-1] + 0.05, data[-1, i], f"{who[0].upper()}{i}", color=COLORS[i], va="center", fontsize=10, weight="bold")
        ax.axvline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("payoff of A1 against D2 (0 = neutral, 1 = win, 3 = crushing win)")
        ax.set_title(f"Optimal {who} frequencies", loc="left", color=INK)
        ax.set_xlim(0, 3.3)
    axes[0].set_ylabel("probability")
    axes[0].text(1.05, 0.02, "the README's unbalanced game", fontsize=8, color=INK2)
    fig.suptitle("Buffing A1 does not change how often A1 is played (1/3 throughout). "
                 "It changes how often A0 and A2 are played.", fontsize=10, color=INK2, y=1.0)
    fig.savefig(os.path.join(OUT, "buff.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_games()
    fig_indifference()
    fig_landscape()
    fig_buff()
    print("done")
