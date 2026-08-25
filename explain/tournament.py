"""Round-robin of adaptive and fixed players on the unbalanced game. Run: python explain/tournament.py"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
from agents import (  # noqa: E402
    BestReplier,
    Cyclic,
    FictitiousPlay,
    Fixed,
    Hedge,
    Nash,
    NGram,
    RegretMatching,
    SafeExploiter,
    Sticky,
    own_view,
    play,
)
from common import COLORS, INK, INK2, INTUITIVE, OUT, SURFACE, UNBALANCED, solve_exact, style  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.facecolor": SURFACE, "savefig.dpi": 160})

M = np.asarray(UNBALANCED, float)
VALUE = float(solve_exact(UNBALANCED).value)
ROUNDS = int(os.environ.get("ROUNDS", 10000))
SEEDS = 3


def roster(attacker: bool, rng):
    """One instance of every player type, from that side's point of view."""
    m = own_view(M, attacker)
    strongest = 1 if attacker else 0  # A1 wins twice, D0 is the defence that beats it
    biased = INTUITIVE if attacker else [0.5, 0.25, 0.25]
    return [
        Nash(m, rng=rng),
        Fixed(m, [1 / 3] * 3, "Uniform", rng=rng),
        Fixed(m, biased, "Intuitive", rng=rng),
        Sticky(m, strongest=strongest, rng=rng),
        Cyclic(m),
        FictitiousPlay(m, rng=rng),
        RegretMatching(m, rng=rng),
        Hedge(m, rng=rng),
        NGram(m, order=3, rng=rng),
        SafeExploiter(m, epsilon=0.1, rng=rng),
    ]


def round_robin():
    names_a = [p.name for p in roster(True, np.random.default_rng(0))]
    names_d = [p.name for p in roster(False, np.random.default_rng(0))]
    table = np.zeros((len(names_a), len(names_d)))
    for seed in range(SEEDS):
        for i in range(len(names_a)):
            for j in range(len(names_d)):
                rng = np.random.default_rng(1000 * seed + 10 * i + j)
                a = roster(True, rng)[i]
                d = roster(False, rng)[j]
                table[i, j] += play(M, a, d, ROUNDS).mean() / SEEDS
    return names_a, names_d, table


def fig_tournament(names_a, names_d, table):
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_facecolor(SURFACE)
    norm = TwoSlopeNorm(vmin=-1, vcenter=VALUE, vmax=1)
    ax.imshow(table, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(names_d)), names_d, rotation=35, ha="right", color=INK2)
    ax.set_yticks(range(len(names_a)), names_a, color=INK2)
    ax.set_xlabel("defender", color=INK2)
    ax.set_ylabel("attacker", color=INK2)
    for i in range(len(names_a)):
        for j in range(len(names_d)):
            v = table[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v - VALUE) > 0.45 else INK)
    ax.set_xticks(np.arange(-0.5, len(names_d)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names_a)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Attacker's mean payoff per round, {ROUNDS} rounds, {SEEDS} seeds. "
                 f"Game value {VALUE:+.2f} (gray). Blue: attacker does better, red: worse.",
                 fontsize=9.5, color=INK2, loc="left")
    fig.savefig(os.path.join(OUT, "tournament.png"), bbox_inches="tight")
    plt.close(fig)


def fig_safe_exploitation():
    eps = np.linspace(0, 0.4, 9)
    md = own_view(M, False)
    opponents = {
        "Intuitive defender": lambda rng: Fixed(md, [0.5, 0.25, 0.25], "Intuitive", rng=rng),
        "Sticky defender": lambda rng: Sticky(md, strongest=0, rng=rng),
        "Best-replying defender": lambda rng: BestReplier(md, rng=rng),
        "NGram(3) defender": lambda rng: NGram(md, order=3, rng=rng),
    }
    results = {k: [] for k in opponents}
    for e in eps:
        for name, make in opponents.items():
            vals = []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                vals.append(play(M, SafeExploiter(M, epsilon=e, rng=rng), make(rng), ROUNDS // 2).mean())
            results[name].append(np.mean(vals))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    style(ax)
    for (name, ys), col in zip(results.items(), COLORS):
        ax.plot(eps, ys, color=col, lw=2.2, marker="o", ms=5, label=name)
    ax.axhline(VALUE, color=INK2, lw=0.8, ls=":")
    ax.text(0.405, VALUE, "game value 1/9\n(what Nash gets)", va="center", fontsize=8.5, color=INK2)
    worst = min(min(v) for v in results.values())
    ax.set_xlabel("epsilon: the most the exploiter accepts to lose per round, relative to the game value")
    ax.set_ylabel("attacker's mean payoff per round")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(min(0.0, worst - 0.02), 0.2)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    ax.set_title("Safe exploitation: more epsilon buys more against biased opponents.\n"
                 "Against a reader the loss grows, but stays far below the budget (the floor is value minus epsilon).",
                 fontsize=9, color=INK2, loc="left")
    fig.savefig(os.path.join(OUT, "safe_exploitation.png"), bbox_inches="tight")
    plt.close(fig)
    return eps, results


def fig_rounds_to_detect():
    sigma = 0.8
    edges = [1 / 9, 0.05, 0.02, 0.01]
    N = 20000
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False, gridspec_kw={"wspace": 0.35})
    t = np.arange(1, N + 1)
    for ax, d, col in zip(axes, edges, COLORS):
        style(ax)
        for _ in range(5):
            steps = d + sigma * rng.standard_normal(N)
            ax.plot(t, np.cumsum(steps), color=col, lw=0.8, alpha=0.6)
        ax.plot(t, d * t, color=INK, lw=1.2)
        ax.fill_between(t, d * t - 2 * sigma * np.sqrt(t), d * t + 2 * sigma * np.sqrt(t), color=col, alpha=0.12, lw=0)
        ax.plot(t, 2 * sigma * np.sqrt(t), color=INK2, lw=1, ls="--")
        n_star = (2 * sigma / d) ** 2
        ax.axvline(n_star, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"edge {d:.3f} per round", loc="left", color=INK, fontsize=10)
        ax.text(0.03, 0.97, f"N* = (2 sigma / d)^2 = {n_star:,.0f}", transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.set_xlabel("rounds")
        ax.set_xlim(0, N)
    axes[0].set_ylabel("cumulative score")
    fig.suptitle("Five sample scores of a player with a fixed edge d (sigma = 0.8 per round). Shaded: expected score +-2 sigma sqrt(N). "
                 "Dashed: the +2 sigma sqrt(N) line a lucky player with no edge can reach. Dotted: where the edge clears it.",
                 fontsize=9, color=INK2, y=1.02)
    fig.savefig(os.path.join(OUT, "rounds_to_detect.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import time

    t0 = time.time()
    names_a, names_d, table = round_robin()
    fig_tournament(names_a, names_d, table)
    print("attacker \\ defender".ljust(18), " ".join(n[:9].rjust(9) for n in names_d))
    for n, row in zip(names_a, table):
        print(n.ljust(18), " ".join(f"{v:+9.3f}" for v in row))
    eps, results = fig_safe_exploitation()
    for name, ys in results.items():
        print(name, " ".join(f"{v:+.3f}" for v in ys))
    fig_rounds_to_detect()
    print(f"done in {time.time() - t0:.0f}s")
