"""Animations for the README. Run: python explain/animations.py  (writes GIFs in images/)"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from common import COLORS, GRID, INK, INK2, INTUITIVE, OPTIMAL, OUT, SURFACE, UNBALANCED, UNIFORM, draw_triangle, style, tern

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.facecolor": SURFACE})
M = np.asarray(UNBALANCED, float)
rng = np.random.default_rng(3)


def fictitious_defender(counts):
    """Best reply to the attacker's observed frequencies (ties broken randomly)."""
    freq = counts / counts.sum()
    p = freq @ M  # attacker's payoff per defence
    best = np.flatnonzero(p == p.min())
    return rng.choice(best)


def simulate_being_read(phases, rounds_per_phase=300):
    """The attacker plays a fixed mix per phase, the defender best-replies to what it has seen."""
    counts = np.ones(3)  # small prior so the first replies are not extreme
    score, att_mix, def_hist = [0.0], [], []
    recent = []
    for name, mix in phases:
        for _ in range(rounds_per_phase):
            a = rng.choice(3, p=mix)
            d = fictitious_defender(counts)
            counts[a] += 1
            recent.append(d)
            score.append(score[-1] + M[a, d])
            att_mix.append(mix)
            window = recent[-50:]
            def_hist.append(np.bincount(window, minlength=3) / len(window))
    return np.array(score), np.array(att_mix), np.array(def_hist)


def anim_being_read():
    phases = [("intuitive 25/50/25", INTUITIVE), ("uniform 1/3 each", UNIFORM), ("optimal 4/9 3/9 2/9", OPTIMAL)]
    R = 300
    score, att, dfn = simulate_being_read(phases, R)
    T = len(att)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    fig.subplots_adjust(top=0.8, bottom=0.14, wspace=0.3)
    style(ax1)
    style(ax2)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-0.6, 5.6)
    ax1.set_xticks([0, 1, 2, 3.5, 4.5, 5.5], ["A0", "A1", "A2", "D0", "D1", "D2"])
    ax1.set_ylabel("frequency")
    ax1.set_title("attacker's mix (left)   defender's last 50 plays (right)", fontsize=9, loc="left", color=INK2)
    bars_a = ax1.bar([0, 1, 2], att[0], color=COLORS[:3], width=0.7)
    bars_d = ax1.bar([3.5, 4.5, 5.5], dfn[0], color=COLORS[:3], width=0.7, alpha=0.5, hatch="//", edgecolor=SURFACE)
    ax2.set_xlim(0, T)
    ax2.set_ylim(min(score) - 10, max(score) + 10)
    ax2.set_xlabel("round")
    ax2.set_ylabel("attacker's cumulative score")
    for k, (name, _) in enumerate(phases):
        ax2.axvspan(k * R, (k + 1) * R, color=GRID if k % 2 else SURFACE, alpha=0.5, lw=0)
        ax2.text(k * R + R / 2, max(score) + 6, name, ha="center", fontsize=8.5, color=INK)
    ax2.plot(np.arange(T + 1), score, color=GRID, lw=1)
    (line,) = ax2.plot([], [], color=INK, lw=2)
    dot = ax2.scatter([], [], color=INK, s=30, zorder=5)
    title = fig.suptitle("", fontsize=10.5, color=INK, y=0.97)
    ax2.axhline(0, color=INK2, lw=0.8)
    # guide slope for the value
    x0 = 2 * R
    ax2.plot([x0, T], [score[x0], score[x0] + (T - x0) / 9], color=COLORS[0], lw=1, ls=":")
    ax2.text(T, score[x0] + (T - x0) / 9, " +1/9 per round", fontsize=8, color=COLORS[0], va="center", ha="left")
    ax2.set_xlim(0, T * 1.18)
    msgs = [
        "The defender learns the attacker's habits and plays the counter (D0 punishes A1). Score goes down.",
        "Uniform play cannot be punished, but it cannot win either: the defender holds it at zero.",
        "The optimal mix wins 1/9 per round on average, whatever the defender does.",
    ]

    def update(f):
        t = int(f * T / 120) + 1
        t = min(t, T)
        for b, v in zip(bars_a, att[t - 1]):
            b.set_height(v)
        for b, v in zip(bars_d, dfn[t - 1]):
            b.set_height(v)
        line.set_data(np.arange(t + 1), score[: t + 1])
        dot.set_offsets([[t, score[t]]])
        k = min((t - 1) // R, 2)
        title.set_text(msgs[k])
        return (*bars_a, *bars_d, line, dot, title)

    anim = FuncAnimation(fig, update, frames=121, interval=80)
    anim.save(os.path.join(OUT, "being_read.gif"), writer=PillowWriter(fps=12), dpi=110)
    plt.close(fig)


def fictitious_play(rounds=4000):
    """Both players best-reply to the other's history. The averages converge to the optimum."""
    ca, cd = np.ones(3), np.ones(3)
    avg_a, avg_d, vals = [], [], []
    for _ in range(rounds):
        pa = M @ (cd / cd.sum())  # each attack against the defender's history
        a = rng.choice(np.flatnonzero(pa == pa.max()))
        pd = (ca / ca.sum()) @ M
        d = rng.choice(np.flatnonzero(pd == pd.min()))
        ca[a] += 1
        cd[d] += 1
        avg_a.append(ca / ca.sum())
        avg_d.append(cd / cd.sum())
        vals.append((ca / ca.sum()) @ M @ (cd / cd.sum()))
    return np.array(avg_a), np.array(avg_d), np.array(vals)


def anim_convergence():
    avg_a, avg_d, vals = fictitious_play()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.25)
    for ax, labels in zip(axes[:2], (["A0", "A1", "A2"], ["D0", "D1", "D2"])):
        ax.set_facecolor(SURFACE)
        draw_triangle(ax, labels)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
    axes[0].set_title("attacker's average mix", loc="left", color=INK, fontsize=10)
    axes[1].set_title("defender's average mix", loc="left", color=INK, fontsize=10)
    for ax, opt in zip(axes[:2], (OPTIMAL, [4 / 9, 2 / 9, 3 / 9])):
        p = tern(opt)
        ax.scatter(p[0], p[1], marker="*", s=180, color=INK, zorder=5)
    xa, xd = tern(avg_a), tern(avg_d)
    (la,) = axes[0].plot([], [], color=COLORS[0], lw=1.2)
    (ld,) = axes[1].plot([], [], color=COLORS[1], lw=1.2)
    da = axes[0].scatter([], [], color=COLORS[0], s=30, zorder=6)
    dd = axes[1].scatter([], [], color=COLORS[1], s=30, zorder=6)
    ax = axes[2]
    style(ax)
    ax.set_xscale("log")
    ax.set_xlim(1, len(vals))
    ax.set_ylim(-0.3, 0.5)
    ax.axhline(1 / 9, color=INK2, lw=0.8, ls=":")
    ax.text(len(vals), 1 / 9, " 1/9", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("round (log scale)")
    ax.set_title("expected payoff of the average mixes", loc="left", color=INK, fontsize=10)
    (lv,) = ax.plot([], [], color=INK, lw=1.5)
    txt = fig.suptitle("", fontsize=10.5, color=INK, y=0.97)
    frames = np.unique(np.geomspace(1, len(vals), 110).astype(int))

    def update(i):
        t = frames[i]
        la.set_data(xa[:t, 0], xa[:t, 1])
        ld.set_data(xd[:t, 0], xd[:t, 1])
        da.set_offsets([xa[t - 1]])
        dd.set_offsets([xd[t - 1]])
        lv.set_data(np.arange(1, t + 1), vals[:t])
        txt.set_text(f"Round {t}: each player replies to the other's history. "
                     f"Attacker avg [{avg_a[t-1][0]:.2f}, {avg_a[t-1][1]:.2f}, {avg_a[t-1][2]:.2f}]  "
                     f"defender avg [{avg_d[t-1][0]:.2f}, {avg_d[t-1][1]:.2f}, {avg_d[t-1][2]:.2f}]")
        return la, ld, da, dd, lv, txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=90)
    anim.save(os.path.join(OUT, "convergence.gif"), writer=PillowWriter(fps=11), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    anim_being_read()
    anim_convergence()
    print("done")
