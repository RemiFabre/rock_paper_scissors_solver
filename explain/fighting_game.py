"""Fighting games as a chain of situational rock-paper-scissors.

Three parts, all on an illustrative model (no measured data):

(a) A small stochastic game (Shapley, 1953): a handful of situations, each a
    zero-sum matrix whose entries are damage swing plus the value of the next
    situation. Solved by value iteration with the exact solver at each state.
(b) Mechanics versus reading: a player who mis-inputs more often but models the
    opponent, against a player with clean execution and a fixed, biased mix.
(c) Rating players from an encounter log: leak (exploitability of the observed
    mix) and read (gain over equilibrium play against the observed opponent).

Run: python explain/fighting_game.py
"""
import os
import random
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, GRID, INK, INK2, OUT, SURFACE, UNBALANCED, solve_exact, style
from rps_solver import attack_payoffs, best_attack, best_defense, defense_payoffs, exploitability

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.facecolor": SURFACE, "savefig.dpi": 160})

# ----------------------------------------------------------------------------- (a) the stochastic game
# Payoffs are damage swing for player 1, in units of roughly 10%. Illustrative numbers.
# A lost stock is a reward of +-KILL and the match continues from neutral.
KILL = 6.0
GAMMA = 0.9

NEUTRAL, LEDGE, LEDGE2, ADV, DISADV = "neutral", "ledge", "ledge (P1 hanging)", "advantage", "disadvantage"


def build_game(kill_chance=0.25, adv_damage=1.0):
    """Situations of the game: name -> (row options, column options, matrix of (reward, {next: prob}))."""
    # Neutral: a plain cyclic game. The winner gets the advantage, sometimes straight to a ledge trap.
    n_opts = ["approach", "wait", "zone"]
    N = {}
    cyc = {("approach", "zone"): 1, ("zone", "wait"): 1, ("wait", "approach"): 1}
    for a in n_opts:
        for d in n_opts:
            if (a, d) in cyc:
                N[a, d] = (1.0, {ADV: 0.6, LEDGE: 0.4})
            elif (d, a) in cyc:
                N[a, d] = (-1.0, {DISADV: 0.6, LEDGE2: 0.4})
            else:
                N[a, d] = (0.0, {NEUTRAL: 1.0})

    # Ledge: player 2 hangs on the ledge, player 1 stands on stage.
    l_att = ["wait", "cover roll", "cover jump", "hit ledge"]
    l_def = ["neutral getup", "roll", "jump", "attack getup", "ledge dash"]
    L = {
        ("wait", "neutral getup"): (1.0, {ADV: 1.0}),
        ("wait", "roll"): (0.0, {NEUTRAL: 1.0}),
        ("wait", "jump"): (-0.5, {NEUTRAL: 1.0}),
        ("wait", "attack getup"): (1.0, {ADV: 1.0}),  # shielded and punished
        ("wait", "ledge dash"): (-0.5, {NEUTRAL: 1.0}),
        ("cover roll", "neutral getup"): (0.0, {NEUTRAL: 1.0}),
        ("cover roll", "roll"): (1.5, {ADV: 1.0}),
        ("cover roll", "jump"): (-0.5, {NEUTRAL: 1.0}),
        ("cover roll", "attack getup"): (-1.0, {DISADV: 1.0}),
        ("cover roll", "ledge dash"): (-0.5, {NEUTRAL: 1.0}),
        ("cover jump", "neutral getup"): (-0.5, {NEUTRAL: 1.0}),
        ("cover jump", "roll"): (0.0, {NEUTRAL: 1.0}),
        ("cover jump", "jump"): (1.5, {LEDGE: 1.0}),  # sent back offstage
        ("cover jump", "attack getup"): (-1.0, {DISADV: 1.0}),
        ("cover jump", "ledge dash"): (-0.5, {NEUTRAL: 1.0}),
        ("hit ledge", "neutral getup"): (-0.5, {NEUTRAL: 1.0}),
        ("hit ledge", "roll"): (0.0, {NEUTRAL: 1.0}),
        ("hit ledge", "jump"): (0.5, {LEDGE: 1.0}),
        ("hit ledge", "attack getup"): (-1.0, {DISADV: 1.0}),
        ("hit ledge", "ledge dash"): (1.5, {LEDGE: 0.5, "kill": 0.5}),  # caught during the vulnerable frames
    }

    # Advantage: player 1 continues a combo, player 2 tries to escape.
    a_att = ["true combo", "bait dodge", "reset"]
    a_def = ["airdodge", "jump out", "attack out"]
    A = {
        ("true combo", "airdodge"): (adv_damage, {ADV: 0.5, LEDGE: 0.5 - kill_chance, "kill": kill_chance}),
        ("true combo", "jump out"): (0.0, {NEUTRAL: 1.0}),
        ("true combo", "attack out"): (adv_damage, {ADV: 1.0}),
        ("bait dodge", "airdodge"): (2 * adv_damage, {ADV: 0.5, LEDGE: 0.5 - kill_chance, "kill": kill_chance}),
        ("bait dodge", "jump out"): (-0.5, {NEUTRAL: 1.0}),
        ("bait dodge", "attack out"): (-1.0, {DISADV: 1.0}),
        ("reset", "airdodge"): (0.5, {ADV: 1.0}),
        ("reset", "jump out"): (0.5, {LEDGE: 1.0}),
        ("reset", "attack out"): (-0.5, {NEUTRAL: 1.0}),
    }
    # Disadvantage and the other ledge: mirror images, roles swapped, payoffs negated.
    mirror = {NEUTRAL: NEUTRAL, ADV: DISADV, DISADV: ADV, LEDGE: LEDGE2, LEDGE2: LEDGE, "kill": "killed"}

    def mirrored(cells):
        return {(d, a): (-r, {mirror[s]: p for s, p in nxt.items()}) for (a, d), (r, nxt) in cells.items()}

    return {
        NEUTRAL: (n_opts, n_opts, N),
        LEDGE: (l_att, l_def, L),
        LEDGE2: (l_def, l_att, mirrored(L)),
        ADV: (a_att, a_def, A),
        DISADV: (a_def, a_att, mirrored(A)),
    }


def continuation_matrix(game, state, V, gamma=GAMMA, kill=KILL):
    """Matrix of reward + gamma * expected value of the next situation."""
    rows, cols, cells = game[state]
    m = []
    for a in rows:
        row = []
        for d in cols:
            r, nxt = cells[a, d]
            total = r
            for s, p in nxt.items():
                if s == "kill":
                    total += p * (kill + gamma * V[NEUTRAL])
                elif s == "killed":
                    total += p * (-kill + gamma * V[NEUTRAL])
                else:
                    total += p * gamma * V[s]
            row.append(total)
        m.append(row)
    return m


def solve_game(game, gamma=GAMMA, kill=KILL, tol=1e-6, max_iter=500):
    """Value iteration. Returns values and the optimal mixes per situation."""
    V = {s: 0.0 for s in game}
    for _ in range(max_iter):
        newV, sols = {}, {}
        for s in game:
            m = [[Fraction(v).limit_denominator(10000) for v in row] for row in continuation_matrix(game, s, V, gamma, kill)]
            sol = solve_exact(m)
            newV[s] = float(sol.value)
            sols[s] = sol
        delta = max(abs(newV[s] - V[s]) for s in game)
        V = newV
        if delta < tol:
            break
    return V, sols


def fig_ledge(game, V, sols, sweep):
    rows, cols, _ = game[LEDGE]
    m = np.array(continuation_matrix(game, LEDGE, V), float)
    sol = sols[LEDGE]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    ax = axes[0]
    lim = np.abs(m).max()
    im = ax.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_title("Ledge situation: damage swing + value of what follows", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("player 2 (on the ledge)")
    ax.set_ylabel("player 1 (on stage)")
    ax.tick_params(colors=INK2)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=8)

    ax = axes[1]
    style(ax)
    names = [f"P1: {r}" for r in rows] + [f"P2: {c}" for c in cols]
    probs = [float(p) for p in sol.attack] + [float(p) for p in sol.defense]
    colors = [COLORS[0]] * len(rows) + [COLORS[1]] * len(cols)
    y = np.arange(len(names))[::-1]
    ax.barh(y, probs, color=colors, height=0.6)
    for yi, p in zip(y, probs):
        ax.text(p + 0.01, yi, f"{p:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("optimal frequency")
    ax.set_title("Optimal mixes at the ledge", loc="left", fontsize=10, color=INK)
    vtxt = "\n".join(f"{s}: {V[s]:+.2f}" for s in [NEUTRAL, LEDGE, ADV])
    ax.text(0.74, y[len(rows)] + 0.3, "situation values\n(player 1)\n" + vtxt, ha="right", va="top", fontsize=8, color=INK2)

    ax = axes[2]
    style(ax)
    xs = [k for k, _ in sweep]
    for i, r in enumerate(rows):
        ax.plot(xs, [float(s.attack[i]) for _, s in sweep], color=COLORS[i], lw=2, label=r)
    ax.set_xlabel("value of a kill (0: early stock, 8: kill percent)")
    ax.set_ylabel("player 1 frequency at the ledge")
    ax.set_title("Same ledge, different follow-ups, different mix", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Illustrative model: numbers are not measured from any game.", fontsize=9, color=INK2, y=1.0)
    fig.savefig(os.path.join(OUT, "ledge_game.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- (b) mechanics versus reading


class Reader:
    """Column player: recency-weighted frequency model of the opponent, best reply mixed with equilibrium."""

    def __init__(self, m, nash, trust, halflife, rng):
        self.m = m
        self.nash = nash
        self.trust = trust
        self.decay = 0.5 ** (1.0 / halflife) if halflife else 0.0
        self.counts = [1.0] * len(m)  # Laplace prior
        self.rng = rng

    def choose(self):
        total = sum(self.counts)
        model = [c / total for c in self.counts]
        br = best_defense(self.m, model)[0]
        p = [(1 - self.trust) * q + (self.trust if j == br else 0.0) for j, q in enumerate(self.nash)]
        return self.rng.choices(range(len(p)), weights=p)[0]

    def observe(self, a):
        self.counts = [c * self.decay for c in self.counts]
        self.counts[a] += 1.0


def play_rounds(m, attack_mix, reader, e_att, e_read, n, rng):
    """Row player uses a fixed mix, column player is a Reader. Returns the column player's mean payoff."""
    k_a, k_d = len(m), len(m[0])
    total = 0.0
    for _ in range(n):
        a = rng.choices(range(k_a), weights=attack_mix)[0]
        if rng.random() < e_att:
            a = rng.randrange(k_a)  # mis-input
        d = reader.choose()
        if rng.random() < e_read:
            d = rng.randrange(k_d)
        total -= m[a][d]
        reader.observe(a)
    return total / n


def fig_mechanics_vs_reading(n_rounds=3000, seed=0):
    m = [[float(v) for v in row] for row in UNBALANCED]
    sol = solve_exact(UNBALANCED)
    nash_def = [float(p) for p in sol.defense]
    v_def = -float(sol.value)  # the defender's guaranteed value
    gaps = np.linspace(0.0, 0.4, 9)
    trusts = np.linspace(0.0, 1.0, 9)
    biases = [0.15, 0.05]
    results = []
    for bias in biases:
        # Human-like row player: overplays the strong option A1 at the expense of A0.
        attack = [4 / 9 - bias, 3 / 9 + bias, 2 / 9]
        grid = np.zeros((len(trusts), len(gaps)))
        for i, trust in enumerate(trusts):
            for j, gap in enumerate(gaps):
                rng = random.Random(seed + 1000 * i + j)
                reader = Reader(m, nash_def, trust, halflife=200, rng=rng)
                grid[i, j] = play_rounds(m, attack, reader, 0.0, gap, n_rounds, rng) - v_def
        results.append((bias, grid))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    lim = max(np.abs(g).max() for _, g in results)
    for ax, (bias, grid) in zip(axes, results):
        im = ax.imshow(grid, origin="lower", cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto",
                       extent=[gaps[0] - 0.025, gaps[-1] + 0.025, trusts[0] - 0.0625, trusts[-1] + 0.0625])
        ax.contour(gaps, trusts, grid, levels=[0.0], colors=[INK], linewidths=1)
        ax.set_xlabel("reader's extra mis-input rate (mechanics gap)")
        ax.set_ylabel("reader's trust in the opponent model")
        ax.set_title(f"Opponent bias {bias:.2f} (overplays the strong option)", loc="left", fontsize=10, color=INK)
        ax.tick_params(colors=INK2)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("reader's payoff per round above the equilibrium value", color=INK2)
    cb.ax.tick_params(colors=INK2)
    fig.suptitle("Blue: reading wins despite worse execution. Black line: break-even. Illustrative model.",
                 fontsize=9.5, color=INK2, y=0.99)
    fig.savefig(os.path.join(OUT, "mechanics_vs_reading.png"), bbox_inches="tight")
    plt.close(fig)
    return results


# ----------------------------------------------------------------------------- (c) rating from logs


def observed_mix(log, state, who):
    """Frequency of each option played by 'who' (0 = row, 1 = column) in a situation."""
    picks = [entry[1 + who] for entry in log if entry[0] == state]
    n_opts = max(picks) + 1 if picks else 0
    return [picks.count(i) / len(picks) for i in range(n_opts)], len(picks)


def leak_and_read(m, x, y, sol):
    """Per-encounter leak and read for both players, from observed mixes x (row) and y (column).

    leak: what a best-replying opponent gains over the value.
    read: gain over playing the equilibrium mix against the observed opponent.
    """
    v = float(sol.value)
    xs, ys = [float(p) for p in sol.attack], [float(p) for p in sol.defense]
    leak1 = v - float(best_defense(m, x)[1])
    leak2 = float(best_attack(m, y)[1]) - v
    realised = sum(x[i] * y[j] * m[i][j] for i in range(len(m)) for j in range(len(m[0])))
    eq_vs_y = sum(xs[i] * y[j] * m[i][j] for i in range(len(m)) for j in range(len(m[0])))
    x_vs_eq = sum(x[i] * ys[j] * m[i][j] for i in range(len(m)) for j in range(len(m[0])))
    return {"leak": (leak1, leak2), "read": (realised - eq_vs_y, x_vs_eq - realised), "realised": realised}


def simulate_log(game, V, sols, n=6000, seed=1):
    """Two players with per-situation biases. Situations are sampled from the chain itself."""
    rng = random.Random(seed)
    mats = {s: [[float(v) for v in row] for row in continuation_matrix(game, s, V)] for s in game}
    # Player 1 leaks at the ledge (waits too much) and reads player 2 in neutral (waits more).
    # Player 2 leaks in neutral (approaches too much) and reads player 1 at the ledge (attack getup more).
    def biased(p, shifts):
        q = [float(v) for v in p]
        for i, dv in shifts.items():
            q[i] += dv
        q = [max(v, 0.0) for v in q]
        return [v / sum(q) for v in q]

    mix1 = {s: [float(p) for p in sols[s].attack] for s in game}
    mix2 = {s: [float(p) for p in sols[s].defense] for s in game}
    mix1[LEDGE] = biased(mix1[LEDGE], {0: 0.2, 1: -0.1, 2: -0.1})
    mix1[NEUTRAL] = biased(mix1[NEUTRAL], {1: 0.15, 2: -0.15})  # a read: wait beats approach
    mix2[NEUTRAL] = biased(mix2[NEUTRAL], {0: 0.15, 1: -0.15})
    mix2[LEDGE] = biased(mix2[LEDGE], {1: -0.1, 3: 0.1})

    log = []
    state = NEUTRAL
    for _ in range(n):
        rows, cols, cells = game[state]
        a = rng.choices(range(len(rows)), weights=mix1[state])[0]
        d = rng.choices(range(len(cols)), weights=mix2[state])[0]
        log.append((state, a, d, mats[state][a][d]))
        nxt = cells[rows[a], cols[d]][1]
        s = rng.choices(list(nxt), weights=list(nxt.values()))[0]
        state = NEUTRAL if s in ("kill", "killed") else s
    return log, mats


def rate_players(log, mats, sols):
    out = {}
    for s in mats:
        x, n1 = observed_mix(log, s, 0)
        y, _ = observed_mix(log, s, 1)
        x += [0.0] * (len(mats[s]) - len(x))
        y += [0.0] * (len(mats[s][0]) - len(y))
        r = leak_and_read(mats[s], x, y, sols[s])
        r["n"] = n1
        r["mix"] = (x, y)
        out[s] = r
    return out


def fig_rating(rating):
    states = [NEUTRAL, LEDGE, LEDGE2, ADV, DISADV]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    w = 0.36
    for ax, key, title in zip(axes, ["leak", "read"], ["Leak: what a perfect reader could take, per encounter",
                                                       "Read: gain over equilibrium play against this opponent"]):
        style(ax)
        xs = np.arange(len(states))
        for k, (who, col) in enumerate([("player 1", COLORS[0]), ("player 2", COLORS[1])]):
            vals = [rating[s][key][k] for s in states]
            bars = ax.bar(xs + (k - 0.5) * w, vals, w, color=col, label=who)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + w / 2, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(xs, [f"{s}\n(n={rating[s]['n']})" for s in states], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("damage swing units per encounter")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Two simulated players rated from an encounter log. Illustrative model.", fontsize=9.5, color=INK2, y=1.0)
    fig.savefig(os.path.join(OUT, "player_rating.png"), bbox_inches="tight")
    plt.close(fig)


def leak_estimation_bias(m, sol, sizes=(30, 100, 300, 1000, 3000), trials=200, seed=3):
    """Exploitability of an empirical mix sampled from the exact optimum: a sample-size artefact."""
    rng = random.Random(seed)
    x = [float(p) for p in sol.attack]
    out = {}
    for n in sizes:
        vals = []
        for _ in range(trials):
            picks = rng.choices(range(len(x)), weights=x, k=n)
            emp = [picks.count(i) / n for i in range(len(x))]
            vals.append(float(exploitability(m, emp, sol.value)))
        out[n] = float(np.mean(vals))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    game = build_game()
    V, sols = solve_game(game)
    print("Situation values (player 1):", {s: round(V[s], 3) for s in game})
    print("Ledge mixes:", [f"{float(p):.2f}" for p in sols[LEDGE].attack], [f"{float(p):.2f}" for p in sols[LEDGE].defense])
    imm = solve_exact([[Fraction(game[LEDGE][2][a, d][0]).limit_denominator(100) for d in game[LEDGE][1]] for a in game[LEDGE][0]])
    print("Ledge mix on immediate damage only:", [f"{float(p):.2f}" for p in imm.attack],
          [f"{float(p):.2f}" for p in imm.defense], "value", round(float(imm.value), 3))
    sweep = []
    for s in game:
        print(f"  {s:20s} V={V[s]:+.3f} P1", [f"{float(p):.2f}" for p in sols[s].attack], "P2", [f"{float(p):.2f}" for p in sols[s].defense])
    for k in np.linspace(0.0, 8.0, 17):
        Vk, sk = solve_game(game, kill=float(k))
        sweep.append((float(k), sk[LEDGE]))
        if k in (0.0, 2.0, 4.0, 6.0, 8.0):
            print(f"  kill worth {k:.1f}: ledge P1 mix", [f"{float(p):.2f}" for p in sk[LEDGE].attack],
                  "P2 mix", [f"{float(p):.2f}" for p in sk[LEDGE].defense], "V(ledge)", round(Vk[LEDGE], 2))
    fig_ledge(game, V, sols, sweep)

    results = fig_mechanics_vs_reading()
    for bias, grid in results:
        print(f"bias {bias}: reader edge at trust=1, gap=0: {grid[-1, 0]:+.3f}; at trust=1, gap=0.4: {grid[-1, -1]:+.3f};"
              f" at trust=0, gap=0.2: {grid[0, 4]:+.3f}")

    log, mats = simulate_log(game, V, sols)
    rating = rate_players(log, mats, sols)
    for s, r in rating.items():
        print(f"{s:13s} n={r['n']:5d} leak P1 {r['leak'][0]:+.3f} P2 {r['leak'][1]:+.3f}  read P1 {r['read'][0]:+.3f} P2 {r['read'][1]:+.3f}")
    fig_rating(rating)
    print("Leak measured on a perfectly optimal player, by sample size:", leak_estimation_bias(mats[LEDGE], sols[LEDGE]))
    print("done")


if __name__ == "__main__":
    main()
