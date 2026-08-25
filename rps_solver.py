"""Exact solver for generalised rock-paper-scissors (two-player zero-sum matrix games).

Rows are the attacker's options, columns the defender's options, entries are the
attacker's payoff (positive means the attacker wins). The defender's payoff is the
opposite, so the game is zero-sum.

Two solvers are provided:

* ``solve_exact`` enumerates square sub-matrices (Shapley and Snow, 1950) and returns
  an optimal pair of mixed strategies with rational probabilities. It handles
  non-square matrices, dominated options and degenerate games.
* ``solve_indifference`` reproduces the original approach of ``solver.py``: find the
  attack mix that makes every defence equally good. It is only correct when the
  equilibrium uses every option on both sides, and it says so when that fails.

Only the standard library is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Sequence

Number = int | float | Fraction
Matrix = Sequence[Sequence[Number]]


@dataclass
class Solution:
    attack: list[Fraction]  # optimal mix for the row player
    defense: list[Fraction]  # optimal mix for the column player
    value: Fraction  # expected payoff per round under optimal play
    method: str

    def __str__(self) -> str:
        return (
            f"attack  = {fmt(self.attack)}\n"
            f"defense = {fmt(self.defense)}\n"
            f"value   = {self.value} ({float(self.value):+.4f} per round, + means attack wins)\n"
            f"method  = {self.method}"
        )


def fmt(v: Sequence[Fraction]) -> str:
    return "[" + ", ".join(str(x) for x in v) + "]"


def to_fractions(m: Matrix) -> list[list[Fraction]]:
    return [[Fraction(x).limit_denominator(10**9) for x in row] for row in m]


# --------------------------------------------------------------------------- basics


def payoff(m: Matrix, x: Sequence[Number], y: Sequence[Number]) -> Fraction:
    """Expected payoff x^T m y."""
    m = to_fractions(m)
    return sum(Fraction(x[i]) * Fraction(y[j]) * m[i][j] for i in range(len(m)) for j in range(len(m[0])))


def defense_payoffs(m: Matrix, x: Sequence[Number]) -> list[Fraction]:
    """Attacker's expected payoff against each pure defence, given attack mix x."""
    m = to_fractions(m)
    return [sum(Fraction(x[i]) * m[i][j] for i in range(len(m))) for j in range(len(m[0]))]


def attack_payoffs(m: Matrix, y: Sequence[Number]) -> list[Fraction]:
    """Attacker's expected payoff of each pure attack, given defence mix y."""
    m = to_fractions(m)
    return [sum(Fraction(y[j]) * m[i][j] for j in range(len(m[0]))) for i in range(len(m))]


def best_defense(m: Matrix, x: Sequence[Number]) -> tuple[int, Fraction]:
    """The pure defence that hurts attack mix x most, and the attacker's payoff against it."""
    p = defense_payoffs(m, x)
    j = min(range(len(p)), key=lambda k: p[k])
    return j, p[j]


def best_attack(m: Matrix, y: Sequence[Number]) -> tuple[int, Fraction]:
    """The pure attack that does best against defence mix y, and its payoff."""
    p = attack_payoffs(m, y)
    i = max(range(len(p)), key=lambda k: p[k])
    return i, p[i]


def exploitability(m: Matrix, x: Sequence[Number], value: Number | None = None) -> Fraction:
    """How much a defender who knows x gains, per round, compared to the game value.

    Zero if and only if x is optimal.
    """
    if value is None:
        value = solve_exact(m).value
    return Fraction(value) - best_defense(m, x)[1]


def is_optimal_pair(m: Matrix, x: Sequence[Number], y: Sequence[Number]) -> bool:
    """True if (x, y) is a pair of optimal strategies (a Nash equilibrium)."""
    m = to_fractions(m)
    x = [Fraction(v) for v in x]
    y = [Fraction(v) for v in y]
    if any(v < 0 for v in x) or any(v < 0 for v in y):
        return False
    if sum(x) != 1 or sum(y) != 1:
        return False
    v = payoff(m, x, y)
    # x guarantees at least v against every defence, y concedes at most v to every attack
    return all(p >= v for p in defense_payoffs(m, x)) and all(p <= v for p in attack_payoffs(m, y))


# --------------------------------------------------------------------- linear algebra


def _solve_linear(a: list[list[Fraction]], b: list[Fraction]) -> list[Fraction] | None:
    """Solve a x = b exactly by Gaussian elimination. None if singular."""
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [v / p for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [rv - f * cv for rv, cv in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def _kernel_solution(m: list[list[Fraction]], rows: tuple[int, ...], cols: tuple[int, ...]):
    """Candidate optimal pair from the square sub-matrix m[rows][cols] (Shapley-Snow).

    On that sub-matrix the attack mix makes every listed defence equally good and the
    defence mix makes every listed attack equally good. Returns None when the
    sub-matrix does not define a unique candidate.
    """
    k = len(rows)
    sub = [[m[i][j] for j in cols] for i in rows]
    # Unknowns: x_S (k values) and v. Equations: sub^T x = v 1, sum x = 1.
    a = [[sub[i][j] for i in range(k)] + [Fraction(-1)] for j in range(k)]
    a.append([Fraction(1)] * k + [Fraction(0)])
    b = [Fraction(0)] * k + [Fraction(1)]
    sol = _solve_linear(a, b)
    if sol is None:
        return None
    xs, v = sol[:k], sol[k]
    a = [[sub[i][j] for j in range(k)] + [Fraction(-1)] for i in range(k)]
    a.append([Fraction(1)] * k + [Fraction(0)])
    sol = _solve_linear(a, b)
    if sol is None:
        return None
    ys = sol[:k]
    x = [Fraction(0)] * len(m)
    y = [Fraction(0)] * len(m[0])
    for i, r in enumerate(rows):
        x[r] = xs[i]
    for j, c in enumerate(cols):
        y[c] = ys[j]
    return x, y, v


# ---------------------------------------------------------------------------- solvers


def solve_indifference(m: Matrix) -> Solution | None:
    """The original method of ``solver.py``: equalise every defence (and every attack).

    Returns the solution when it is certified optimal, None otherwise. It fails when
    the optimal play leaves some option unused (dominated or simply unneeded options).
    """
    m = to_fractions(m)
    n, k = len(m), len(m[0])
    if n != k:
        return None
    res = _kernel_solution(m, tuple(range(n)), tuple(range(k)))
    if res is None:
        return None
    x, y, v = res
    if is_optimal_pair(m, x, y):
        return Solution(x, y, v, "indifference (every option used)")
    return None


def solve_exact(m: Matrix) -> Solution:
    """Optimal strategies for both players, with exact rational probabilities.

    Enumerates square sub-matrices from the largest down (so fully mixed solutions are
    preferred when several optimal strategies exist) and returns the first candidate
    that is certified optimal against the full matrix.
    """
    m = to_fractions(m)
    n, k = len(m), len(m[0])
    if n == 0 or k == 0:
        raise ValueError("empty matrix")
    for size in range(min(n, k), 0, -1):
        for rows in combinations(range(n), size):
            for cols in combinations(range(k), size):
                res = _kernel_solution(m, rows, cols)
                if res is None:
                    continue
                x, y, v = res
                if is_optimal_pair(m, x, y):
                    na, nd = sum(1 for p in x if p > 0), sum(1 for p in y if p > 0)
                    used = "every option used" if na == n and nd == k else f"{na} of {n} attacks, {nd} of {k} defences used"
                    return Solution(x, y, v, f"kernel enumeration ({used})")
    raise RuntimeError("no optimal pair found (this should not happen for a finite matrix game)")


def parse_matrix(text: str) -> list[list[Fraction]]:
    """Parse '1 0 -1; -1 1 1; 0 -1 1' or '[[1,0,-1],[-1,1,1],[0,-1,1]]' into a matrix."""
    text = text.strip()
    if text.startswith("["):
        import ast

        return to_fractions(ast.literal_eval(text))
    rows = [r for r in text.replace("\n", ";").split(";") if r.strip()]
    return [[Fraction(v) for v in r.replace(",", " ").split()] for r in rows]


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "1 0 -1; -1 1 1; 0 -1 1"
    print(solve_exact(parse_matrix(text)))
