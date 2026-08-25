import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "explain"))

from fighting_game import (  # noqa: E402
    ADV,
    DISADV,
    LEDGE,
    NEUTRAL,
    build_game,
    continuation_matrix,
    leak_and_read,
    observed_mix,
    solve_game,
)
from rps_solver import solve_exact  # noqa: E402

UNBALANCED = [[1, 0, -1], [-1, 1, 1], [0, -1, 1]]


def test_value_iteration_converges_to_valid_mixes():
    game = build_game()
    V, sols = solve_game(game)
    for s, sol in sols.items():
        assert abs(sum(sol.attack) - 1) < 1e-9 and min(sol.attack) >= 0
        assert abs(sum(sol.defense) - 1) < 1e-9 and min(sol.defense) >= 0
    # The game is symmetric: neutral is even, advantage mirrors disadvantage.
    assert abs(V[NEUTRAL]) < 1e-4
    assert abs(V[ADV] + V[DISADV]) < 1e-4
    assert V[LEDGE] > 0  # the player on stage is favoured


def test_equilibrium_has_no_leak_and_no_read():
    game = build_game()
    V, sols = solve_game(game)
    m = continuation_matrix(game, LEDGE, V)
    sol = solve_exact([[round(v, 6) for v in row] for row in m])
    r = leak_and_read(m, [float(p) for p in sol.attack], [float(p) for p in sol.defense], sol)
    assert all(abs(v) < 1e-4 for v in r["leak"])
    assert all(abs(v) < 1e-4 for v in r["read"])


def test_leak_and_read_on_known_mixes():
    sol = solve_exact(UNBALANCED)
    m = [[float(v) for v in row] for row in UNBALANCED]
    x = [0.25, 0.5, 0.25]
    # Against the equilibrium defence: the attacker leaks 0.36 but nothing is read either way.
    r = leak_and_read(m, x, [float(p) for p in sol.defense], sol)
    assert abs(r["leak"][0] - (1 / 9 + 0.25)) < 1e-9
    assert abs(r["leak"][1]) < 1e-9
    assert abs(r["read"][0]) < 1e-9 and abs(r["read"][1]) < 1e-9
    # Against a defender who always plays D0 (the best reply): the defender's read is the whole leak.
    r = leak_and_read(m, x, [1.0, 0.0, 0.0], sol)
    assert abs(r["read"][1] - (1 / 9 + 0.25)) < 1e-9
    assert abs(r["read"][0] + (1 / 9 + 0.25)) < 1e-9


def test_observed_mix_counts_options():
    log = [("s", 0, 1, 0.0), ("s", 2, 1, 0.0), ("t", 1, 0, 0.0), ("s", 0, 0, 0.0)]
    x, n = observed_mix(log, "s", 0)
    y, _ = observed_mix(log, "s", 1)
    assert n == 3 and x == [2 / 3, 0.0, 1 / 3] and y == [1 / 3, 2 / 3]
