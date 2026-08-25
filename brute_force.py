"""Brute force check: grid search over both players' mixes.

Usage:
    python brute_force.py                       # default (unbalanced) game
    python brute_force.py "1 0 -1; -1 1 1; 0 -1 1" [steps]

For every attack mix on the grid, find the defence that hurts it most; keep the attack
whose worst case is best. This is the definition of optimal play used in this project.
"""
import sys
from itertools import combinations

from rps_solver import parse_matrix

nb_steps = 50


def grid(n, steps):
    """All mixes over n options whose probabilities are multiples of 1/steps."""
    for cuts in combinations(range(steps + n - 1), n - 1):
        parts = []
        prev = -1
        for c in list(cuts) + [steps + n - 1]:
            parts.append(c - prev - 1)
            prev = c
        yield [p / steps for p in parts]


def expected_value(m, attack, defense):
    return sum(a * d * m[i][j] for i, a in enumerate(attack) for j, d in enumerate(defense))


def main():
    m = [[float(v) for v in row] for row in parse_matrix(" ".join(sys.argv[1:2]) or "1 0 -1; -1 1 1; 0 -1 1")]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else nb_steps
    defenses = list(grid(len(m[0]), steps))

    max_value = -float("inf")
    best_attack = best_defence_global = None
    for attack in grid(len(m), steps):
        # Worst case for this attack
        best_defence = min(defenses, key=lambda d: expected_value(m, attack, d))
        min_value = expected_value(m, attack, best_defence)
        if min_value > max_value:
            max_value = min_value
            best_attack = attack
            best_defence_global = best_defence
    print(
        "Done !\nBest attack: {}\nBest defence against it: {}\nExpected value: {}".format(
            best_attack, best_defence_global, max_value
        )
    )


if __name__ == "__main__":
    main()
