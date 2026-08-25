from fractions import Fraction as F

import pytest

from rps_solver import (
    best_defense,
    exploitability,
    is_optimal_pair,
    parse_matrix,
    solve_exact,
    solve_indifference,
)

BALANCED = [[1, 0, -1], [-1, 1, 0], [0, -1, 1]]
UNBALANCED = [[1, 0, -1], [-1, 1, 1], [0, -1, 1]]


def test_balanced_game():
    s = solve_exact(BALANCED)
    assert s.attack == [F(1, 3)] * 3
    assert s.defense == [F(1, 3)] * 3
    assert s.value == 0


def test_unbalanced_game_matches_readme():
    s = solve_exact(UNBALANCED)
    assert s.attack == [F(4, 9), F(3, 9), F(2, 9)]
    assert s.defense == [F(4, 9), F(2, 9), F(3, 9)]
    assert s.value == F(1, 9)
    assert solve_indifference(UNBALANCED).attack == s.attack


def test_high_risk_high_reward_example():
    s = solve_exact([[1.5, 0, -1], [-1.5, 1, 1], [0, -1, 1]])
    assert s.attack == [F(10, 23), F(8, 23), F(5, 23)]
    assert s.value == F(3, 23)


def test_dominated_option_is_not_played():
    # A1 is A0 with a worse column: never played.
    m = [[1, 0, -1], [0, 0, -1], [-1, 1, 0], [0, -1, 1]]
    s = solve_exact(m)
    assert s.attack[1] == 0
    assert is_optimal_pair(m, s.attack, s.defense)
    assert solve_indifference(m) is None


def test_indifference_method_rejects_invalid_solution():
    # The linear system has a solution with a negative "probability".
    m = [[0, 1, -1], [1, 1, -1], [0, 1, 0]]
    assert solve_indifference(m) is None
    s = solve_exact(m)
    assert s.value == 0 and s.attack == [0, 0, 1]


def test_non_square():
    m = [[1, -1, 0, 2], [-1, 1, 1, -1]]
    s = solve_exact(m)
    assert is_optimal_pair(m, s.attack, s.defense)
    assert s.attack == [F(1, 2), F(1, 2)] and s.value == 0


def test_matching_pennies_and_pure_game():
    assert solve_exact([[1, -1], [-1, 1]]).value == 0
    s = solve_exact([[3, 5], [1, 2]])  # row 0 dominates, column 0 dominates
    assert s.attack == [1, 0] and s.defense == [1, 0] and s.value == 3


def test_exploitability_of_intuitive_mix():
    # Playing the strongest attack most is exploitable in the unbalanced game.
    naive = [F(1, 4), F(1, 2), F(1, 4)]
    assert exploitability(UNBALANCED, naive) > 0
    assert exploitability(UNBALANCED, [F(4, 9), F(3, 9), F(2, 9)]) == 0
    assert best_defense(UNBALANCED, naive)[0] == 0  # D0 punishes A1


def test_parse_matrix():
    assert parse_matrix("1 0 -1; -1 1 1; 0 -1 1") == [[1, 0, -1], [-1, 1, 1], [0, -1, 1]]
    assert parse_matrix("[[1.5, 0], [0, 1]]")[0][0] == F(3, 2)


@pytest.mark.parametrize("seed", range(20))
def test_random_games_are_certified(seed):
    import random

    rng = random.Random(seed)
    n, k = rng.randint(1, 4), rng.randint(1, 4)
    m = [[rng.randint(-2, 2) for _ in range(k)] for _ in range(n)]
    s = solve_exact(m)
    assert is_optimal_pair(m, s.attack, s.defense)
