# coding: utf-8
"""Symbolic solver: find the attack mix that makes every defence equally good.

Usage:
    python solver.py                       # solves the default (unbalanced) game
    python solver.py "1 0 -1; -1 1 1; 0 -1 1"
    python solver.py "[[1.5, 0, -1], [-1.5, 1, 1], [0, -1, 1]]"

Rows are attacks, columns are defences, entries are the attacker's payoff.
The indifference method used here is only valid when the optimal play uses every
option on both sides. The result is therefore checked, and when the check fails the
general solver of rps_solver.py is used instead.
"""
import sys

from sympy import Matrix, Rational, collect, expand, init_printing, symbols

from rps_solver import is_optimal_pair, parse_matrix, solve_exact

# First row is A0, second row is A1, etc. If the attacker does A0 and the defender does D1, the result is m.row(0)[1].
m = Matrix([[1, 0, -1], [-1, 1, 1], [0, -1, 1]])


def create_symbols(letter, nb):
    return symbols(" ".join(letter + str(i) for i in range(nb)))


def indifference_mix(matrix, own_syms, other_syms, sign):
    """Mix over own_syms that makes every option of the other player equally good.

    sign is +1 when solving for the attacker (rows), -1 for the defender (columns).
    Returns (solution dict, list of the other player's coefficients).
    """
    poly = 0
    # Calculating the expected value of the match symbolically.
    # e.g. Matrix([[1, 0, -1], [-1, 1, 1], [0, -1, 1]]) gives:
    # poly = A0 * (D0 - D2) + A1 * (-D0 + D1 + D2) + A2 * (-D1 + D2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            a, d = (own_syms[i], other_syms[j]) if sign > 0 else (other_syms[i], own_syms[j])
            poly += a * d * matrix.row(i)[j]
    collected = collect(expand(poly), other_syms)
    coeffs = [collected.coeff(s) for s in other_syms]
    equations = []
    for i in range(len(coeffs) - 1):
        # All of the other player's choices should be equally impactful for this mix to be optimal
        equations.append(coeffs[i] - coeffs[i + 1])
    # The last equation is just A0+A1+...+An = 1
    equations.append(sum(own_syms) - 1)
    print("equations = {}".format(equations))
    return Matrix(equations).jacobian(own_syms), equations, coeffs


def solve_side(matrix, own_syms, other_syms, sign):
    from sympy import solve

    _, equations, coeffs = indifference_mix(matrix, own_syms, other_syms, sign)
    results = solve(equations, own_syms, dict=True)
    if len(results) != 1 or any(v.free_symbols for v in results[0].values()):
        return None, coeffs
    return [results[0][s] for s in own_syms], coeffs


def main():
    global m
    init_printing()
    if len(sys.argv) > 1:
        m = Matrix([[Rational(v) for v in row] for row in parse_matrix(" ".join(sys.argv[1:]))])
    print("Matrix (rows are attacks, cols are defences) {}".format(m))

    nb_attacks = m.shape[0]
    nb_defs = m.shape[1]

    # Creating A0, A1, ..., Am variables. These represent the probability of doing each attack.
    asyms = create_symbols("A", nb_attacks)
    # Creating D0, D1, ..., Dn variables. These represent the probability of doing each defence.
    dsyms = create_symbols("D", nb_defs)

    attack, dcoeffs = solve_side(m, asyms, dsyms, +1)
    print("attack = {}".format(attack))
    # The same trick gives the defence mix that makes every attack equally good.
    defense, acoeffs = solve_side(m, dsyms, asyms, -1)
    print("defense = {}".format(defense))

    certified = attack is not None and defense is not None and is_optimal_pair(m.tolist(), attack, defense)
    if certified:
        # If the attack is optimal, then any defence yields the same expected value. We can, for example,
        # evaluate one of the "coeffs" with the found attack to calculate the EV:
        value = dcoeffs[0].subs(dict(zip(asyms, attack)))
        print("Expected value under optimal play from both sides (+ means attack wins) = {}".format(value))
    else:
        print(
            "The indifference method does not apply here: the optimal play does not use every option.\n"
            "Falling back to the general solver (rps_solver.py):"
        )
        print(solve_exact(m.tolist()))


if __name__ == "__main__":
    main()
