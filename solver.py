# coding: utf-8
from sympy import *
from sympy import roots, solve_poly_system
from sympy import init_printing

# First row is A0, second raw is A1, etc. If the attacker does A0 and the defender does D1, the result is m.row(0)[1].
m = Matrix([[1, 0, -1], [-1, 1, 1], [0, -1, 1]])
# print(m.row(0)[1])


def create_symbols(letter, nb):
    result = ""
    for i in range(nb):
        result += letter + str(i) + " "
    return symbols(result)


def main():
    global m
    init_printing()
    print("Matrix (rows are attacks, cols are defences) {}".format(m))

    nb_attacks = m.shape[0]
    nb_defs = m.shape[1]

    # Creating A0, A1, ..., Am variables. These represent the probability of doing each attack.
    asyms = create_symbols("A", nb_attacks)
    # Creating D0, D1, ..., Dn variables. These represent the probability of doing each defence.
    dsyms = create_symbols("D", nb_defs)
    print(asyms)
    print(dsyms)
    poly = 0
    # Calculating the expected value of the match symbolically.
    # e.g. Matrix([[1, 0, -1], [-1, 1, 1], [0, -1, 1]]) gives:
    # poly = A1 * (D1 - D3) + A2 * (-D1 + D2 + D3) + A3 * (-D2 + D3)
    for i in range(m.shape[0]):
        # print("Attack {} has values {}".format(i, m.row(i)))
        for j in range(m.shape[1]):
            poly += asyms[i] * dsyms[j] * m.row(i)[j]
    print("Expected value (base) = {}".format(poly))
    expanded = expand(poly)
    collected = collect(expanded, dsyms)

    print("expanded = {}".format(expanded))
    print("collected = {}".format(collected))
    coeffs = []
    for i in range(nb_defs):
        coeffs.append(collected.coeff(dsyms[i]))
    print("coeffs = {}".format(coeffs))
    equations = []
    for i in range(len(coeffs) - 1):
        # All of the defender choices should be equally impactful for this to be the optimal attack
        equations.append(coeffs[i] - coeffs[i + 1])
    last_eq = -1
    for i in range(nb_attacks):
        # The last equation is just A0+A1+...+An = 1
        last_eq += asyms[i]
    equations.append(last_eq)
    print("equations = {}".format(equations))

    # Solving the set of linear equations (this will give the optimal attack)
    results = solve(equations)
    print("results = {}".format(results))
    # If the attack is optimal, then any defences yield the same expected value. We can, for example, evaluate one of the "coeffs" with the found attack to calculate the EV:
    print(
        "Expected value under optimal play from both sides (+ means attack wins) = {}".format(
            coeffs[0].subs(results)
        )
    )


if __name__ == "__main__":
    main()

