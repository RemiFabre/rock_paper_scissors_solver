# Verification of the original solver

A review of the 2020 code and README, with every claim checked against an independent method.

## Definitions

The README defines optimal play as "the strategy the defender can least abuse": the mix whose worst case, over every defence, is best. That is the maximin strategy. For a two-player zero-sum game (the defender's payoff is minus the attacker's, which the matrix convention implies), the minimax theorem says the maximin mix, the minimax mix of the defender, and the Nash equilibrium are the same thing. The definition is correct and standard.

## The examples

| Game | README | Check (`rps_solver.solve_exact`, exact rationals) |
|---|---|---|
| Balanced `[[1,0,-1],[-1,1,0],[0,-1,1]]` | attack [1/3, 1/3, 1/3] | attack [1/3, 1/3, 1/3], defence [1/3, 1/3, 1/3], value 0 |
| Unbalanced `[[1,0,-1],[-1,1,1],[0,-1,1]]` | attack [4/9, 3/9, 2/9], value 1/9 | attack [4/9, 3/9, 2/9], defence [4/9, 2/9, 3/9], value 1/9 |
| High risk `[[1.5,0,-1],[-1.5,1,1],[0,-1,1]]` | not stated | attack [10/23, 8/23, 5/23], value 3/23 |

Both README numbers are right, and the pair (attack, defence) is certified: the attack earns at least 1/9 against every defence, the defence concedes at most 1/9 to every attack. `brute_force.py` agrees to grid precision. An LP solver (scipy `linprog`, HiGHS) agrees on 3000 random matrices up to 5x5, square and not.

## The method: where it is right and where it breaks

`solver.py` finds the attack mix that makes every defence earn the same. This is the indifference principle, and it is the right idea: at the optimum, the defender gains nothing by switching defence.

It is only a full solution when the optimal play uses every option on both sides. When an option should be played with probability zero (dominated, or simply not needed), the equal-payoff equations no longer describe the optimum. Three things can then happen, all seen on random `{-1, 0, 1}` matrices (3000 draws, sizes 2 to 4):

| Outcome of the original method | Count |
|---|---|
| Correct optimum | 959 |
| Linear system singular, no unique solution | 1172 |
| A "probability" comes out negative | 691 |
| Valid-looking probabilities, but not the maximin | 178 |

The last row is the dangerous one. Example: `[[0,0,-1,-1],[-1,-1,-1,-1],[1,0,1,0],[0,1,1,0]]`. Equalising every column gives the pure attack A1 with value -1, which does satisfy the equations, while the true optimum is A2 with value 0. The original code prints the wrong answer without warning.

For rock-paper-scissors-like games (every option beats something and is beaten by something, no option dominated), the equilibrium is usually fully mixed and the method works, which is why the README examples are fine.

## What changed

* `rps_solver.py` (new, standard library only): exact solver by enumeration of square sub-matrices (Shapley and Snow, 1950). Each candidate is certified against the full matrix, so the answer is always an optimal pair, including for non-square matrices and dominated options. Also provides best replies, exploitability and a certification function.
* `solver.py`: same sympy approach, now solving both sides, certifying the pair, and falling back to `rps_solver` with a message when the indifference method does not apply. Matrix can be given on the command line.
* `brute_force.py`: general matrix and grid. The original grid `range(nb_steps)` stopped one step short of probability 1, and the value formula was hard-coded for one game.
* `requirements.txt`: `sympy==1.5.1` and `mpmath==1.1.0` do not install on current Python; unpinned.
* README, "high risk/high reward" example: the matrix shown changes `A1 vs D0` to -1.5, not A0's payoffs. A high-risk A0 would be `[[1.5, 0, -1.5], ...]`. The interactive page uses that version.
* Tests: `tests/test_rps_solver.py` covers the examples, dominated options, the failure case above, non-square games and random games.

## A property worth knowing

In the unbalanced game the frequency of A1 is fixed by one equation, `A0 - A1 = A1 - A2`, which does not involve A1's payoff against D2. Raising that payoff from 0 to 3 leaves A1 at exactly 1/3 and moves the weight from A2 to A0 (`images/buff.png`). Buffing an option changes how often its counter, and its counter's counter, are played; not how often the option itself is played.
