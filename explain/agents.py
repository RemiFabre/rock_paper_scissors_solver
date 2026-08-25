"""Players for repeated matrix games.

Every player sees the game from its own point of view: ``m[i][j]`` is the payoff it
gets when it plays option ``i`` and the opponent plays option ``j``. The attacker
passes the game matrix as is; the defender passes ``-M^T``.

Interface: ``reset()``, ``act() -> int``, ``observe(own, other, payoff)``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rps_solver import solve_exact  # noqa: E402


def own_view(m, attacker: bool):
    """Payoff matrix from the player's own point of view."""
    m = np.asarray(m, float)
    return m if attacker else -m.T


def nash_mix(m):
    return np.array([float(v) for v in solve_exact(m.tolist()).attack])


def exploitability_of(m, x):
    """Game value minus what x guarantees against the worst reply (0 iff x is optimal)."""
    value = float(solve_exact(m.tolist()).value)
    return value - (np.asarray(x) @ m).min()


class Player:
    name = "player"

    def __init__(self, m, rng=None):
        self.m = np.asarray(m, float)
        self.n, self.k = self.m.shape
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        pass

    def act(self) -> int:
        raise NotImplementedError

    def observe(self, own, other, payoff):
        pass

    def sample(self, p):
        p = np.clip(np.asarray(p, float), 0, None)
        p = p / p.sum() if p.sum() > 0 else np.full(self.n, 1 / self.n)
        return int(self.rng.choice(self.n, p=p))

    def best_reply(self, q):
        """Pure option with the highest payoff against the opponent mix q."""
        return int(np.argmax(self.m @ q))


class Fixed(Player):
    """Always plays the same mix (optimal, uniform, biased, ...)."""

    def __init__(self, m, mix, name="fixed", rng=None):
        self.mix = np.asarray(mix, float)
        self.name = name
        super().__init__(m, rng)

    def act(self):
        return self.sample(self.mix)


class Nash(Fixed):
    def __init__(self, m, rng=None):
        super().__init__(m, nash_mix(np.asarray(m, float)), "Nash", rng)


class Cyclic(Player):
    """Deterministic pattern, e.g. 0, 1, 2, 0, 1, 2, ..."""

    name = "Cyclic"

    def __init__(self, m, pattern=None, rng=None):
        self.pattern = list(pattern) if pattern is not None else list(range(np.asarray(m).shape[0]))
        super().__init__(m, rng)

    def reset(self):
        self.t = 0

    def act(self):
        a = self.pattern[self.t % len(self.pattern)]
        self.t += 1
        return a


class Sticky(Player):
    """Human-like: over-plays the strongest option and rarely repeats itself."""

    name = "Sticky"

    def __init__(self, m, strongest=None, repeat_penalty=0.7, rng=None):
        m = np.asarray(m, float)
        self.strongest = int(np.argmax(m.sum(axis=1))) if strongest is None else strongest
        self.repeat_penalty = repeat_penalty
        super().__init__(m, rng)

    def reset(self):
        self.last = None

    def act(self):
        p = np.ones(self.n)
        p[self.strongest] = 2.0
        if self.last is not None:
            p[self.last] *= 1 - self.repeat_penalty
        return self.sample(p)

    def observe(self, own, other, payoff):
        self.last = own


class FictitiousPlay(Player):
    """Best reply to the opponent's empirical frequencies."""

    name = "FictitiousPlay"

    def __init__(self, m, prior=1.0, rng=None):
        self.prior = prior
        super().__init__(m, rng)

    def reset(self):
        self.counts = np.full(self.k, self.prior)

    def act(self):
        return self.best_reply(self.counts / self.counts.sum())

    def observe(self, own, other, payoff):
        self.counts[other] += 1


class RegretMatching(Player):
    """Hart and Mas-Colell (2000): play in proportion to positive cumulative regret."""

    name = "RegretMatching"

    def reset(self):
        self.regret = np.zeros(self.n)

    def act(self):
        pos = np.clip(self.regret, 0, None)
        return self.sample(pos) if pos.sum() > 0 else self.sample(np.ones(self.n))

    def observe(self, own, other, payoff):
        self.regret += self.m[:, other] - self.m[own, other]


class Hedge(Player):
    """Multiplicative weights on cumulative payoff of each option."""

    name = "Hedge"

    def __init__(self, m, eta=0.1, rng=None):
        self.eta = eta
        super().__init__(m, rng)

    def reset(self):
        self.score = np.zeros(self.n)

    def act(self):
        w = np.exp(self.eta * (self.score - self.score.max()))
        return self.sample(w)

    def observe(self, own, other, payoff):
        self.score += self.m[:, other]


class NGram(Player):
    """Predicts the opponent's next move from its last ``order`` moves, best-replies to it.

    Contexts of length order, order-1, ..., 0 are tried in turn; the first one seen
    at least ``min_count`` times is used.
    """

    def __init__(self, m, order=3, min_count=3, rng=None):
        self.order = order
        self.min_count = min_count
        self.name = f"NGram({order})"
        super().__init__(m, rng)

    def reset(self):
        self.history = []
        self.tables = [dict() for _ in range(self.order + 1)]

    def _context(self, length):
        return tuple(self.history[len(self.history) - length:]) if length else ()

    def predict(self):
        for length in range(self.order, -1, -1):
            if len(self.history) < length:
                continue
            counts = self.tables[length].get(self._context(length))
            if counts is not None and counts.sum() >= self.min_count:
                return counts / counts.sum()
        return np.full(self.k, 1 / self.k)

    def act(self):
        return self.best_reply(self.predict())

    def observe(self, own, other, payoff):
        for length in range(self.order + 1):
            if len(self.history) < length:
                continue
            table = self.tables[length]
            ctx = self._context(length)
            if ctx not in table:
                table[ctx] = np.zeros(self.k)
            table[ctx][other] += 1
        self.history.append(other)


class SafeExploiter(Player):
    """Best-replies to a model of the opponent, but never risks more than epsilon.

    Plays (1 - a) * Nash + a * best_reply with the largest a such that the mix's
    exploitability stays below epsilon. epsilon = 0 is exactly Nash.
    """

    def __init__(self, m, epsilon=0.1, prior=2.0, rng=None):
        self.epsilon = epsilon
        self.prior = prior
        self.name = f"Safe({epsilon:g})"
        super().__init__(m, rng)
        self.nash = nash_mix(self.m)
        self.value = float(solve_exact(self.m.tolist()).value)

    def reset(self):
        self.counts = np.full(self.k, self.prior)

    def _exploitability(self, x):
        return self.value - (x @ self.m).min()

    def mix(self):
        q = self.counts / self.counts.sum()
        br = np.zeros(self.n)
        br[self.best_reply(q)] = 1
        if self._exploitability(br) <= self.epsilon:
            return br
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            x = (1 - mid) * self.nash + mid * br
            if self._exploitability(x) <= self.epsilon:
                lo = mid
            else:
                hi = mid
        return (1 - lo) * self.nash + lo * br

    def act(self):
        return self.sample(self.mix())

    def observe(self, own, other, payoff):
        self.counts[other] += 1


class BestReplier(Player):
    """Worst-case adversary: best-replies to the opponent's empirical mix (no prior)."""

    name = "BestReplier"

    def reset(self):
        self.counts = np.zeros(self.k)

    def act(self):
        if self.counts.sum() == 0:
            return self.sample(np.ones(self.n))
        return self.best_reply(self.counts / self.counts.sum())

    def observe(self, own, other, payoff):
        self.counts[other] += 1


def play(m, attacker: Player, defender: Player, rounds: int):
    """Run ``rounds`` rounds. Returns the attacker's payoffs, one per round."""
    m = np.asarray(m, float)
    attacker.reset()
    defender.reset()
    out = np.empty(rounds)
    for t in range(rounds):
        a, d = attacker.act(), defender.act()
        p = m[a, d]
        out[t] = p
        attacker.observe(a, d, p)
        defender.observe(d, a, -p)
    return out
