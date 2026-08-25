import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "explain"))
from agents import Cyclic, FictitiousPlay, Fixed, NGram, Nash, SafeExploiter, exploitability_of, own_view, play  # noqa: E402
from common import INTUITIVE, OPTIMAL, UNBALANCED  # noqa: E402

M = np.asarray(UNBALANCED, float)


def test_fixed_plays_its_mix():
    p = Fixed(M, OPTIMAL, rng=np.random.default_rng(0))
    acts = np.array([p.act() for _ in range(20000)])
    freq = np.bincount(acts, minlength=3) / len(acts)
    assert np.allclose(freq, OPTIMAL, atol=0.02)


def test_fictitious_play_best_replies_to_biased_opponent():
    rng = np.random.default_rng(1)
    attacker = Fixed(M, INTUITIVE, rng=rng)  # over-plays A1
    defender = FictitiousPlay(own_view(M, attacker=False), rng=rng)
    play(M, attacker, defender, 300)
    assert defender.act() == 0  # D0 beats A1


def test_safe_exploiter_zero_is_nash():
    s = SafeExploiter(M, epsilon=0.0, rng=np.random.default_rng(2))
    s.counts = np.array([100.0, 1.0, 1.0])  # a very biased opponent model
    assert exploitability_of(M, s.mix()) < 1e-9
    assert np.allclose(s.mix(), OPTIMAL)


def test_safe_exploiter_respects_budget():
    s = SafeExploiter(M, epsilon=0.1, rng=np.random.default_rng(3))
    s.counts = np.array([1.0, 1.0, 100.0])
    assert exploitability_of(M, s.mix()) <= 0.1 + 1e-9
    assert exploitability_of(M, s.mix()) > 0.09


def test_ngram_detects_cycle():
    rng = np.random.default_rng(4)
    attacker = Cyclic(M, [0, 1, 2, 2])
    defender = NGram(own_view(M, attacker=False), order=3, rng=rng)
    payoffs = play(M, attacker, defender, 200)
    assert payoffs[100:].mean() < -0.9  # the defender counters nearly every move


def test_nash_takes_the_value_from_everyone():
    rng = np.random.default_rng(5)
    for mix in (INTUITIVE, [1, 0, 0], [0.2, 0.2, 0.6]):
        p = play(M, Nash(M, rng=rng), Fixed(own_view(M, False), mix, rng=rng), 20000)
        assert abs(p.mean() - 1 / 9) < 0.03
