import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auu.cost_sim import AgentConfig, simulate, tail_metrics, with_guards  # noqa: E402
from auu.decision import CostProfile, ece, policy_cost  # noqa: E402
from auu.queueing import erlang_c_wait  # noqa: E402


def test_chow_threshold():
    p = CostProfile("refund", c_error=1000, c_escalate=15)
    assert abs(p.max_error_prob - 15 / 1015) < 1e-12
    assert abs(p.min_confidence - 1000 / 1015) < 1e-12


def test_policy_cost_counts_regret():
    p = CostProfile("x", c_error=10, c_escalate=1)
    correct = np.array([True, False, True, False])
    act = np.array([True, True, False, False])
    r = policy_cost(correct, act, p)
    assert r["cost_per_decision"] == (10 + 1) / 4


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 200_000)
    correct = rng.random(200_000) < conf
    assert ece(conf, correct) < 0.01


def test_erlang_c_matches_mm1():
    lam, mu = 9.0, 10.0
    assert abs(erlang_c_wait(lam, mu, 1) - (lam / mu) / (mu - lam)) < 1e-12


def test_guards_cut_the_tail():
    cfg = AgentConfig()
    a = tail_metrics(simulate(cfg, 20_000, 0).cost)
    b = tail_metrics(simulate(with_guards(cfg, max_steps=12, loop_window=2), 20_000, 0).cost)
    assert b["mean"] < a["mean"] and b["cvar95"] < a["cvar95"]
