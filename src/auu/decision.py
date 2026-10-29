"""Uncertainty-aware autonomy: when should an agent act on its own?

The agent proposes an action with confidence p (its probability of being right).
It can either *act* or *escalate* to a human. Costs are expressed as regret:

* ``c_error``    : cost of acting autonomously and being wrong.
* ``c_escalate`` : cost of escalating a case the agent would have got right
                   (wasted human time, delay).

Expected regret: act -> (1 - p) * c_error ; escalate -> p * c_escalate.
Acting is optimal iff  P(error) = 1 - p  <  c_escalate / (c_escalate + c_error).

This is Chow's (1970) reject rule with asymmetric costs, the same structure as
the newsvendor critical ratio. It is only optimal if p is *calibrated*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


@dataclass(frozen=True)
class CostProfile:
    name: str
    c_error: float
    c_escalate: float

    @property
    def max_error_prob(self) -> float:
        return self.c_escalate / (self.c_escalate + self.c_error)

    @property
    def min_confidence(self) -> float:
        return 1.0 - self.max_error_prob


def policy_cost(correct: np.ndarray, act: np.ndarray, profile: CostProfile) -> dict:
    """Realised average regret per decision and operational rates."""
    wrong_acts = act & ~correct
    wasted_escalations = ~act & correct
    cost = wrong_acts * profile.c_error + wasted_escalations * profile.c_escalate
    return {
        "cost_per_decision": float(cost.mean()),
        "autonomy_rate": float(act.mean()),
        "error_rate_when_acting": float(wrong_acts.sum() / max(act.sum(), 1)),
        "errors_per_1000": float(1000 * wrong_acts.mean()),
    }


# ----------------------------------------------------------------- calibration
def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width bins (Guo et al., 2017)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidence, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def reliability(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidence, bins[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf.append(confidence[m].mean())
            acc.append(correct[m].mean())
            cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: one scalar T minimising NLL on a held-out set."""
    def nll(log_t):
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return -lp[np.arange(len(y)), y].mean()

    res = minimize_scalar(nll, bounds=(-3, 5), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    return softmax(logits / t, axis=1)


def fit_binary_calibrator(confidence: np.ndarray, correct: np.ndarray):
    """Isotonic map from raw confidence to P(correct); used for verbalised LLM confidence."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(confidence, correct.astype(float))
    return iso
