"""End-to-end case study: the customer-service agent of a (fictional) electricity retailer.

One simulated month, 100,000 customer requests across six intents. Every number
is a stated assumption in INTENTS / ECON below, so the case can be re-run with
any company's figures.

We start from a *typical deployment* and add, one at a time, the practices this
repository argues for:

  S0  typical deployment: no loop guard, "act if confidence >= 90%", staff by the mean
  S1  + loop detector and step cap that escalate instead of retrying
  S2  + autonomy threshold per intent from the cost of each error (Chow's rule)
  S3  + calibrated confidence, learned in a shadow-mode pilot (agent proposes, humans decide);
        the team is still sized on the daily mean, so the extra escalations queue up
  S4  + escalation team staffed on the p90 hourly rate instead of the daily mean
  S5  + investment in a better model/context (fewer errors and loops, more tokens)

For each scenario we report the monthly cost split into tokens, human team,
wrong autonomous actions and complaints caused by waiting, plus operational KPIs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from auu.cost_sim import AgentConfig, simulate, with_guards  # noqa: E402
from auu.decision import fit_binary_calibrator  # noqa: E402
from auu.queueing import simulate_day, staff_for  # noqa: E402

OUT = ROOT / "results" / "case_study"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Intent:
    name: str
    share: float        # share of monthly volume
    c_error: float      # EUR, cost of a wrong autonomous action (refunds paid wrongly, rework, regulatory risk)
    accuracy: float     # mean probability that the agent's proposed action is right
    concentration: float  # Beta concentration of per-request difficulty (lower = more spread)
    p_hard: float       # share of multi-step requests
    p_stuck: float      # share of hard requests that loop


INTENTS = [
    Intent("billing question",      0.40,   2, 0.97, 14, 0.04, 0.25),
    Intent("address / contract",    0.20,  15, 0.95, 12, 0.06, 0.30),
    Intent("outage report",         0.10,   5, 0.95, 12, 0.05, 0.30),
    Intent("meter-reading dispute", 0.15,  40, 0.88,  6, 0.15, 0.40),
    Intent("refund / compensation", 0.10, 120, 0.90,  6, 0.12, 0.35),
    Intent("formal complaint",      0.05,  60, 0.80,  5, 0.25, 0.40),
]

ECON = dict(
    monthly_requests=100_000,
    working_days=22,
    hours_per_day=12,
    service_minutes=6.0,          # human handling time per escalation
    agent_hour_cost=28.0,         # EUR, fully loaded
    c_escalate=6.0,               # EUR, marginal cost of one escalation (staff time at p90 staffing + complaint risk)
    complaint_base=0.02,          # P(complaint) for an escalated customer with no wait
    complaint_per_min=0.012,      # extra P(complaint) per minute waited
    complaint_cost=60.0,          # EUR, handling + churn risk
    pilot_requests=4_000,         # shadow-mode sample with human labels used for calibration
    # improvement project (S5)
    project_capex=150_000.0,
    project_opex_month=4_000.0,
    improve_error_factor=0.6,     # errors x0.6 on the three hard intents
    improve_loop_factor=0.5,      # loops x0.5
    improve_token_factor=1.3,     # +30% tokens per step (richer context)
)
HOURLY_SHAPE = np.array([14, 26, 44, 50, 36, 22, 18, 24, 38, 42, 30, 16], dtype=float)
HOURLY_SHAPE /= HOURLY_SHAPE.mean()
HARD_INTENTS = {"meter-reading dispute", "refund / compensation", "formal complaint"}


def stated_confidence(p_true: np.ndarray, rng) -> np.ndarray:
    """LLM-like verbalised confidence: compressed into 80-99%, weakly related to p_true."""
    s = 0.80 + 0.19 * p_true**0.5 + rng.normal(0, 0.03, p_true.shape)
    return np.clip(np.round(s * 20) / 20, 0.5, 0.99)  # people/LLMs answer in round numbers


def draw_requests(intent: Intent, n: int, rng, improved: bool):
    acc = intent.accuracy
    if improved and intent.name in HARD_INTENTS:
        acc = 1 - (1 - acc) * ECON["improve_error_factor"]
    a, b = acc * intent.concentration, (1 - acc) * intent.concentration
    p_true = rng.beta(a, b, n)
    correct = rng.random(n) < p_true
    return p_true, correct, stated_confidence(p_true, rng)


def calibrators(improved: bool, seed: int) -> dict:
    """Shadow mode: the agent proposes, humans decide, we log both -> isotonic map per intent."""
    rng = np.random.default_rng(seed + 999)
    cal = {}
    for it in INTENTS:
        n = max(200, int(ECON["pilot_requests"] * it.share))
        _, correct, conf = draw_requests(it, n, rng, improved)
        cal[it.name] = fit_binary_calibrator(conf, correct)
    return cal


def run(scenario: str, seed: int = 11) -> dict:
    guards = scenario >= "S1"
    cost_threshold = scenario >= "S2"
    calibrated = scenario >= "S3"
    staff_p90 = scenario >= "S4"
    improved = scenario >= "S5"

    rng = np.random.default_rng(seed)
    cal = calibrators(improved, seed) if calibrated else None
    tok_f = ECON["improve_token_factor"] if improved else 1.0

    totals = dict(tokens=0.0, wrong_actions=0.0, requests=0, autonomous=0, errors=0, escalations=0,
                  loops=0, false_cuts=0)
    per_intent = {}
    for i, it in enumerate(INTENTS):
        n = int(ECON["monthly_requests"] * it.share)
        p_stuck = it.p_stuck * (ECON["improve_loop_factor"] if improved else 1.0)
        cfg = AgentConfig(p_hard=it.p_hard, p_stuck_given_hard=p_stuck,
                          output_tokens_mean=300 * tok_f, tool_tokens_mean=800 * tok_f)
        if guards:
            cfg = with_guards(cfg, max_steps=12, loop_window=2)
        sim = simulate(cfg, n, seed + i)
        p_true, correct, conf = draw_requests(it, n, rng, improved)

        c_err, c_esc = it.c_error, ECON["c_escalate"]
        min_conf = c_err / (c_err + c_esc) if cost_threshold else 0.90
        signal = cal[it.name].predict(conf) if calibrated else conf
        act = (signal >= min_conf) & ~sim.escalated   # stuck / cut requests always end with a human
        wrong = act & ~correct

        totals["tokens"] += sim.cost.sum()
        totals["wrong_actions"] += wrong.sum() * c_err
        totals["requests"] += n
        totals["autonomous"] += act.sum()
        totals["errors"] += wrong.sum()
        totals["escalations"] += (~act).sum()
        totals["loops"] += sim.stuck.sum()
        totals["false_cuts"] += (sim.escalated & ~sim.stuck).sum()
        per_intent[it.name] = {
            "volume": n, "threshold": float(min_conf), "autonomy": float(act.mean()),
            "errors_per_1000": float(1000 * wrong.mean()), "wrong_action_cost": float(wrong.sum() * c_err),
        }

    # ---- human team: escalations arrive with the hourly profile
    svc_h = ECON["service_minutes"] / 60
    esc_per_hour = totals["escalations"] / (ECON["working_days"] * ECON["hours_per_day"])
    profile = HOURLY_SHAPE * esc_per_hour
    rate_for_staff = np.quantile(profile, 0.9) if staff_p90 else profile.mean()
    agents = staff_for(rate_for_staff, svc_h)
    waits = np.concatenate([simulate_day(profile, svc_h, agents, seed=seed * 100 + d)
                            for d in range(ECON["working_days"])]) * 60
    p_complaint = np.clip(ECON["complaint_base"] + ECON["complaint_per_min"] * waits, 0, 1)
    staff_cost = agents * ECON["hours_per_day"] * ECON["working_days"] * ECON["agent_hour_cost"]
    complaints = p_complaint.sum()

    cost = {
        "tokens": float(totals["tokens"]),
        "human team": float(staff_cost),
        "wrong autonomous actions": float(totals["wrong_actions"]),
        "complaints from waiting": float(complaints * ECON["complaint_cost"]),
    }
    if improved:
        cost["improvement project (opex)"] = ECON["project_opex_month"]
    return {
        "scenario": scenario,
        "monthly_cost": cost,
        "total": float(sum(cost.values())),
        "kpis": {
            "autonomy_rate": totals["autonomous"] / totals["requests"],
            "errors_per_1000": 1000 * totals["errors"] / totals["requests"],
            "escalations": int(totals["escalations"]),
            "looping_requests": int(totals["loops"]),
            "good_requests_cut_by_guard": int(totals["false_cuts"]),
            "agents": agents,
            "mean_wait_min": float(waits.mean()),
            "p95_wait_min": float(np.quantile(waits, 0.95)),
            "complaints": float(complaints),
        },
        "per_intent": per_intent,
    }


SCENARIOS = {
    "S0": "Typical deployment",
    "S1": "+ loop detector & step cap",
    "S2": "+ cost-based threshold per intent",
    "S3": "+ calibration from shadow mode",
    "S4": "+ staff on p90 hourly rate",
    "S5": "+ better model / context",
}

if __name__ == "__main__":
    results = {k: run(k) for k in SCENARIOS}
    s4, s5 = results["S4"]["total"], results["S5"]["total"]
    roi = {
        "monthly_saving_vs_S4": s4 - s5,
        "capex": ECON["project_capex"],
        "payback_months": ECON["project_capex"] / (s4 - s5) if s5 < s4 else None,
    }
    (OUT / "results.json").write_text(json.dumps({"econ": ECON, "intents": [it.__dict__ for it in INTENTS],
                                                  "scenarios": results, "improvement_roi": roi}, indent=2))
    base = results["S0"]["total"]
    for k, r in results.items():
        kp = r["kpis"]
        print(f"{k} {SCENARIOS[k]:36s} total €{r['total']:>9,.0f}  ({r['total'] / base - 1:+.0%})  "
              + "  ".join(f"{c}: €{v:,.0f}" for c, v in r["monthly_cost"].items())
              + f"\n    autonomy {kp['autonomy_rate']:.0%}  errors/1000 {kp['errors_per_1000']:.1f}  "
              f"agents {kp['agents']}  wait mean {kp['mean_wait_min']:.1f} p95 {kp['p95_wait_min']:.1f} min  "
              f"complaints {kp['complaints']:.0f}  cut {kp['good_requests_cut_by_guard']}")
    print("ROI", roi)
