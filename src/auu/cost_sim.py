"""Monte Carlo simulator of the token cost of a tool-using LLM agent.

Each query is solved through a loop of steps. At every step the agent re-sends
the full context (system prompt + history), calls the model, and appends the
model output and the tool result to the history. Context therefore grows with
each step and the cost of a query grows super-linearly with its number of steps.

Most queries are easy and finish in one or two steps. A minority are hard, and
some hard queries get *stuck*: the agent keeps calling the same tool with the
same arguments. Stuck queries only end when a guard (step cap or loop detector)
stops them and the query is escalated to a human.

All prices and distributions are explicit assumptions in ``AgentConfig`` so the
results can be re-run under other assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class AgentConfig:
    # workload mix
    p_hard: float = 0.08          # share of hard queries
    p_stuck_given_hard: float = 0.35  # share of hard queries that get stuck in a loop
    # steps needed to solve (when not stuck)
    easy_extra_steps_mean: float = 0.6   # easy: 1 + Poisson(mean)
    hard_extra_steps_mean: float = 3.0   # hard: 3 + Poisson(mean)
    # tokens (lognormal, parameterised by mean and coefficient of variation)
    system_tokens: int = 2000
    output_tokens_mean: float = 300
    tool_tokens_mean: float = 800
    tokens_cv: float = 0.8
    # prices in EUR per million tokens (hypothetical, typical order of magnitude)
    price_input_per_m: float = 3.0
    price_output_per_m: float = 15.0
    # guards
    max_steps: int = 40           # hard cap on steps; 40 ~ "no real guard"
    loop_window: int | None = None  # escalate if the same action repeats this many times in a row


@dataclass
class SimulationResult:
    cost: np.ndarray          # EUR per query
    steps: np.ndarray         # steps executed per query
    escalated: np.ndarray     # bool, query ended in escalation to a human
    stuck: np.ndarray         # bool, query entered a loop
    hard: np.ndarray          # bool, query was hard
    config: AgentConfig = field(repr=False)


def _lognormal(rng: np.random.Generator, mean: float, cv: float, size) -> np.ndarray:
    sigma2 = np.log(1.0 + cv**2)
    mu = np.log(mean) - sigma2 / 2.0
    return rng.lognormal(mu, np.sqrt(sigma2), size)


def simulate(cfg: AgentConfig, n: int = 100_000, seed: int = 0) -> SimulationResult:
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < cfg.p_hard
    stuck = hard & (rng.random(n) < cfg.p_stuck_given_hard)

    needed = np.where(
        hard,
        3 + rng.poisson(cfg.hard_extra_steps_mean, n),
        1 + rng.poisson(cfg.easy_extra_steps_mean, n),
    )
    # a stuck query never finishes on its own
    needed = np.where(stuck, np.iinfo(np.int32).max, needed)

    # when a stuck query starts repeating itself (it does some real work first)
    loop_start = np.where(stuck, 1 + rng.integers(0, 3, n), np.iinfo(np.int32).max)

    limit = np.full(n, cfg.max_steps)
    if cfg.loop_window is not None:
        # the detector fires after `loop_window` identical consecutive actions
        limit = np.minimum(limit, loop_start + cfg.loop_window)

    steps = np.minimum(needed, limit)
    escalated = needed > limit

    max_s = int(steps.max())
    out_tok = _lognormal(rng, cfg.output_tokens_mean, cfg.tokens_cv, (n, max_s))
    tool_tok = _lognormal(rng, cfg.tool_tokens_mean, cfg.tokens_cv, (n, max_s))
    active = np.arange(max_s)[None, :] < steps[:, None]
    out_tok *= active
    tool_tok *= active

    # input at step k = system + sum of outputs and tool results of steps < k
    history = np.cumsum(out_tok + tool_tok, axis=1) - (out_tok + tool_tok)
    in_tok = (cfg.system_tokens + history) * active

    cost = (
        in_tok.sum(1) * cfg.price_input_per_m + out_tok.sum(1) * cfg.price_output_per_m
    ) / 1e6
    return SimulationResult(cost=cost, steps=steps, escalated=escalated, stuck=stuck, hard=hard, config=cfg)


def tail_metrics(cost: np.ndarray, alpha: float = 0.95) -> dict:
    """Mean, quantiles, CVaR_alpha (mean of the worst 1-alpha) and tail share of spend."""
    c = np.sort(cost)
    q = np.quantile(c, alpha)
    tail = c[c >= q]
    top_k = int(np.ceil(len(c) * (1 - alpha)))
    return {
        "mean": float(c.mean()),
        "p50": float(np.quantile(c, 0.50)),
        "p95": float(q),
        "p99": float(np.quantile(c, 0.99)),
        "cvar95": float(tail.mean()),
        "cvar_over_mean": float(tail.mean() / c.mean()),
        "top5_share_of_spend": float(c[-top_k:].sum() / c.sum()),
    }


def with_guards(cfg: AgentConfig, max_steps: int | None = None, loop_window: int | None = None) -> AgentConfig:
    return replace(
        cfg,
        max_steps=cfg.max_steps if max_steps is None else max_steps,
        loop_window=loop_window,
    )
