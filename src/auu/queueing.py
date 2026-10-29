"""Queueing tools for the human team that receives escalations.

* ``erlang_c_wait``: exact mean wait of an M/M/c queue (Erlang C).
* ``simulate_day``: discrete-event simulation of a working day with an hourly
  arrival profile (peaks), FIFO service by ``c`` agents, exponential service.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


def erlang_c_wait(lam: float, mu: float, c: int) -> float:
    """Mean waiting time in queue W_q for M/M/c. Returns inf if unstable."""
    rho = lam / (c * mu)
    if rho >= 1:
        return math.inf
    a = lam / mu
    s = sum(a**k / math.factorial(k) for k in range(c))
    top = a**c / (math.factorial(c) * (1 - rho))
    p_wait = top / (s + top)
    return p_wait / (c * mu - lam)


def simulate_day(hourly_rate: np.ndarray, service_mean: float, c: int, seed: int = 0) -> np.ndarray:
    """Return the waiting time (same unit as service_mean, hours) of each arrival.

    ``hourly_rate[h]`` is the Poisson arrival rate during hour h.
    """
    rng = np.random.default_rng(seed)
    arrivals = []
    for h, lam in enumerate(hourly_rate):
        k = rng.poisson(lam)
        arrivals.extend(h + rng.random(k))
    arrivals = np.sort(np.array(arrivals))
    free_at = [0.0] * c
    heapq.heapify(free_at)
    waits = np.empty(len(arrivals))
    for i, t in enumerate(arrivals):
        f = heapq.heappop(free_at)
        start = max(t, f)
        waits[i] = start - t
        heapq.heappush(free_at, start + rng.exponential(service_mean))
    return waits


def staff_for(rate: float, service_mean: float, target_utilisation: float = 0.85) -> int:
    return max(1, math.ceil(rate * service_mean / target_utilisation))
