"""Study 1b - Staffing the escalation team with averages.

(a) Erlang C: mean wait vs utilisation for 1, 5 and 20 agents.
(b) Discrete-event simulation of 200 working days with a peaked hourly profile,
    staffing by the daily mean rate vs by the 90th percentile hourly rate.
"""
import numpy as np

from common import FIG, GOLD, MUTED, PINK, TEAL, plt, save_json
from auu.queueing import erlang_c_wait, simulate_day, staff_for

SERVICE_H = 0.1  # 6 minutes per escalation
# arrivals per hour, 08:00-20:00, two peaks (mean 30/h)
profile = np.array([14, 26, 44, 50, 36, 22, 18, 24, 38, 42, 30, 16], dtype=float)
profile *= 30 / profile.mean()

# (a) analytic curve
util = np.linspace(0.5, 0.97, 60)
curves = {}
for c in (1, 5, 20):
    mu = 1 / SERVICE_H
    curves[c] = [erlang_c_wait(u * c * mu, mu, c) / SERVICE_H for u in util]

# (b) simulation
rules = {
    "staff by daily mean": staff_for(profile.mean(), SERVICE_H),
    "staff by p90 hourly rate": staff_for(np.quantile(profile, 0.9), SERVICE_H),
}
sim = {}
for name, c in rules.items():
    w = np.concatenate([simulate_day(profile, SERVICE_H, c, seed=s) for s in range(200)]) * 60
    sim[name] = {
        "agents": c,
        "mean_wait_min": float(w.mean()),
        "p95_wait_min": float(np.quantile(w, 0.95)),
        "share_wait_over_15min": float((w > 15).mean()),
    }
# by-hour mean wait for the figure
def hourly_wait(c):
    waits = [[] for _ in range(12)]
    for s in range(200):
        rng = np.random.default_rng(s)
        arr = []
        for h, lam in enumerate(profile):
            arr.extend(h + rng.random(rng.poisson(lam)))
        arr = np.sort(arr)
        free = np.zeros(c)
        for t in arr:
            i = free.argmin()
            start = max(t, free[i])
            waits[int(t)].append((start - t) * 60)
            free[i] = start + rng.exponential(SERVICE_H)
    return [float(np.mean(x)) for x in waits]

hourly = {name: hourly_wait(c) for name, c in rules.items()}

save_json("02_escalation_queue.json", {
    "service_minutes": SERVICE_H * 60, "hourly_profile": profile.round(2).tolist(),
    "erlang_c_wait_over_service": {
        str(c): {f"{u:.2f}": v for u, v in zip(util, curves[c]) if abs(u - 0.9) < 0.005 or abs(u - 0.95) < 0.005}
        for c in curves
    },
    "mm1_wait_over_service": {"0.90": 0.9 / 0.1, "0.95": 0.95 / 0.05},
    "simulation": sim, "hourly_mean_wait_min": hourly,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for c, col in zip(curves, (PINK, GOLD, TEAL)):
    axes[0].plot(util * 100, curves[c], color=col, lw=2.2, label=f"{c} agent{'s' if c > 1 else ''}")
axes[0].set_yscale("log")
axes[0].set_xlabel("utilisation of the team (%)")
axes[0].set_ylabel("mean wait / service time")
axes[0].set_title("Erlang C: waits explode near full utilisation")
axes[0].legend()
hours = [f"{8 + h}h" for h in range(12)]
ax2 = axes[1]
ax2.bar(hours, profile, color="#e6e2da", label="escalations per hour")
ax2.set_ylabel("escalations / hour", color=MUTED)
ax3 = ax2.twinx()
for (name, w), col in zip(hourly.items(), (PINK, TEAL)):
    ax3.plot(hours, w, color=col, lw=2.2, marker="o", ms=3, label=f"{name} ({rules[name]} agents)")
ax3.set_ylabel("mean wait (min)")
ax3.grid(False)
ax3.spines["right"].set_visible(True)
ax2.set_title("Same daily volume, different staffing rule")
ax3.legend(loc="upper right", fontsize=8)
ax2.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "02_escalation_queue.png")

for n, v in sim.items():
    print(n, v)
