"""Study 1 - The cost of an agent is not its mean.

Simulates 200k queries of a tool-using agent and compares four guard policies.
Outputs: results/01_tail_costs.json and two figures.
"""
import numpy as np

from common import FIG, GOLD, INK, MUTED, PINK, TEAL, BLUE, plt, save_json
from auu.cost_sim import AgentConfig, simulate, tail_metrics, with_guards

N, SEED = 200_000, 7
base = AgentConfig()
policies = {
    "no guard (cap 40 steps)": base,
    "step cap = 8": with_guards(base, max_steps=8),
    "loop detector (2 repeats)": with_guards(base, loop_window=2),
    "cap 12 + loop detector": with_guards(base, max_steps=12, loop_window=2),
}

rows = {}
sims = {}
for name, cfg in policies.items():
    r = simulate(cfg, N, SEED)
    sims[name] = r
    m = tail_metrics(r.cost)
    stuck_share = float(r.cost[r.stuck].sum() / r.cost.sum())
    legit_escalations = float((r.escalated & ~r.stuck).mean())  # good queries cut by the guard
    m.update({
        "stuck_share_of_queries": float(r.stuck.mean()),
        "stuck_share_of_spend": stuck_share,
        "escalation_rate": float(r.escalated.mean()),
        "false_escalation_rate": legit_escalations,
        "mean_steps": float(r.steps.mean()),
    })
    rows[name] = m

# sensitivity: how mean cost reacts to the share of looping queries (Jensen in action)
sweep = []
for p_stuck in np.linspace(0.0, 0.6, 13):
    cfg = AgentConfig(p_stuck_given_hard=float(p_stuck))
    r0 = simulate(cfg, 60_000, SEED)
    r1 = simulate(with_guards(cfg, max_steps=12, loop_window=2), 60_000, SEED)
    sweep.append({
        "stuck_share": float(r0.stuck.mean()),
        "mean_no_guard": float(r0.cost.mean()),
        "mean_guarded": float(r1.cost.mean()),
        "cvar95_no_guard": tail_metrics(r0.cost)["cvar95"],
        "cvar95_guarded": tail_metrics(r1.cost)["cvar95"],
    })

save_json("01_tail_costs.json", {"config": base.__dict__, "n_queries": N, "policies": rows, "sensitivity": sweep})

# ---- figure 1: distribution, mean vs CVaR (no guard)
r = sims["no guard (cap 40 steps)"]
m = rows["no guard (cap 40 steps)"]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
bins = np.logspace(np.log10(r.cost.min()), np.log10(r.cost.max()), 70)
ax.hist(r.cost[~r.stuck], bins=bins, color=TEAL, alpha=0.85, label="normal queries")
ax.hist(r.cost[r.stuck], bins=bins, color=PINK, alpha=0.9, label="queries stuck in a loop")
ax.set_xscale("log")
ax.set_yscale("log")
top = ax.get_ylim()[1]
for x, lab, col, ha, yf in [(m["mean"], "mean", INK, "right", 0.6), (m["p95"], "p95", MUTED, "left", 0.08),
                            (m["cvar95"], "CVaR 95%", GOLD, "left", 0.6)]:
    ax.axvline(x, color=col, lw=1.6, ls="--")
    pad = " " if ha == "left" else ""
    ax.text(x, top * yf, f"{pad}{lab}\n{pad}€{x:.3f} ", color=col, fontsize=8, va="top", ha=ha)
ax.set_xlabel("cost per query (EUR, log scale)")
ax.set_ylabel("number of queries (log)")
ax.set_title(f"{m['stuck_share_of_queries']:.1%} of queries cause {m['stuck_share_of_spend']:.0%} of the spend")
ax.legend(loc="lower left")
fig.savefig(FIG / "01_cost_distribution.png")

# ---- figure 2: guard policies
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
names = list(rows)
short = ["no guard", "cap 8", "loop det.", "cap 12+loop"]
cols = [PINK, BLUE, TEAL, GOLD]
axes[0].bar(short, [rows[n]["mean"] * 1000 for n in names], color=cols)
axes[0].set_title("mean cost per 1,000 queries (EUR)")
axes[1].bar(short, [rows[n]["false_escalation_rate"] * 100 for n in names], color=cols)
axes[1].set_title("good queries cut by the guard (%)")
for a in axes:
    a.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "01_guards.png")

# ---- figure 3: sensitivity
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = [s["stuck_share"] * 100 for s in sweep]
ax.plot(xs, [s["mean_no_guard"] * 1000 for s in sweep], color=PINK, lw=2.2, label="no guard")
ax.plot(xs, [s["mean_guarded"] * 1000 for s in sweep], color=TEAL, lw=2.2, label="cap 12 + loop detector")
ax.set_xlabel("share of queries that get stuck (%)")
ax.set_ylabel("EUR per 1,000 queries")
ax.set_title("Every extra point of looping queries moves the bill")
ax.legend()
fig.savefig(FIG / "01_sensitivity.png")

for n in names:
    print(f"{n:28s} mean €{rows[n]['mean']:.4f}  p99 €{rows[n]['p99']:.3f}  CVaR95/mean {rows[n]['cvar_over_mean']:.1f}x  "
          f"top5% share {rows[n]['top5_share_of_spend']:.0%}  false esc {rows[n]['false_escalation_rate']:.2%}")
