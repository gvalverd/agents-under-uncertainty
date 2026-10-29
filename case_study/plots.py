import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import FIG, GOLD, INK, MUTED, PINK, TEAL, BLUE, plt  # noqa: E402

d = json.loads((ROOT / "results" / "case_study" / "results.json").read_text())
sc = d["scenarios"]
keys = list(sc)
labels = ["S0\ntypical", "S1\n+ loop\nguard", "S2\n+ cost\nthresholds", "S3\n+ calibration\n(team on mean)",
          "S4\n+ staff on\np90", "S5\n+ better\nmodel"]
comps = ["wrong autonomous actions", "complaints from waiting", "human team", "tokens", "improvement project (opex)"]
cols = [PINK, GOLD, BLUE, TEAL, MUTED]

fig, ax = plt.subplots(figsize=(9.5, 4.4))
bottom = [0.0] * len(keys)
for comp, col in zip(comps, cols):
    vals = [sc[k]["monthly_cost"].get(comp, 0) / 1000 for k in keys]
    ax.bar(labels, vals, bottom=bottom, color=col, label=comp, width=0.62)
    bottom = [b + v for b, v in zip(bottom, vals)]
for i, k in enumerate(keys):
    ax.text(i, bottom[i] + 4, f"€{sc[k]['total'] / 1000:.0f}k", ha="center", fontsize=9, color=INK, fontweight="bold")
ax.set_ylabel("monthly cost (thousand EUR)")
ax.set_title("Electricity retailer, 100k requests/month: where the money goes")
ax.legend(fontsize=8, loc="upper right")
ax.tick_params(axis="x", labelsize=8)
ax.set_ylim(0, max(bottom) * 1.15)
fig.savefig(FIG / "05_case_cost_waterfall.png")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
short = [l.split("\n")[0] for l in labels]
for ax, (kpi, title, fmt) in zip(axes, [("autonomy_rate", "autonomy (%)", 100), ("errors_per_1000", "wrong actions per 1,000", 1),
                                        ("p95_wait_min", "p95 wait for a human (min)", 1)]):
    ax.bar(short, [sc[k]["kpis"][kpi] * fmt for k in keys], color=[PINK, BLUE, BLUE, GOLD, TEAL, TEAL])
    ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "05_case_kpis.png")

intents = list(sc["S0"]["per_intent"])
fig, ax = plt.subplots(figsize=(8.5, 3.6))
w = 0.27
for j, (k, col) in enumerate([("S0", PINK), ("S4", TEAL), ("S5", GOLD)]):
    ax.bar([i + (j - 1) * w for i in range(len(intents))], [sc[k]["per_intent"][n]["autonomy"] * 100 for n in intents],
           width=w, color=col, label={"S0": "S0 typical (90% fixed)", "S4": "S4 full method", "S5": "S5 + better model"}[k])
ax.set_xticks(range(len(intents)), [n.replace(" / ", "/\n").replace("-reading", "\nreading") for n in intents], fontsize=8)
ax.set_ylabel("share handled alone (%)")
ax.set_title("Autonomy per intent: act where errors are cheap, escalate where they are expensive")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
fig.savefig(FIG / "05_case_autonomy_per_intent.png")
print("ok")
