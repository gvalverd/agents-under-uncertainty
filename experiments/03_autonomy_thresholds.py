"""Study 2 - When should an agent act on its own?

Recreates two classic results on real data and puts them together:
  * Chow (1970): optimal reject (escalate) rule under asymmetric costs.
  * Guo et al. (2017): modern classifiers are over-confident; temperature
    scaling fixes most of it (ECE before/after).

The "agent" is a classifier on scikit-learn's handwritten digits dataset
(1,797 real images, offline). Gaussian Naive Bayes is used on purpose: like
LLM self-reported confidence, it is strongly over-confident. A small MLP is
reported as a second, better-calibrated model.

Each prediction is either executed (act) or escalated to a human. We compare:
  always act | fixed 90% threshold | cost threshold on raw confidence |
  cost threshold on calibrated confidence.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import FIG, GOLD, INK, MUTED, PINK, TEAL, BLUE, plt, save_json
from auu.decision import (CostProfile, apply_temperature, ece, fit_binary_calibrator, fit_temperature,
                          policy_cost, reliability)

PROFILES = [
    CostProfile("approve a 1,000 EUR refund", c_error=1000, c_escalate=15),
    CostProfile("update a customer address", c_error=100, c_escalate=15),
    CostProfile("route an e-mail", c_error=2, c_escalate=15),
]
SEEDS = range(20)


def run_seed(seed, model_name):
    X, y = load_digits(return_X_y=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.6, random_state=seed, stratify=y_tmp)
    if model_name == "gaussian_nb":
        model = GaussianNB().fit(X_tr, y_tr)
        logit = lambda Z: model.predict_joint_log_proba(Z)  # noqa: E731
    else:
        sc = StandardScaler().fit(X_tr)
        model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-5, max_iter=2000, random_state=seed)
        model.fit(sc.transform(X_tr), y_tr)

        def logit(Z):
            return np.log(np.clip(model.predict_proba(sc.transform(Z)), 1e-12, 1))

    # Guo et al. (2017): one temperature fitted on a held-out calibration split
    t = fit_temperature(logit(X_cal), y_cal)
    raw = apply_temperature(logit(X_te), 1.0)
    conf_ts = apply_temperature(logit(X_te), t).max(1)
    # top-label isotonic calibration on the same split (robust when one temperature is not enough)
    raw_cal = apply_temperature(logit(X_cal), 1.0)
    iso = fit_binary_calibrator(raw_cal.max(1), raw_cal.argmax(1) == y_cal)
    pred = raw.argmax(1)
    correct = pred == y_te
    conf_raw = raw.max(1)
    conf_cal = iso.predict(conf_raw)

    out = {
        "accuracy": float(correct.mean()), "temperature": t,
        "ece_raw": ece(conf_raw, correct), "ece_temperature": ece(conf_ts, correct),
        "ece_cal": ece(conf_cal, correct),
        "mean_conf_raw": float(conf_raw.mean()), "mean_conf_cal": float(conf_cal.mean()),
        "profiles": {},
    }
    for p in PROFILES:
        pol = {
            "always act": np.ones_like(correct, dtype=bool),
            "always escalate": np.zeros_like(correct, dtype=bool),
            "fixed 90% (raw)": conf_raw >= 0.90,
            "cost threshold (raw)": conf_raw >= p.min_confidence,
            "cost threshold (calibrated)": conf_cal >= p.min_confidence,
        }
        out["profiles"][p.name] = {k: policy_cost(correct, a, p) for k, a in pol.items()}
    return out, (conf_raw, conf_cal, correct)


def aggregate(runs):
    agg = {k: float(np.mean([r[k] for r in runs])) for k in
           ("accuracy", "temperature", "ece_raw", "ece_temperature", "ece_cal", "mean_conf_raw", "mean_conf_cal")}
    agg["profiles"] = {}
    for p in PROFILES:
        agg["profiles"][p.name] = {"min_confidence": p.min_confidence}
        for pol in runs[0]["profiles"][p.name]:
            agg["profiles"][p.name][pol] = {
                m: float(np.mean([r["profiles"][p.name][pol][m] for r in runs]))
                for m in runs[0]["profiles"][p.name][pol]
            }
    return agg


results, example = {}, {}
for model_name in ("gaussian_nb", "mlp"):
    runs = []
    for s in SEEDS:
        r, ex = run_seed(s, model_name)
        runs.append(r)
        if s == 0:
            example[model_name] = ex
    results[model_name] = aggregate(runs)
save_json("03_autonomy_thresholds.json", {"seeds": len(SEEDS), "models": results})

# ---- figure: reliability diagram (Guo et al. style), Naive Bayes
conf_raw, conf_cal, correct = example["gaussian_nb"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, conf, title, col in [(axes[0], conf_raw, "raw confidence", PINK), (axes[1], conf_cal, "after isotonic calibration", TEAL)]:
    c, a, n = reliability(conf, correct, 10)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.bar(c, a, width=0.08, color=col, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title(f"{title}  (ECE {ece(conf, correct):.2f})")
fig.suptitle("Gaussian Naive Bayes on handwritten digits - an over-confident agent", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "03_reliability.png")

# ---- figure: cost vs threshold for the refund profile
p = PROFILES[0]
ths = np.linspace(0.5, 0.99999, 400)
fig, ax = plt.subplots(figsize=(7, 3.6))
for conf, lab, col in [(conf_raw, "raw confidence", PINK), (conf_cal, "calibrated confidence", TEAL)]:
    ax.plot(ths * 100, [policy_cost(correct, conf >= t, p)["cost_per_decision"] for t in ths], color=col, lw=2, label=lab)
ax.axvline(90, color=MUTED, ls="--")
ax.text(90, ax.get_ylim()[1] * 0.95, " fixed 90%", color=MUTED, fontsize=8, va="top")
ax.axvline(p.min_confidence * 100, color=GOLD, ls="--")
ax.text(p.min_confidence * 100, ax.get_ylim()[1] * 0.5, f"Chow threshold \n{p.min_confidence:.1%} ", color=GOLD,
        fontsize=8, va="top", ha="right")
ax.set_xlabel("confidence threshold to act alone (%)")
ax.set_ylabel("expected regret per decision (EUR)")
ax.set_title(f"'{p.name}': error {p.c_error:.0f} EUR vs escalation {p.c_escalate:.0f} EUR")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.80))
fig.savefig(FIG / "03_cost_vs_threshold.png")

# ---- figure: policy comparison across profiles (NB)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
pols = ["always act", "fixed 90% (raw)", "cost threshold (raw)", "cost threshold (calibrated)"]
cols = [PINK, MUTED, BLUE, TEAL]
for ax, prof in zip(axes, PROFILES):
    vals = [results["gaussian_nb"]["profiles"][prof.name][q]["cost_per_decision"] for q in pols]
    ax.bar(["act", "90%", "cost raw", "cost cal."], vals, color=cols)
    ax.set_title(prof.name, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel("EUR regret per decision")
fig.tight_layout()
fig.savefig(FIG / "03_policies.png")

for m, r in results.items():
    print(f"\n{m}: acc {r['accuracy']:.3f}  ECE raw {r['ece_raw']:.3f}  temp {r['ece_temperature']:.3f} (T={r['temperature']:.2f})  isotonic {r['ece_cal']:.3f}  mean conf raw {r['mean_conf_raw']:.3f}")
    for prof, d in r["profiles"].items():
        print(f"  {prof} (act if conf >= {d['min_confidence']:.3f})")
        for pol, v in d.items():
            if pol == "min_confidence":
                continue
            print(f"    {pol:30s} €{v['cost_per_decision']:7.2f}/decision  autonomy {v['autonomy_rate']:.0%}  "
                  f"errors/1000 {v['errors_per_1000']:.1f}")
