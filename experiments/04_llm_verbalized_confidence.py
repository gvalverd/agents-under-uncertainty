"""Study 3 (optional, needs an API key) - Is an LLM's stated confidence usable as an autonomy signal?

Generates questions with exactly known answers (no dataset download, no
contamination), asks the model for an answer plus a confidence 0-100, and
records correctness, confidence and token usage.

    # collect (Anthropic)
    export ANTHROPIC_API_KEY=...
    python 04_llm_verbalized_confidence.py collect --provider anthropic --model claude-haiku-4-5-20251001 --n 300

    # collect (any OpenAI-compatible endpoint)
    export OPENAI_API_KEY=... ; export OPENAI_BASE_URL=https://api.openai.com/v1
    python 04_llm_verbalized_confidence.py collect --provider openai --model gpt-4o-mini --n 300

    # analyse whatever is in results/llm_runs/*.jsonl
    python 04_llm_verbalized_confidence.py analyze

Reference: Xiong et al. (2024), "Can LLMs Express Their Uncertainty? An Empirical
Evaluation of Confidence Elicitation in LLMs", ICLR 2024 - verbalised confidence
tends to be over-confident and clustered at 80-100%.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import urllib.request
from pathlib import Path

import numpy as np

from common import FIG, MUTED, PINK, TEAL, plt, save_json, RES
from auu.decision import CostProfile, ece, fit_binary_calibrator, policy_cost, reliability

RUNS = RES / "llm_runs"
PROMPT = (
    "Answer the question. Then estimate the probability (0-100) that your answer is exactly correct.\n"
    'Reply ONLY with JSON: {{"answer": "<answer>", "confidence": <0-100>}}\n\nQuestion: {q}'
)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = ["strawberry", "mississippi", "bookkeeper", "onomatopoeia", "parallelogram", "assessment",
         "committee", "possession", "indivisibility", "abracadabra", "unbelievable", "sleeplessness"]


# --------------------------------------------------------------- questions
def make_question(rng: random.Random) -> dict:
    kind = rng.choice(["mul", "date", "letters", "sum"])
    if kind == "mul":
        d = rng.choice([2, 3, 4, 5])
        a, b = rng.randint(10 ** (d - 1), 10**d - 1), rng.randint(10 ** (d - 1), 10**d - 1)
        return {"kind": f"mul{d}", "q": f"What is {a} * {b}? Give only the integer.", "gold": str(a * b)}
    if kind == "sum":
        xs = [rng.randint(100, 9999) for _ in range(rng.choice([3, 6, 10]))]
        return {"kind": f"sum{len(xs)}", "q": f"What is {' + '.join(map(str, xs))}? Give only the integer.",
                "gold": str(sum(xs))}
    if kind == "date":
        start = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 364))
        n = rng.choice([3, 17, 45, 200, 1000])
        target = start + dt.timedelta(days=n)
        return {"kind": f"date{n}", "q": f"{start.isoformat()} is a {DAYS[start.weekday()]}. "
                f"What day of the week is it {n} days later? Give only the day name in English.",
                "gold": DAYS[target.weekday()]}
    w = rng.choice(WORDS)
    ch = rng.choice(sorted(set(w)))
    return {"kind": "letters", "q": f"How many times does the letter '{ch}' appear in the word '{w}'? "
            "Give only the integer.", "gold": str(w.count(ch))}


def normalise(s: str) -> str:
    return re.sub(r"[\s,\.]", "", str(s)).lower()


# --------------------------------------------------------------- providers
def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def call(provider: str, model: str, prompt: str) -> tuple[str, int, int]:
    if provider == "anthropic":
        out = _post("https://api.anthropic.com/v1/messages",
                    {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
                    {"model": model, "max_tokens": 200, "messages": [{"role": "user", "content": prompt}]})
        return out["content"][0]["text"], out["usage"]["input_tokens"], out["usage"]["output_tokens"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    out = _post(f"{base}/chat/completions",
                {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
                {"model": model, "max_tokens": 200, "messages": [{"role": "user", "content": prompt}]})
    u = out.get("usage", {})
    return out["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def collect(args) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{args.provider}__{args.model.replace('/', '_')}.jsonl"
    rng = random.Random(args.seed)
    with path.open("a") as f:
        for i in range(args.n):
            q = make_question(rng)
            try:
                text, tin, tout = call(args.provider, args.model, PROMPT.format(q=q["q"]))
                m = re.search(r"\{.*\}", text, re.S)
                parsed = json.loads(m.group(0)) if m else {}
                ans, conf = parsed.get("answer", ""), float(parsed.get("confidence", np.nan))
            except Exception as e:  # keep going, record the failure
                text, tin, tout, ans, conf = f"ERROR {e}", 0, 0, "", np.nan
            row = {**q, "answer": ans, "confidence": conf / 100 if conf == conf else None,
                   "correct": normalise(ans) == normalise(q["gold"]), "input_tokens": tin,
                   "output_tokens": tout, "raw": text}
            f.write(json.dumps(row) + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{args.n}")
    print("saved", path)


# --------------------------------------------------------------- analysis
def analyze(_args) -> None:
    files = sorted(RUNS.glob("*.jsonl"))
    if not files:
        print("No runs found in results/llm_runs. Run `collect` first.")
        return
    profiles = [CostProfile("refund 1000", 1000, 15), CostProfile("address 100", 100, 15), CostProfile("e-mail 2", 2, 15)]
    report = {}
    for path in files:
        rows = [json.loads(l) for l in path.open() if l.strip()]
        rows = [r for r in rows if r["confidence"] is not None]
        conf = np.array([min(max(r["confidence"], 0), 1) for r in rows])
        corr = np.array([r["correct"] for r in rows])
        # 2-fold cross-fitted isotonic calibration so we never calibrate on the evaluation rows
        idx = np.random.default_rng(0).permutation(len(rows))
        half = len(rows) // 2
        cal = np.empty_like(conf)
        for a, b in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
            cal[b] = fit_binary_calibrator(conf[a], corr[a]).predict(conf[b])
        res = {"n": len(rows), "accuracy": float(corr.mean()), "mean_confidence": float(conf.mean()),
               "ece_raw": ece(conf, corr), "ece_calibrated": ece(cal, corr), "profiles": {}}
        for p in profiles:
            res["profiles"][p.name] = {
                "fixed 90% (stated)": policy_cost(corr, conf >= 0.9, p),
                "cost threshold (stated)": policy_cost(corr, conf >= p.min_confidence, p),
                "cost threshold (calibrated)": policy_cost(corr, cal >= p.min_confidence, p),
                "always escalate": policy_cost(corr, np.zeros_like(corr), p),
            }
        tok = np.array([r["input_tokens"] + r["output_tokens"] for r in rows])
        res["tokens"] = {"mean": float(tok.mean()), "p95": float(np.quantile(tok, 0.95)), "max": int(tok.max())}
        report[path.stem] = res

        fig, ax = plt.subplots(figsize=(4.6, 4))
        c, a, n = reliability(conf, corr, 10)
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED)
        ax.scatter(c, a, s=np.sqrt(n) * 12, color=PINK, label="stated")
        c2, a2, n2 = reliability(cal, corr, 10)
        ax.scatter(c2, a2, s=np.sqrt(n2) * 12, color=TEAL, label="calibrated (cross-fitted)")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{path.stem}\nECE {res['ece_raw']:.2f} -> {res['ece_calibrated']:.2f}", fontsize=9)
        ax.legend(fontsize=8)
        fig.savefig(FIG / f"04_reliability__{path.stem}.png")
        print(json.dumps({k: v for k, v in res.items() if k != "profiles"}, indent=1))
    save_json("04_llm_verbalized_confidence.json", report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    sub.add_parser("analyze")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else analyze(a)
