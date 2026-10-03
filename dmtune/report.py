"""Before/after report: laya.evals metrics + extras + robustness + cost, for every runner on one test split."""
import copy
import gc
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from laya.evals import Dataset, Example, evaluate
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score)

from .data import expected_of, load_jsonl, option_keys, preprocess
from .model import load_backend
from .train import Memory


def dataset(rows):
    return Dataset([Example.from_dict({
        "state": r["state"], "questions": r["questions"], "tags": r.get("tags", []),
        "expected": {q: expected_of(r, q) for q in r["questions"]
                     if q in r.get("expected", {}) or q in r.get("target", {})}}) for r in rows])


def p_correct(answer, expected, qdef):
    if qdef["type"] == "noul":
        return answer["noul"] if expected else 1 - answer["noul"]
    return answer.get("probabilities", {}).get(str(expected), 0.0)


def coverage_at_risk(conf, correct, risk=0.05):
    """Largest share of decisions you can automate (most confident first) at <= `risk` error."""
    order = np.argsort(-conf)
    err = np.cumsum(1 - correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.nonzero(err <= risk)[0]
    return float((ok.max() + 1) / len(conf)) if len(ok) else 0.0


def extras(cases, qdefs):
    judged = [c for c in cases if c["correct"] is not None]  # laya leaves score answers to MAE/QWK
    conf = np.array([c["confidence"] for c in judged], float)
    corr = np.array([c["correct"] for c in judged], float)
    out = {"accuracy": float(corr.mean()) if judged else float("nan"),
           "nll": float(np.mean([-np.log(max(p_correct(c["answer"], c["expected"], qdefs[c["qid"]]), 1e-12))
                                 for c in cases])),
           "coverage@5%risk": coverage_at_risk(conf, corr) if judged else float("nan")}
    for qid in sorted({c["qid"] for c in cases}):
        cs, t = [c for c in cases if c["qid"] == qid], qdefs[qid]["type"]
        y = [c["expected"] for c in cs]
        if t == "choice":
            pred = [c["answer"]["choice"] for c in cs]
            out[f"{qid}/macro_f1"] = float(f1_score(y, pred, average="macro"))
            out[f"{qid}/balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
        elif t == "noul" and len({bool(v) for v in y}) == 2:
            s = [c["answer"]["noul"] for c in cs]
            out[f"{qid}/auroc"] = float(roc_auc_score([bool(v) for v in y], s))
            out[f"{qid}/auprc"] = float(average_precision_score([bool(v) for v in y], s))
        elif t == "score":
            out[f"{qid}/qwk"] = float(cohen_kappa_score(y, [round(c["answer"]["score"]) for c in cs], weights="quadratic"))
            k = len(qdefs[qid]["criteria"])
            rps = []
            for c in cs:
                p = np.array([c["answer"]["probabilities"].get(str(i), 0.0) for i in range(k)])
                rps.append(((np.cumsum(p) - (np.arange(k) >= int(c["expected"]))) ** 2).sum() / (k - 1))
            out[f"{qid}/rps"] = float(np.mean(rps))
    return out


def top(a):
    if "choice" in a:
        return a["choice"]
    return a["noul"] >= 0.5 if "noul" in a else round(a["score"])


def robustness(runner, rows, gray):
    """flip rate: choice answers that change when the option order is reversed (Jev's known weak spot).
    media reliance: answers that change when the image is replaced by a flat gray one (low = ignores media)."""
    flips, n_flip, media, n_media = 0, 0, 0, 0
    for r in rows:
        base = runner.predict(r["state"], r["questions"])["answers"]
        rev = {q: {**d, "criteria": dict(reversed(list(d["criteria"].items())))}
               for q, d in r["questions"].items() if d["type"] == "choice"}
        if rev:
            ans = runner.predict(r["state"], rev)["answers"]
            flips += sum(top(ans[q]) != top(base[q]) for q in rev)
            n_flip += len(rev)
        if isinstance(r["state"], dict) and "image" in r["state"]:
            ans = runner.predict({**r["state"], "image": gray}, r["questions"])["answers"]
            media += sum(top(ans[q]) != top(base[q]) for q in base)
            n_media += len(base)
    out = {}
    if n_flip:
        out["permutation_flip_rate"] = flips / n_flip
    if n_media:
        out["media_reliance"] = media / n_media
    return out


def score_runner(name, runner, rows, qdefs, gray):
    mem = Memory(getattr(runner, "device", torch.device("cpu")))
    rep = evaluate(runner, dataset(rows))
    res = {"overall": rep.overall, "extras": extras(rep.cases, qdefs),
           "robustness": robustness(runner, rows, gray), "slices": rep.slices,
           "calib": [(c["confidence"], c["correct"]) for c in rep.cases if c["correct"] is not None]}
    res["cost"] = {"latency_p50_ms": rep.overall.get("latency_p50_ms"),
                   "latency_p95_ms": rep.overall.get("latency_p95_ms"), "peak_memory_gb": mem.sample()}
    print(f"  {name}: accuracy={res['extras']['accuracy']:.3f} ece={rep.overall.get('ece', float('nan')):.3f}")
    return res


ROWS = [("accuracy", "extras"), ("choice_accuracy", "overall"), ("noul_accuracy", "overall"), ("score_mae", "overall"),
        ("ece", "overall"), ("brier", "overall"), ("nll", "extras"), ("aurc", "overall"),
        ("selective_accuracy@50", "overall"), ("selective_accuracy@80", "overall"), ("coverage@5%risk", "extras"),
        ("permutation_flip_rate", "robustness"), ("media_reliance", "robustness"),
        ("latency_p50_ms", "cost"), ("latency_p95_ms", "cost"), ("peak_memory_gb", "cost")]
LOWER_IS_BETTER = {"score_mae", "ece", "brier", "nll", "aurc", "rps", "permutation_flip_rate", "latency_p50_ms",
                   "latency_p95_ms", "peak_memory_gb"}


def markdown(results, majority, train):
    names = list(results)
    per_q = sorted({k for r in results.values() for k in r["extras"] if "/" in k})
    rows = ROWS + [(k, "extras") for k in per_q]
    head = "| metric | " + " | ".join(names) + (" | Δ fine-tuned − base |" if "base" in results and "fine-tuned" in results else " |")
    lines = ["# dmtune report", "", f"Majority-class accuracy on this test split: **{majority:.3f}**", "",
             head, "|" + "---|" * (len(names) + 1 + ("Δ" in head))]
    for metric, group in rows:
        vals = [results[n][group].get(metric) for n in names]
        if all(v is None for v in vals):
            continue
        cells = ["–" if v is None else f"{v:.3f}" for v in vals]
        lower = metric.split("/")[-1] in LOWER_IS_BETTER  # per-question keys look like "urgency/rps"
        if "Δ" in head:
            b, f = results["base"][group].get(metric), results["fine-tuned"][group].get(metric)
            d = "–" if b is None or f is None else f"{f - b:+.3f}" + (" ✓" if (f < b) == lower and f != b else "")
            cells.append(d)
        arrow = " (↓)" if lower else ""
        lines.append(f"| {metric}{arrow} | " + " | ".join(cells) + " |")
    if train:
        lines += ["", "## Training", "", f"- device: {train['device']}, trainable params: {train['trainable_params_m']}M",
                  f"- wall time: {train['train_seconds']}s, peak memory: {train['peak_memory_gb']} GB",
                  "- val NLL by epoch: " + ", ".join(f"{e['val_nll']:.3f}" for e in train["log"])]
    lines += ["", "ECE/Brier/AURC use `answer_confidence` (max calibrated probability). "
              "flip rate: choice answers changed by reversing option order. media reliance: answers changed when "
              "the image is swapped for flat gray (higher = the model actually looks). ![reliability](reliability.png)"]
    return "\n".join(lines) + "\n"


def reliability_plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="perfect")
    for name, r in results.items():
        conf, corr = np.array(r["calib"]).T
        acc, mean_conf = calibration_curve(corr, conf, n_bins=8, strategy="quantile")
        ax.plot(mean_conf, acc, marker="o", label=name)
    ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1), title="Reliability")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def compare(cfg, min_accuracy=None, max_ece=None):
    dc, out = cfg["data"], Path(cfg["out"])
    rows = preprocess(load_jsonl(dc["test"]), dc.get("preprocess"))
    qdefs = {q: d for r in rows for q, d in r["questions"].items()}
    gray = str(Path(tempfile.gettempdir()) / "dmtune_gray.png")
    Image.new("RGB", (512, 512), (128, 128, 128)).save(gray)
    device = cfg.get("train", {}).get("device", "auto")
    runners = [("base", cfg["model"], None), ("fine-tuned", cfg["model"], out)]
    runners += [(r.get("name", r["backend"]), r, None) for r in cfg.get("compare", {}).get("runners", [])]
    results = {}
    for name, mcfg, ckpt in runners:
        runner = load_backend(copy.deepcopy(mcfg), checkpoint=ckpt, device=device)
        results[name] = score_runner(name, runner, rows, qdefs, gray)
        del runner
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    labels = [(c, expected_of(r, c)) for r in rows for c in r["questions"]]
    freq = Counter(labels)
    majority = sum(max(n for (q2, _), n in freq.items() if q2 == q) for q in qdefs) / len(labels)
    train = json.loads((out / "train_log.json").read_text()) if (out / "train_log.json").exists() else None
    (out / "report.md").write_text(markdown(results, majority, train))
    reliability_plot(results, out / "reliability.png")
    (out / "report.json").write_text(json.dumps(
        {"majority_accuracy": majority, "train": train,
         "runners": {n: {k: v for k, v in r.items() if k != "calib"} for n, r in results.items()}}, indent=2, default=str))
    print(f"wrote {out / 'report.md'}")
    tuned = results["fine-tuned"]
    failed = []
    if min_accuracy is not None and tuned["extras"]["accuracy"] < min_accuracy:
        failed.append(f"accuracy {tuned['extras']['accuracy']:.3f} < {min_accuracy}")
    if max_ece is not None and tuned["overall"].get("ece", 1.0) > max_ece:
        failed.append(f"ece {tuned['overall']['ece']:.3f} > {max_ece}")
    return results, failed
