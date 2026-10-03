"""dmtune train|compare|predict <recipe.yaml>"""
import argparse
import json
import sys
from pathlib import Path

import yaml


def load_cfg(path):
    cfg = yaml.safe_load(Path(path).read_text())
    base = Path(path).resolve().parent
    for k in ("train", "val", "test"):
        if k in cfg.get("data", {}):
            cfg["data"][k] = str(base / cfg["data"][k])
    cfg["out"] = str(base / cfg.get("out", "runs/latest"))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dmtune", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="fine-tune; writes weights, calibration and train_log.json to `out`").add_argument("recipe")
    c = sub.add_parser("compare", help="base vs fine-tuned (+ extra runners) on the test split -> report.md/json/png")
    c.add_argument("recipe")
    c.add_argument("--min-accuracy", type=float)
    c.add_argument("--max-ece", type=float)
    p = sub.add_parser("predict", help="answer one state with the fine-tuned model")
    p.add_argument("recipe")
    p.add_argument("--state", help="JSON state, e.g. '{\"body\": \"...\"}'")
    p.add_argument("--image", help="shortcut for --state '{\"image\": PATH}'")
    p.add_argument("--base", action="store_true", help="use the base model, not the fine-tuned one")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.recipe)

    if a.cmd == "train":
        from .train import train

        train(cfg)
    elif a.cmd == "compare":
        from .report import compare

        _, failed = compare(cfg, a.min_accuracy, a.max_ece)
        if failed:
            sys.exit("gate failed: " + "; ".join(failed))
    else:
        from .data import load_jsonl
        from .model import load_backend

        state = json.loads(a.state) if a.state else {}
        if a.image:
            state["image"] = str(Path(a.image).resolve())
        questions = cfg.get("questions") or load_jsonl(cfg["data"]["test"])[0]["questions"]
        be = load_backend(cfg["model"], checkpoint=None if a.base else cfg["out"],
                          device=cfg.get("train", {}).get("device", "auto"))
        print(json.dumps(be.predict(state, questions), indent=2))


if __name__ == "__main__":
    main()
