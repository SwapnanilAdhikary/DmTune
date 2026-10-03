"""Convert LocalLLaMA/typed-decisions (the benchmark Laya reports on) to dmtune JSONL, soft targets kept.

The quick check trains on a subset (N_TRAIN rows); set N_TRAIN = None for the full Laya reproduction.
"""
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT = Path(__file__).parent / "data"
N_TRAIN, N_VAL, N_TEST = 300, 100, 200


def rows(split):
    path = hf_hub_download("LocalLLaMA/typed-decisions", f"all/{split}-00000-of-00001.parquet", repo_type="dataset")
    out = []
    for r in pq.read_table(path).to_pylist():
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        expected = {q: (g["label"] == "true" if qs[q]["type"] == "noul" else
                        int(g["label"]) if qs[q]["type"] == "score" else g["label"]) for q, g in gold.items()}
        out.append({"state": json.loads(r["state"]), "questions": qs, "expected": expected,
                    "target": {q: g["probabilities"] for q, g in gold.items()}, "tags": [r["workflow"]]})
    return out


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(0)
    train = rows("train")
    rng.shuffle(train)
    test = rows("test")
    rng.shuffle(test)
    splits = {"train": train[N_VAL:][:N_TRAIN], "val": train[:N_VAL], "test": test[:N_TEST]}
    for name, split in splits.items():
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split))
        print(name, len(split))


if __name__ == "__main__":
    main()
