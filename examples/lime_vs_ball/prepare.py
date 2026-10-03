"""Build the lime vs green-ball demo set from two public HF datasets of real photos.

    lime        <- Docty/Fruits-262   label "lime"
    green_ball  <- ilee0022/Caltech-256  text "tennis-ball"

Only the parquet row groups that hold matching rows are downloaded.
Writes data/images/*.jpg and data/{train,val,test}.jsonl (60/20/20, seeded).
"""
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

OUT = Path(__file__).parent / "data"
PER_CLASS = 100
SIDE = 384

QUESTIONS = {
    "kind": {
        "type": "choice",
        "instructions": "What is the round green object in this photo?",
        "criteria": {
            "lime": "a lime: citrus fruit with a glossy, dimpled green peel",
            "green_ball": "a green ball: a fuzzy or smooth sports ball, such as a tennis ball",
        },
    },
    "is_lime": {"type": "noul", "instructions": "The photo shows a lime (the citrus fruit)."},
}


def fetch(fs, repo, col, value, n):
    out = []
    for path in sorted(fs.glob(f"datasets/{repo}/data/*.parquet")):
        pf = pq.ParquetFile(fs.open(path))
        ci = pf.metadata.schema.names.index(col)
        for rg in range(pf.num_row_groups):
            st = pf.metadata.row_group(rg).column(ci).statistics
            if st is not None and st.has_min_max and not (st.min <= value <= st.max):
                continue  # sorted shards (Fruits-262) skip almost everything here
            if value not in pf.read_row_group(rg, columns=[col]).column(col).to_pylist():
                continue
            rows = pf.read_row_group(rg, columns=["image", col]).to_pylist()
            out += [r["image"]["bytes"] for r in rows if r[col] == value]
            print(f"  {repo}: {len(out)}/{n}", flush=True)
            if len(out) >= n:
                return out[:n]
    return out


def label_id(fs, repo, name):
    pf = pq.ParquetFile(fs.open(sorted(fs.glob(f"datasets/{repo}/data/*.parquet"))[0]))
    meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"].index(name)


def main():
    fs = HfFileSystem()
    sources = {
        "lime": fetch(fs, "Docty/Fruits-262", "label", label_id(fs, "Docty/Fruits-262", "lime"), PER_CLASS),
        "green_ball": fetch(fs, "ilee0022/Caltech-256", "text", "tennis-ball", PER_CLASS),
    }
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, blobs in sources.items():
        for i, b in enumerate(blobs):
            img = Image.open(io.BytesIO(b)).convert("RGB")
            img.thumbnail((SIDE, SIDE))
            rel = f"images/{cls}_{i:03d}.jpg"
            img.save(OUT / rel, quality=90)
            rows.append({"state": {"image": rel}, "questions": QUESTIONS,
                         "expected": {"kind": cls, "is_lime": cls == "lime"}, "tags": [cls]})
    n = min(len(v) for v in sources.values())  # balance classes
    rng = random.Random(0)
    splits = {"train": [], "val": [], "test": []}
    for cls in sources:
        cls_rows = [r for r in rows if r["tags"] == [cls]][:n]
        rng.shuffle(cls_rows)
        a, b = int(0.6 * n), int(0.8 * n)
        splits["train"] += cls_rows[:a]
        splits["val"] += cls_rows[a:b]
        splits["test"] += cls_rows[b:]
    for name, split in splits.items():
        rng.shuffle(split)
        with open(OUT / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in split)
        print(name, len(split))


if __name__ == "__main__":
    main()
