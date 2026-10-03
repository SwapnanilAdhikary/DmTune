"""Dataset rows (Jev/Laya wire schema + media paths), targets, answer decoding, preprocessors.

A row:
    {"state": {"body": "...", "image": "imgs/a.jpg"},
     "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {...}}},
     "expected": {"q": "label"},              # hard label: choice key / bool / level index
     "target":   {"q": {"label": 0.9, ...}},  # optional soft label, overrides expected
     "tags": ["slice"]}
"""
import json
import math
from pathlib import Path

import numpy as np

MEDIA = ("image", "video", "audio")


def load_jsonl(path):
    base = Path(path).parent
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            for key in ("state", "questions"):
                if key not in r:
                    raise ValueError(f"{path}:{n}: missing {key!r}")
            if "expected" not in r and "target" not in r:
                raise ValueError(f"{path}:{n}: needs 'expected' or 'target'")
            if isinstance(r["state"], dict):
                for m in MEDIA:
                    if m in r["state"]:
                        r["state"][m] = str(base / r["state"][m])  # absolute paths pass through
            rows.append(r)
    return rows


def option_keys(qdef):
    t = qdef["type"]
    if t == "choice":
        return [str(k) for k in qdef["criteria"]]
    if t == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(qdef["criteria"]))]


def target(qdef, expected=None, soft=None):
    """Probability vector over options in canonical order, or None if the row has no label."""
    keys = option_keys(qdef)
    if soft is not None:
        if qdef["type"] == "noul" and not isinstance(soft, dict):
            v = np.array([1 - float(soft), float(soft)])
        elif isinstance(soft, dict):
            v = np.array([float(soft.get(k, 0.0)) for k in keys])
        else:
            v = np.array(soft, dtype=float)
        return v / v.sum()
    if expected is None:
        return None
    v = np.zeros(len(keys))
    if qdef["type"] == "noul":
        v[int(bool(expected))] = 1.0
    else:
        v[keys.index(str(expected))] = 1.0
    return v


def expected_of(row, qid):
    """Hard label for evaluation: `expected`, else the argmax of the soft `target`."""
    if qid in row.get("expected", {}):
        return row["expected"][qid]
    qdef = row["questions"][qid]
    k = option_keys(qdef)[int(np.argmax(target(qdef, soft=row["target"][qid])))]
    return k == "true" if qdef["type"] == "noul" else (int(k) if qdef["type"] == "score" else k)


def decode(qdef, p):
    """Calibrated option probabilities -> a Jev/Laya-shaped answer."""
    conf = float(p.max())
    if qdef["type"] == "choice":
        keys = option_keys(qdef)
        return {"type": "choice", "choice": keys[int(p.argmax())],
                "probabilities": {k: float(v) for k, v in zip(keys, p)}, "answer_confidence": conf}
    if qdef["type"] == "noul":
        return {"type": "noul", "noul": float(p[1]), "answer_confidence": conf}
    return {"type": "score", "score": float((np.arange(len(p)) * p).sum()),
            "probabilities": {str(i): float(v) for i, v in enumerate(p)}, "answer_confidence": conf}


# --- preprocessors: plain state -> state functions, listed under data.preprocess in the YAML ---

def storyboard(state, frames=4):
    """Video -> one grid image of evenly spaced frames, so image models can judge clips.

    ponytail: temporal resolution is capped at `frames`; upgrade to multi-frame encoders
    (V-JEPA 2) when motion matters more than appearance.
    """
    import av
    from PIL import Image

    with av.open(state["video"]) as c:
        imgs = [f.to_image() for f in c.decode(video=0)]  # ponytail: decodes whole clip, fine for short clips
    picks = [imgs[round(i * (len(imgs) - 1) / max(1, frames - 1))] for i in range(frames)]
    g = math.ceil(math.sqrt(frames))
    w, h = picks[0].size
    sheet = Image.new("RGB", (g * w, math.ceil(frames / g) * h))
    for i, im in enumerate(picks):
        sheet.paste(im, ((i % g) * w, (i // g) * h))
    out = Path(state["video"]).with_suffix(".storyboard.jpg")
    sheet.save(out)
    return {**{k: v for k, v in state.items() if k != "video"}, "image": str(out)}


_ASR = {}


def load_audio(path, rate=16000):
    import av

    with av.open(path) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = [r.to_ndarray() for f in c.decode(audio=0) for r in rs.resample(f)]
        chunks += [r.to_ndarray() for r in rs.resample(None)]
    return np.concatenate(chunks, axis=1)[0]


def transcribe(state, model="openai/whisper-base"):
    """Speech -> state.transcript, so the text decision model can judge calls and voice notes."""
    from transformers import pipeline

    if model not in _ASR:
        _ASR[model] = pipeline("automatic-speech-recognition", model=model, chunk_length_s=30)
    text = _ASR[model]({"raw": load_audio(state["audio"]), "sampling_rate": 16000})["text"]
    return {**{k: v for k, v in state.items() if k != "audio"}, "transcript": text.strip()}


PREPROCESSORS = {"storyboard": storyboard, "transcribe": transcribe}


def preprocess(rows, specs):
    """specs: ["storyboard"] or [{"transcribe": {"model": "openai/whisper-small"}}]."""
    for spec in specs or []:
        name, kw = (spec, {}) if isinstance(spec, str) else next(iter(spec.items()))
        fn = PREPROCESSORS[name]
        for r in rows:
            r["state"] = fn(r["state"], **(kw or {}))
    return rows
