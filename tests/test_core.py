"""Offline checks (no model downloads): schema, targets, loss, head, metrics, storyboard."""
import json

import numpy as np
import pytest
import torch

from dmtune.data import decode, expected_of, load_jsonl, option_keys, target
from dmtune.report import coverage_at_risk, extras
from dmtune.train import rlcd_loss

CHOICE = {"type": "choice", "instructions": "?", "criteria": {"lime": "", "green_ball": ""}}
NOUL = {"type": "noul", "instructions": "is lime"}
SCORE = {"type": "score", "instructions": "how", "criteria": ["low", "mid", "high"]}


def test_targets_and_decode():
    assert option_keys(NOUL) == ["false", "true"]
    assert target(CHOICE, "green_ball").tolist() == [0, 1]
    assert target(NOUL, True).tolist() == [0, 1]
    assert target(SCORE, 2).tolist() == [0, 0, 1]
    assert np.allclose(target(NOUL, soft=0.8), [0.2, 0.8])
    assert np.allclose(target(CHOICE, soft={"lime": 3, "green_ball": 1}), [0.75, 0.25])
    assert target(CHOICE) is None
    a = decode(SCORE, np.array([0.2, 0.3, 0.5]))
    assert a["score"] == pytest.approx(1.3) and a["answer_confidence"] == 0.5
    assert decode(CHOICE, np.array([0.1, 0.9]))["choice"] == "green_ball"


def test_load_jsonl_resolves_media(tmp_path):
    row = {"state": {"image": "a.jpg"}, "questions": {"k": CHOICE}, "target": {"k": {"lime": 1}}}
    (tmp_path / "d.jsonl").write_text("# comment\n" + json.dumps(row) + "\n\n")
    rows = load_jsonl(tmp_path / "d.jsonl")
    assert rows[0]["state"]["image"] == str(tmp_path / "a.jpg")
    assert expected_of(rows[0], "k") == "lime"
    (tmp_path / "bad.jsonl").write_text(json.dumps({"state": "x", "questions": {}}))
    with pytest.raises(ValueError):
        load_jsonl(tmp_path / "bad.jsonl")


def test_rlcd_loss_learns():
    torch.manual_seed(0)
    logits = torch.zeros(3, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    tgt = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float)
    qtype = torch.tensor([0, 1, 2])
    opt = torch.optim.SGD([logits], lr=0.5)
    first = None
    for _ in range(30):
        loss, ce = rlcd_loss(logits.masked_fill(~mask, -1e4), tgt, qtype, mask, sigma=0.3)
        first = first if first is not None else ce.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert ce.item() < first * 0.5
    assert torch.isfinite(loss)


def test_head_logits_matches_laya():
    """Our vision head path must equal laya's DecisionModel.forward on the same hidden states."""
    from laya.common import DecisionModel
    from transformers import ModernBertConfig, ModernBertModel

    from dmtune.model import head_logits

    torch.manual_seed(0)
    enc = ModernBertModel(ModernBertConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                                           num_hidden_layers=2, num_attention_heads=2, pad_token_id=0))
    dm = DecisionModel(enc, head_layers=2).eval()
    ids = torch.randint(1, 100, (2, 12))
    att = torch.ones_like(ids)
    att[1, 9:] = 0
    b = {"input_ids": ids, "attention_mask": att, "marker_pos": torch.tensor([[2, 5, 7], [2, 4, 0]]),
         "marker_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool), "qtype": torch.tensor([0, 2])}
    with torch.no_grad():
        ref = dm(ids, att, b["marker_pos"], b["marker_mask"], b["qtype"])[0]
        ours = head_logits(dm, enc(input_ids=ids, attention_mask=att).last_hidden_state, b)
    assert torch.allclose(ref, ours, atol=1e-5)


def test_metrics():
    assert coverage_at_risk(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 0, 1.0]), 0.05) == 0.5
    cases = [{"qid": "n", "expected": e, "confidence": max(p, 1 - p), "correct": (p >= 0.5) == e,
              "answer": {"noul": p}} for e, p in [(True, 0.9), (False, 0.2), (True, 0.6), (False, 0.4)]]
    cases.append({"qid": "s", "expected": 2, "confidence": 0.6, "correct": None,  # laya: score has no "correct"
                  "answer": {"score": 1.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}})
    m = extras(cases, {"n": NOUL, "s": SCORE})
    assert m["accuracy"] == 1.0 and m["n/auroc"] == 1.0
    assert m["s/rps"] == pytest.approx((0.1 ** 2 + 0.3 ** 2) / 2)
    assert m["nll"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.6, 0.6, 0.7])))


def test_storyboard(tmp_path):
    av = pytest.importorskip("av")
    from PIL import Image

    from dmtune.data import storyboard

    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 32, 32, "yuv420p"
        for i in range(10):
            f = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 25, np.uint8), format="rgb24")
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    st = storyboard({"video": str(path), "body": "x"}, frames=4)
    assert "video" not in st and st["body"] == "x"
    assert Image.open(st["image"]).size == (64, 64)


def test_systemone_runner_with_laya_evals():
    """The HTTP runner speaks Jev's wire format and plugs into laya.evals unchanged."""
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from dmtune.report import dataset
    from dmtune.systemone import SystemOneRunner
    from laya.evals import evaluate

    seen = {}

    class Jev(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.update(body, auth=self.headers.get("authorization"))
            out = {"answers": {"k": {"type": "choice", "choice": "lime", "probabilities": {"lime": 0.8, "green_ball": 0.2},
                                     "confidence": 0.6}}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Jev)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["DMTUNE_TEST_KEY"] = "secret"
    runner = SystemOneRunner({"url": f"http://127.0.0.1:{srv.server_port}/v1/systemone", "model": "jev-latest",
                              "api_key_env": "DMTUNE_TEST_KEY"})
    rows = [{"state": {"body": "a small green citrus"}, "questions": {"k": CHOICE}, "expected": {"k": "lime"}}]
    rep = evaluate(runner, dataset(rows))
    srv.shutdown()
    assert rep.overall["choice_accuracy"] == 1.0
    assert seen["model"] == "jev-latest" and seen["auth"] == "Bearer secret"
    with pytest.raises(ValueError):
        runner.predict({"image": "x.jpg"}, {"k": CHOICE})
