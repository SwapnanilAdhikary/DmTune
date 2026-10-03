"""One training loop for every backend: CE + RLCD (Laya's recipe), option shuffling, calibration."""
import json
import math
import random
import resource
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from laya.calibrate import fit_temperature_map
from laya.common import proper_reward

from .data import load_jsonl, preprocess, target
from .model import load_backend


def rlcd_loss(logits, tgt, qtype, mask, sigma, rl_weight=1.0, samples=4):
    """CE + REINFORCE over Gaussian-perturbed logits rewarded by a strictly proper score.

    Same maths as Laya's laya_finetune_typed_decisions_mps.py. Returns (loss, ce).
    """
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    noisy = logits.detach().unsqueeze(0) + eps
    with torch.no_grad():
        q = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
        r = proper_reward(q, tgt.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    ce = -(tgt * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return ce + rl_weight * -(adv * logp).mean(), ce


def build_items(be, rows, rng=None):
    """(state, question) pairs with targets; rng shuffles option order per item (Kev-style aug)."""
    items = []
    for r in rows:
        for qid, qdef in r["questions"].items():
            t = target(qdef, r.get("expected", {}).get(qid), r.get("target", {}).get(qid))
            if t is None:
                continue
            order = None
            if rng is not None:
                order = list(range(len(t)))
                rng.shuffle(order)
                t = t[order]
            it = be.encode(r["state"], qdef, order)
            it["target"] = t.tolist()
            items.append(it)
    return items


class Memory:
    """Peak accelerator memory, in GB, for the report's cost row."""

    def __init__(self, device):
        self.device, self.peak = device, 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def sample(self):
        if self.device.type == "cuda":
            self.peak = torch.cuda.max_memory_allocated() / 1e9
        elif self.device.type == "mps":
            self.peak = max(self.peak, torch.mps.driver_allocated_memory() / 1e9)
        else:
            self.peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
        return self.peak


def amp(device):
    """Low-power precision policy: CUDA autocasts (bf16 on Ampere+, fp16 + scaler on Turing);
    MPS/CPU stay fp32. ponytail: MPS autocast is skipped for stability; try bf16 if memory-bound."""
    if device.type != "cuda":
        return nullcontext(), False
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dt), dt == torch.float16


@torch.no_grad()
def val_pass(be, items, bs):
    """(mean NLL, accuracy, calibration records) on unshuffled items."""
    be.model.eval()
    nll, correct, records = [], [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = be.collate(chunk)
        logits = be.forward(b).float()
        mask, tgt = b["marker_mask"], b["target"]
        nll += (-(tgt * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)).tolist()
        correct += (logits.argmax(-1) == tgt.argmax(-1)).float().tolist()
        for row, it in zip(logits.cpu().numpy(), chunk):
            k = len(it["markers"])
            records.append((it["qtype"], row[:k], np.asarray(it["target"]), k))
    return float(np.mean(nll)), float(np.mean(correct)), records


def train(cfg):
    t0 = time.time()
    tc, dc = cfg.get("train", {}), cfg["data"]
    seed = tc.get("seed", 0)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    rng = random.Random(seed)

    be = load_backend(cfg["model"], device=tc.get("device", "auto"))
    be.setup_training(tc)
    dev = be.device
    tr = preprocess(load_jsonl(dc["train"]), dc.get("preprocess"))
    va = preprocess(load_jsonl(dc["val"]), dc.get("preprocess"))
    va_items = build_items(be, va)

    mb, accum, epochs = tc.get("micro_batch", 4), tc.get("grad_accum", 4), tc.get("epochs", 4)
    n_items = sum(len(r["questions"]) for r in tr)
    steps = epochs * math.ceil(n_items / (mb * accum))
    opt = torch.optim.AdamW(be.param_groups(tc.get("lr_encoder", 2e-4), tc.get("lr_head", 5e-4)), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-6)
    ctx, fp16 = amp(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    mem = Memory(dev)

    n_params = sum(p.numel() for p in be.model.parameters() if p.requires_grad)
    print(f"device={dev.type} mode={be.mode} trainable={n_params / 1e6:.2f}M items/epoch={n_items} steps={steps}")
    nll0, acc0, _ = val_pass(be, va_items, mb)
    log = [{"epoch": 0, "val_nll": nll0, "val_acc": acc0}]
    print(f"epoch 0  val_nll={nll0:.4f} val_acc={acc0:.3f}")
    best, best_state = nll0, be.trainable_state()

    for epoch in range(epochs):
        be.model.train()
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, epochs - 1)
        items = build_items(be, tr, rng if tc.get("shuffle_options", True) else None)
        rng.shuffle(items)
        losses = []
        for i in range(0, len(items), mb):
            b = be.collate(items[i:i + mb])
            with ctx:
                logits = be.forward(b)
            loss, ce = rlcd_loss(logits.float(), b["target"], b["qtype"], b["marker_mask"], sigma,
                                 tc.get("rl_weight", 1.0))
            scaler.scale(loss / accum).backward()
            losses.append(ce.item())
            if (i // mb + 1) % accum == 0 or i + mb >= len(items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            mem.sample()
        nll, acc, _ = val_pass(be, va_items, mb)
        log.append({"epoch": epoch + 1, "train_ce": float(np.mean(losses)), "val_nll": nll, "val_acc": acc})
        print(f"epoch {epoch + 1}  train_ce={np.mean(losses):.4f} val_nll={nll:.4f} val_acc={acc:.3f}  peak={mem.peak:.2f}GB")
        if nll < best:
            best, best_state = nll, be.trainable_state()

    be.model.load_state_dict(best_state, strict=False)
    _, _, records = val_pass(be, va_items, mb)
    cal = fit_temperature_map(records)
    be.temps = {"temperature": [float(t) for t in cal["temperature"]],
                "temperature_by_options": {k: float(v) for k, v in cal["temperature_by_options"].items()}}
    out = Path(cfg["out"])
    stats = {"train_seconds": round(time.time() - t0, 1), "peak_memory_gb": round(mem.peak, 3),
             "device": dev.type, "trainable_params_m": round(n_params / 1e6, 3), "log": log}
    be.save(out, {"train": stats})
    if cfg["model"]["backend"] == "laya" and tc.get("export_laya"):
        be.export_laya(out)
    (out / "train_log.json").write_text(json.dumps(stats, indent=2))
    print(f"saved {out}  best val_nll={best:.4f}  temps={be.temps}")
    return out
