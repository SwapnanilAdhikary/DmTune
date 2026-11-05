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

from .data import load_jsonl, option_keys, preprocess, target
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


def objective_loss(name, logits, tgt, qtype, mask, sigma, tc):
    """Training objective switch. Returns (loss, ce). `rlcd` is dmtune's default (CE + REINFORCE)."""
    if name == "rlcd":
        return rlcd_loss(logits, tgt, qtype, mask, sigma, tc.get("rl_weight", 1.0))
    z = logits.masked_fill(~mask, -1e4)
    logp, p = torch.log_softmax(z, -1), torch.softmax(z, -1)
    ce = -(tgt * logp).sum(-1).mean()
    k = mask.sum(-1, keepdim=True).float()
    if name == "ce":
        return ce, ce
    if name == "brier":
        return (((p - tgt) ** 2) * mask).sum(-1).mean(), ce
    if name == "focal":
        return -(((1 - p) ** tc.get("focal_gamma", 2.0)) * tgt * logp).sum(-1).mean(), ce
    if name == "ce_ls":
        eps = tc.get("ls_eps", 0.1)
        return -(((1 - eps) * tgt + eps / k * mask) * logp).sum(-1).mean(), ce
    if name == "ce_sd":  # spectral decoupling: CE + L2 on logits (Pezeshki et al. 2021, Puli's margin limiter)
        return ce + tc.get("sd_weight", 0.05) * ((logits ** 2) * mask).sum(-1).div(k.squeeze(-1)).mean(), ce
    if name == "poly1":  # PolyLoss-1: CE + eps*(1-p_true). ponytail: still exponentially tailed, not the polynomial-tailed arm
        return ce + tc.get("poly_eps", 2.0) * (1 - (p * tgt).sum(-1)).mean(), ce
    raise ValueError(f"unknown loss {name!r}")


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
            it["tag"] = (r.get("tags") or [None])[0]
            cue = r.get("meta", {}).get("cue")
            keys = option_keys(qdef)
            it["cue_idx"] = keys.index(str(cue)) if cue is not None and str(cue) in keys else None
            if order is not None and it["cue_idx"] is not None:
                it["cue_idx"] = order.index(it["cue_idx"])
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


@torch.no_grad()
def eval_split(be, items, bs, keep=False):
    """acc, nll, mean P(cue option), mean true-label margin on an unshuffled split; keep=True adds per-item logits."""
    be.model.eval()
    acc, nll, cue, marg, zs = [], [], [], [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = be.collate(chunk)
        z = be.forward(b).float().masked_fill(~b["marker_mask"], -1e4)
        tgt = b["target"]
        lp, pr = torch.log_softmax(z, -1), torch.softmax(z, -1)
        nll += (-(tgt * lp).sum(-1)).tolist()
        acc += (z.argmax(-1) == tgt.argmax(-1)).float().tolist()
        true = tgt.argmax(-1)
        zt = z.gather(1, true[:, None]).squeeze(1)
        marg += (zt - z.scatter(1, true[:, None], -1e4).max(-1).values).tolist()
        for j, it in enumerate(chunk):
            if it.get("cue_idx") is not None:
                cue.append(pr[j, it["cue_idx"]].item())
            if keep:
                zs.append([round(v, 4) for v in z[j, :len(it["markers"])].tolist()])
    out = {"acc": float(np.mean(acc)), "nll": float(np.mean(nll)), "margin": float(np.mean(marg)),
           "cue_prob": float(np.mean(cue)) if cue else None}
    if keep:  # for post-hoc temperature scaling in the analysis
        out.update(z=zs, y=[int(np.argmax(it["target"])) for it in items], cue=[it.get("cue_idx") for it in items])
    return out


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
    evals = {n: build_items(be, preprocess(load_jsonl(pth), dc.get("preprocess"))) for n, pth in dc.get("evals", {}).items()}
    loss_name = tc.get("loss", "rlcd")

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
    log = [{"epoch": 0, "val_nll": nll0, "val_acc": acc0, "evals": {n: eval_split(be, it, mb) for n, it in evals.items()}}]
    print(f"epoch 0  val_nll={nll0:.4f} val_acc={acc0:.3f} loss={loss_name}")
    best, best_state = nll0, be.trainable_state()

    for epoch in range(epochs):
        be.model.train()
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, epochs - 1)
        items = build_items(be, tr, rng if tc.get("shuffle_options", True) else None)
        rng.shuffle(items)
        losses, g_all, g_conf, marg = [], 0.0, 0.0, {}
        for i in range(0, len(items), mb):
            chunk = items[i:i + mb]
            b = be.collate(chunk)
            with ctx:
                logits = be.forward(b)
            lf = logits.float()
            lf.retain_grad()
            loss, ce = objective_loss(loss_name, lf, b["target"], b["qtype"], b["marker_mask"], sigma, tc)
            scaler.scale(loss / accum).backward()
            gn = lf.grad.norm(dim=-1).tolist()  # mechanism: share of the logit-gradient carried by conflict examples
            z, tg = lf.detach().masked_fill(~b["marker_mask"], -1e4), b["target"].argmax(-1)
            m = (z.gather(1, tg[:, None]).squeeze(1) - z.scatter(1, tg[:, None], -1e4).max(-1).values).tolist()
            for it, g, mm in zip(chunk, gn, m):
                g_all += g
                g_conf += g * (it["tag"] == "conflict")
                marg.setdefault(it["tag"], []).append(mm)
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
        ev = {n: eval_split(be, it, mb) for n, it in evals.items()}
        log.append({"epoch": epoch + 1, "train_ce": float(np.mean(losses)), "val_nll": nll, "val_acc": acc,
                    "grad_share_conflict": g_conf / max(g_all, 1e-12),
                    "train_margin": {str(k): float(np.mean(v)) for k, v in marg.items()}, "evals": ev})
        print(f"epoch {epoch + 1}  train_ce={np.mean(losses):.4f} val_nll={nll:.4f} val_acc={acc:.3f}  peak={mem.peak:.2f}GB  "
              + " ".join(f"{n}={e['acc']:.3f}" for n, e in ev.items()) + f"  gshare={log[-1]['grad_share_conflict']:.3f}")
        if tc.get("select", "best_val") == "best_val" and nll < best:
            best, best_state = nll, be.trainable_state()

    if tc.get("select", "best_val") == "best_val":
        be.model.load_state_dict(best_state, strict=False)
    final_logits = ({n: eval_split(be, it, mb, keep=True) for n, it in {**evals, "val": va_items}.items()}
                    if tc.get("save_logits") else None)
    _, _, records = val_pass(be, va_items, mb)
    cal = fit_temperature_map(records)
    be.temps = {"temperature": [float(t) for t in cal["temperature"]],
                "temperature_by_options": {k: float(v) for k, v in cal["temperature_by_options"].items()}}
    out = Path(cfg["out"])
    stats = {"train_seconds": round(time.time() - t0, 1), "peak_memory_gb": round(mem.peak, 3),
             "device": dev.type, "trainable_params_m": round(n_params / 1e6, 3), "log": log}
    if final_logits:
        stats["final_logits"] = final_logits
    if tc.get("save", True):
        be.save(out, {"train": stats})
    out.mkdir(parents=True, exist_ok=True)
    if cfg["model"]["backend"] == "laya" and tc.get("export_laya"):
        be.export_laya(out)
    (out / "train_log.json").write_text(json.dumps(stats, indent=2))
    print(f"saved {out}  best val_nll={best:.4f}  temps={be.temps}")
    return out
