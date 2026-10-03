"""Local decision-model backends. Both use Laya's DecisionModel head (option-marker scoring):

    laya    text (+ JSON/tabular/conversation state, + speech via a transcript)   ModernBERT
    vision  image (+ video via storyboard)                                          ModernVBERT

Shared: LoRA/head/full training modes, per-bucket temperatures, Jev-shaped predict(), save/load.
"""
import json
from pathlib import Path

import numpy as np
import torch
from laya.agent import Agent
from laya.common import (QTYPES, DecisionModel, build_head, build_sequence, collate_items,
                         serialize_state, temp_bucket)
from safetensors.torch import load_file, save_file

from .data import MEDIA, decode


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def head_logits(dm, h, b):
    """DecisionModel.forward minus the encoder call and act head, for encoders that need extra inputs."""
    h = h + dm.type_emb(b["qtype"])[:, None, :]
    pad = ~b["attention_mask"].bool()
    for layer in dm.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    m = torch.gather(h, 1, b["marker_pos"][:, :, None].expand(-1, -1, h.size(-1)))
    return dm.scorer(m).squeeze(-1).float().masked_fill(~b["marker_mask"], -1e4)


class DecisionBackend:
    lora_targets = ["Wqkv", "Wo", "Wi"]  # ModernBERT attention + MLP

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg.get("name") or cfg["base"]
        self.temps = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
        self.lora = None
        self.device = torch.device("cpu")

    # -- subclass hooks: encode(state, qdef, order) -> item; forward(batch) -> logits[B, K]

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def collate(self, items):
        b = collate_items([items], self.tok.pad_token_id)
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in b.items()}

    def setup_training(self, tcfg):
        mode = tcfg.get("mode", "lora")
        for p in self.model.parameters():
            p.requires_grad = mode == "full"
        for mod in (self.model.head, self.model.type_emb, self.model.scorer):
            for p in mod.parameters():
                p.requires_grad = True
        if mode == "lora":
            lc = tcfg.get("lora") or {}
            self.apply_lora({"r": lc.get("r", 16), "lora_alpha": lc.get("alpha", 32),
                             "lora_dropout": lc.get("dropout", 0.05),
                             "target_modules": lc.get("targets", self.lora_targets)})
        if tcfg.get("grad_checkpointing", True):
            enc = self.model.encoder.get_base_model() if self.lora else self.model.encoder
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.mode = mode

    def apply_lora(self, lora_cfg):
        from peft import LoraConfig, get_peft_model

        self.lora = lora_cfg
        self.model.encoder = get_peft_model(self.model.encoder, LoraConfig(**lora_cfg))

    def param_groups(self, lr_encoder, lr_head):
        enc = {id(p) for p in self.model.encoder.parameters()}
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        return [{"params": [p for _, p in trainable if id(p) in enc], "lr": lr_encoder},
                {"params": [p for _, p in trainable if id(p) not in enc], "lr": lr_head}]

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def save(self, out, extra=None):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.trainable_state().items()}, str(out / "weights.safetensors"))
        meta = {"model": self.cfg, "mode": self.mode, "lora": self.lora, "temps": self.temps, **(extra or {})}
        (out / "dmtune.json").write_text(json.dumps(meta, indent=2))

    def load_finetuned(self, ckpt):
        ckpt = Path(ckpt)
        meta = json.loads((ckpt / "dmtune.json").read_text())
        if meta["lora"]:
            self.apply_lora(meta["lora"])
        res = self.model.load_state_dict(load_file(str(ckpt / "weights.safetensors")), strict=False)
        if res.unexpected_keys:
            raise ValueError(f"checkpoint does not match {self.name}: {res.unexpected_keys[:3]}")
        if self.lora:
            self.model.encoder = self.model.encoder.merge_and_unload()  # plain weights: faster inference
            self.lora = None
        self.temps = meta["temps"]
        self.name = f"{self.name} (fine-tuned)"
        return self

    def probs(self, logits_row, item):
        k, qt = len(item["markers"]), item["qtype"]
        t = self.temps["temperature_by_options"].get(temp_bucket(qt, k), self.temps["temperature"][qt])
        z = logits_row[:k] / t
        p = np.exp(z - z.max())
        return p / p.sum()

    @torch.no_grad()
    def predict(self, state, questions, **_):
        """Jev /v1/systemone-shaped: {"answers": {qid: {...}}}. Also the laya.evals runner contract."""
        self.model.eval()
        qids = list(questions)
        items = [self.encode(state, questions[q]) for q in qids]
        logits = self.forward(self.collate(items)).cpu().numpy()
        return {"model": self.name,
                "answers": {q: decode(questions[q], self.probs(logits[j], items[j])) for j, q in enumerate(qids)}}


class LayaText(DecisionBackend):
    """Laya checkpoint (ModernBERT-large 421M, or mmBERT-base 322M with subfolder=multilingual)."""

    def __init__(self, cfg):
        import laya

        cfg = {"base": "convaiinnovations/laya", **cfg}
        super().__init__(cfg)
        agent = laya.load(cfg["base"], subfolder=cfg.get("subfolder"), device="cpu")
        self.model, self.tok = agent.model.float(), agent.tok
        self.max_len = cfg.get("max_len", agent.cfg.get("max_len", 512))
        self.head_max_len = cfg.get("head_max_len", agent.cfg.get("head_max_len", 192))
        self.agent_cfg = agent.cfg
        self.temps = {"temperature": [float(t) for t in agent.temperature],  # shipped calibration
                      "temperature_by_options": dict(agent.temperature_by_options)}

    def encode(self, state, qdef, order=None):
        q = Agent._to_internal(qdef)
        ids, markers = build_sequence(self.tok, state, q, self.max_len, self.head_max_len,
                                      option_order=order, truncate_left=isinstance(state, list))
        return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]}

    def forward(self, b):
        return self.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])[0]

    def export_laya(self, out):
        """Write a plain Laya checkpoint (laya.load / laya-serve / laya ONNX export can read it)."""
        out = Path(out)
        if self.lora:
            self.model.encoder = self.model.encoder.merge_and_unload()
            self.lora = None
        save_file({k: v.detach().half().cpu().contiguous() for k, v in self.model.state_dict().items()},
                  str(out / "model.safetensors"))
        self.model.encoder.config.save_pretrained(out / "encoder")
        self.tok.save_pretrained(out / "tokenizer")
        (out / "rl_agent_config.json").write_text(json.dumps({**self.agent_cfg, **self.temps, "fine_tuned": True}, indent=2))


class ModernVBertVision(DecisionBackend):
    """ModernVBERT (250M: ModernBERT-family text encoder + SigLIP2 vision) with a fresh Laya-style head.

    Sequence: [CLS] <type> instr [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <image block> state_text [SEP]
    Markers sit before the image block, so its 67 tokens never move them.
    """

    def __init__(self, cfg):
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        cfg = {"base": "ModernVBERT/modernvbert", **cfg}
        super().__init__(cfg)
        self.proc = AutoProcessor.from_pretrained(cfg["base"])
        self.tok = self.proc.tokenizer
        enc = AutoModel.from_pretrained(cfg["base"], dtype=torch.float32)
        enc.config.hidden_size = enc.config.text_config.hidden_size  # DecisionModel sizes its head from this
        torch.manual_seed(cfg.get("seed", 0))  # the head is new: make "before" reproducible
        self.model = DecisionModel(enc, head_layers=cfg.get("head_layers", 2))
        self.max_len = cfg.get("max_len", 512)
        self.head_max_len = cfg.get("head_max_len", 192)
        blank = Image.new("RGB", (64, 64), (127, 127, 127))
        self.image_block = self.proc(text="<image>", images=[blank], add_special_tokens=False,
                                     do_image_splitting=False)["input_ids"][0]
        self.image_block = [int(i) for i in self.image_block]

    def encode(self, state, qdef, order=None):
        from PIL import Image

        if not isinstance(state, dict) or "image" not in state:
            raise ValueError("vision backend needs state.image (use the storyboard preprocessor for video)")
        q = Agent._to_internal(qdef)
        ids, markers, _ = build_head(self.tok, q, self.head_max_len, option_order=order)
        rest = {k: v for k, v in state.items() if k not in MEDIA}
        text = self.tok(serialize_state(rest) if rest else "", add_special_tokens=False)["input_ids"]
        room = max(0, self.max_len - len(ids) - len(self.image_block) - 1)
        ids = ids + self.image_block + text[:room] + [self.tok.sep_token_id]
        img = Image.open(state["image"]).convert("RGB")  # ponytail: re-read per question; cache if I/O shows up
        px = self.proc.image_processor([img], do_image_splitting=False, return_tensors="pt")
        return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]],
                "pixel_values": px["pixel_values"], "pixel_attention_mask": px["pixel_attention_mask"]}

    def collate(self, items):
        b = super().collate(items)
        for k in ("pixel_values", "pixel_attention_mask"):
            b[k] = torch.cat([it[k] for it in items]).to(self.device)
        return b

    def forward(self, b):
        h = self.model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                               pixel_values=b["pixel_values"],
                               pixel_attention_mask=b["pixel_attention_mask"]).last_hidden_state
        return head_logits(self.model, h, b)


BACKENDS = {"laya": LayaText, "vision": ModernVBertVision}


def load_backend(model_cfg, checkpoint=None, device="auto"):
    if model_cfg["backend"] == "systemone":
        from .systemone import SystemOneRunner

        return SystemOneRunner(model_cfg)
    be = BACKENDS[model_cfg["backend"]](model_cfg)
    if checkpoint:
        be.load_finetuned(checkpoint)
    return be.to(pick_device(device))
