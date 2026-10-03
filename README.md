# dmtune

Fine-tune open **decision models** on your own data, then get an honest **before/after report**. Decision models are System-1 models such as Laya: instead of writing text, they answer a typed `choice` / `score` / `noul` question about a state with calibrated probabilities. dmtune covers text, images, video and speech, and it is built to run on a laptop: Apple Silicon or an RTX 3050 with 4 GB.

**Supported inputs:**

| You have | Backend | Model | Size |
|---|---|---|---|
| text, JSON, tables, conversations | `laya` | [Laya](https://github.com/NandhaKishorM/laya) (ModernBERT-large); `subfolder: multilingual` for mmBERT | 421M / 322M |
| images | `vision` | [ModernVBERT](https://huggingface.co/ModernVBERT/modernvbert) + Laya's option-marker head | 250M |
| video | `vision` + `storyboard` preprocessor | frames tiled into one image | 250M |
| speech | `laya` + `transcribe` preprocessor | Whisper-base, then Laya | 74M + 421M |
| Jev, laya-serve, Kev | `systemone` (evaluation only) | any `/v1/systemone` endpoint | — |

Background on Jev, Laya, Kev and the rest of the decision-model class is in [docs/research.md](docs/research.md).

## Quickstart: lime vs green ball (image)

```bash
git clone https://github.com/SwapnanilAdhikary/DmTune.git && cd DmTune
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[media,data,test]"
.venv/bin/python examples/lime_vs_ball/prepare.py            # ~200 real photos from two public HF datasets
.venv/bin/dmtune train   examples/lime_vs_ball/recipe.yaml
.venv/bin/dmtune compare examples/lime_vs_ball/recipe.yaml   # -> runs/lime_vs_ball/report.md
.venv/bin/dmtune predict examples/lime_vs_ball/recipe.yaml --image my_photo.jpg
```

## Measured on an Apple M5 laptop (MPS)

| Demo | Model | Base → fine-tuned accuracy | ECE | Other | Train time / peak memory |
|---|---|---|---|---|---|
| lime vs green ball, 40 test photos | ModernVBERT + LoRA | 0.50 → **0.988** | 0.084 → 0.014 | option-order flip rate 1.0 → 0.0 | 9 min / 3.7 GB |
| typed-decisions, 200 test states, 300 training rows, 2 epochs | Laya + LoRA | 0.42 → **0.705** | 0.152 → 0.122 | flip rate 0.36 → 0.06; auto-decidable at ≤5% risk 0 → 15% | 31 min / 4.5 GB |

Full reports are under `examples/*/runs/*/report.md`. The single lime error is a wire basket full of tennis balls, called "lime" with confidence 1.0. A 40-image validation split is too small for temperature fitting to fix over-confidence like that.

## Your own data

Write one JSONL row per state. This is Jev/Laya's wire schema, plus media paths and optional soft labels:

```json
{"state": {"body": "Charged twice, refund me", "image": "imgs/receipt.jpg"},
 "questions": {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments, refunds", "tech": "bugs"}},
               "urgent": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
               "churn": {"type": "noul", "instructions": "The customer threatens to cancel."}},
 "expected": {"dept": "billing", "urgent": 1, "churn": false},
 "target": {"dept": {"billing": 0.9, "tech": 0.1}},
 "tags": ["web"]}
```

- `expected` holds hard labels.
- `target` is optional and holds soft labels; it overrides `expected` during training.
- `tags` become report slices.
- Media paths are relative to the JSONL file.

Next, copy a recipe (from `examples/` or `recipes/`) and point its `data:` entries at your train, val and test files. Then run `train` and `compare`.

## What training does

Training follows Laya's own recipe:
1. **Loss:** cross-entropy + **RLCD** (REINFORCE with strictly proper scoring-rule rewards), using `laya.common.proper_reward`.
2. **Option shuffling:** option order is shuffled every step, which fixes the option-order bias documented for Jev.
3. **Temperature fitting:** after training, `laya.calibrate` fits one temperature per (question type, number of options) on the validation split.

**Training modes:**
- `lora` (default): fits in 4 GB.
- `head`: the encoder is frozen; works on CPU.
- `full`: trains everything; needs about 10 GB or more.

**Precision:** CUDA uses bf16 on Ampere and newer (RTX 30xx and up), or fp16 with a scaler on Turing (RTX 20xx). MPS and CPU run in fp32.

**Laya export:** `export_laya: true` also writes a plain Laya checkpoint, which `laya.load`, `laya-serve` and Laya's ONNX export can read.

## What `compare` reports

Base vs fine-tuned (plus any extra `compare.runners`, such as Jev) on the test split. Output goes to `report.md`, `report.json` and `reliability.png`:

| Group | Metrics |
|---|---|
| Accuracy | accuracy, choice/noul accuracy, macro-F1, balanced accuracy, noul AUROC/AUPRC, score MAE/QWK/RPS, majority-class baseline |
| Calibration | ECE, Brier, NLL, reliability diagram |
| Selective risk | AURC, selective accuracy @50%/@80% coverage, coverage at ≤5% risk (how much you can automate) |
| Robustness | permutation flip rate; media reliance (answers that change when the image is blanked); per-tag slices for adversarial sets |
| Cost | latency p50/p95, peak memory, training time, trainable parameters |

**CI gates:** `dmtune compare recipe.yaml --min-accuracy 0.9 --max-ece 0.1` exits non-zero if a gate fails.

**Comparing against Jev:** add it as a runner. It is text-only and needs your API key.

```yaml
compare:
  runners:
    - {backend: systemone, name: jev, url: https://api.typesafe.ai/v1/systemone, model: jev-latest, api_key_env: TYPESAFE_API_KEY}
```

## Tests

`.venv/bin/python -m pytest -q tests`. These run offline.

## Not built yet (add when needed)

- A VLM letter-logit backend (SmolVLM2 / Qwen3.5-0.8B), for a stronger zero-shot start when you have few labels.
- Kev training, for more than 20 options.
- Multi-frame / V-JEPA 2 video, for when motion matters.
- CLAP, for non-speech sounds.
- ONNX export for the `vision` backend.
- Multi-GPU training.
- A web UI.
