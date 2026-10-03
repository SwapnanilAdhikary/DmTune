# dmtune

**Teach a small, fast "decision model" to make *your* decisions, then see exactly how much better it got.**

dmtune fine-tunes open decision models on your own labelled examples: text, images, video or speech. It then writes a before/after report. Everything runs on a laptop: an Apple Silicon Mac, or an NVIDIA RTX card with as little as 4 GB of memory.

```bash
pip install "dmtune[media]"
```

---

## What is a decision model?

A decision model doesn't write text. You give it something to look at (the **state**) and a **question**, and it answers with a probability for each option. It does this in one quick step (tens to hundreds of milliseconds).

```
state:     photo.jpg
question:  "What is the round green object?"   options: lime / green_ball
answer:    lime 0.97, green_ball 0.03
```

There are three kinds of question:

| Type | Use it for | Label you provide |
|---|---|---|
| `choice` | pick one of several options | the option's name, e.g. `"billing"` |
| `score` | rate on an ordered scale | the level's position, starting at 0, e.g. `1` for "medium" in `["low", "medium", "high"]` |
| `noul` | yes/no statements (Jev and Laya call these "noul") | `true` or `false` |

Popular decision models are [Jev](https://typesafe.ai) (closed, API only) and [Laya](https://github.com/NandhaKishorM/laya) (open source). Out of the box, Laya is often wrong on new kinds of decisions: on its own benchmark it scores below "always guess the most common answer". **Fine-tuning on a few hundred of your own examples fixes that.** dmtune makes that fine-tuning easy and measures the result. For background, see [docs/research.md](docs/research.md).

---

## Try it: lime vs green ball (about 15 minutes)

The demo trains an image model to tell limes from tennis balls. The examples live in the GitHub repo, so clone it:

```bash
git clone https://github.com/SwapnanilAdhikary/DmTune.git
cd DmTune
pip install -e ".[media,data]"

python examples/lime_vs_ball/prepare.py              # 1. download ~200 photos, split into train/val/test
dmtune train   examples/lime_vs_ball/recipe.yaml     # 2. fine-tune (≈10 min on a laptop)
dmtune compare examples/lime_vs_ball/recipe.yaml     # 3. before/after report
dmtune predict examples/lime_vs_ball/recipe.yaml --image any_photo.jpg   # 4. ask about a new photo
```

**Step 3 writes** `examples/lime_vs_ball/runs/lime_vs_ball/report.md`. What we got on an M5 MacBook:

| | Before fine-tuning | After |
|---|---|---|
| Accuracy | 50% (coin flip) | **98.8%** |
| Calibration error (lower = more honest confidence) | 0.084 | 0.014 |
| Training time / peak memory | | 9 min / 3.7 GB |

**Step 4 prints** something like:

```json
{"answers": {"kind": {"choice": "lime", "probabilities": {"lime": 1.0, "green_ball": 0.0}},
             "is_lime": {"noul": 0.99}}}
```

---

## Use it on your own data

### Step 1: Write your examples as JSONL (one example per line)

```json
{"state": {"body": "I was charged twice, refund me"}, "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments and refunds", "tech": "bugs and outages"}}, "angry": {"type": "noul", "instructions": "The customer is angry."}}, "expected": {"team": "billing", "angry": true}}
```

| Field | What it is |
|---|---|
| `state` | What the model looks at. Use any text or JSON fields. Add `"image"`, `"video"` or `"audio"` with a file path (relative to the JSONL file). |
| `questions` | One or more questions, each with a `type`, plain-English `instructions` and, for `choice` and `score`, the options in `criteria`. The text after each option name helps the model; write it like a definition. |
| `expected` | The correct answer to each question (see the label table above). |
| `target` *(optional)* | Use instead of `expected` when your labels are uncertain, e.g. `{"team": {"billing": 0.8, "tech": 0.2}}`. |
| `tags` *(optional)* | Labels such as `["mobile", "spanish"]`. The report breaks results down by tag. |

Make three files:
- `train.jsonl`: most of your data;
- `val.jsonl`: about 20%, used to pick the best epoch and calibrate confidence;
- `test.jsonl`: about 20%, used only for the report.

A few hundred examples is a good start.

### Step 2: Write a recipe (a small YAML file)

```yaml
model:
  backend: laya            # what kind of input; see the table below
data:
  train: data/train.jsonl  # paths are relative to this recipe file
  val: data/val.jsonl
  test: data/test.jsonl
out: runs/my_model         # where the model and reports are saved
```

That's all you need. Choose `backend` by what your `state` contains:

| Your state contains | Set | Model used |
|---|---|---|
| text, JSON, tables, chat logs | `backend: laya` | Laya, 421M parameters. Add `subfolder: multilingual` for non-English (322M) |
| images | `backend: vision` | ModernVBERT, 250M |
| video | `backend: vision`, plus `preprocess: [storyboard]` under `data:` | Each clip becomes a 2×2 grid of frames, then judged like an image |
| speech recordings | `backend: laya`, plus `preprocess: [transcribe]` under `data:` | Whisper transcribes, then Laya decides |

Ready-made recipes for each case are in [`examples/`](examples) and [`recipes/`](recipes).

<details>
<summary>Optional training settings (the defaults work for most cases)</summary>

```yaml
train:
  mode: lora             # lora: small add-on weights, fits in 4 GB (default)
                         # head: only the final layer, cheapest, works on CPU
                         # full: retrain everything, needs ~10 GB+
  epochs: 4              # passes over the training data
  micro_batch: 4         # examples per step; lower to 2 if you run out of memory
  grad_accum: 4          # steps per weight update (effective batch = micro_batch × grad_accum)
  lr_encoder: 2.0e-4     # learning rate of the model body
  lr_head: 5.0e-4        # learning rate of the decision head (use 1e-4 for Laya, whose head is already trained)
  grad_checkpointing: true  # saves memory, ~30% slower
  device: auto           # auto picks cuda > mps > cpu
  export_laya: false     # laya backend only: also save a plain Laya checkpoint for laya-serve / ONNX
```
</details>

### Step 3: Train, compare, use

```bash
dmtune train   my_recipe.yaml                    # saves the fine-tuned model to `out`
dmtune compare my_recipe.yaml                    # writes report.md, report.json, reliability.png to `out`
dmtune predict my_recipe.yaml --state '{"body": "my card was declined"}'
dmtune predict my_recipe.yaml --image photo.jpg  # for vision models
```

`predict` asks the questions from the first row of your test file.

---

## Reading the report

`report.md` puts the original model and your fine-tuned model side by side, with the difference. A ✓ marks an improvement. What the main numbers mean:

| Metric | Plain meaning | Good direction |
|---|---|---|
| `accuracy` | How often the top answer is right. Compare it with the "majority-class" line at the top, the score you'd get by always guessing the most common answer. | higher |
| `ece` | How honest the confidence is: when the model says 90%, is it right about 90% of the time? | lower |
| `brier`, `nll` | Overall quality of the probabilities | lower |
| `coverage@5%risk` | Share of decisions you could automate if you only accept answers it's confident about, keeping errors at or below 5% | higher |
| `permutation_flip_rate` | How often an answer changes just because the options were listed in a different order | lower |
| `media_reliance` | How often answers change when the image is replaced with a blank gray one. Near 0 means the model is ignoring the image. | higher |
| `latency_p50_ms`, `peak_memory_gb` | Speed and memory use | lower |
| `<question>/macro_f1`, `/auroc`, `/qwk` | Per-question detail for choice, yes/no and score questions | higher |

`reliability.png` plots confidence against actual accuracy; a well-calibrated model sits on the diagonal.

**Use it as a quality gate** (e.g. in CI): `dmtune compare my_recipe.yaml --min-accuracy 0.9 --max-ece 0.1` fails if the model is below either bar.

**Compare against Jev**, for text only, with your TypeSafe API key in `TYPESAFE_API_KEY`. Add this to the recipe:

```yaml
compare:
  runners:
    - {backend: systemone, name: jev, url: https://api.typesafe.ai/v1/systemone, model: jev-latest, api_key_env: TYPESAFE_API_KEY}
```

Any server with the same API works the same way, including `laya-serve` and Kev.

---

## Hardware

| Machine | What works |
|---|---|
| Apple Silicon Mac | everything (tested on an M5 with 16 GB) |
| NVIDIA RTX, 4 GB (e.g. 3050 Laptop) | the default `lora` mode; use `micro_batch: 2` for text (estimated from memory use on a Mac; not yet run on a real 4 GB card) |
| NVIDIA RTX, 10 GB+ | also `mode: full` |
| CPU only | `mode: head` (slow but works) |

---

## How training works (for the curious)

dmtune follows Laya's published training recipe:
1. **Learning:** the model learns from your labels (cross-entropy). It also gets a reward for *honest* probabilities, using RLCD, a reinforcement-learning method with proper scoring rules.
2. **Option shuffling:** options are shuffled during training, so the answer doesn't depend on the order they're listed in.
3. **Calibration:** after training, each question type gets a "temperature" fitted on your validation set. This scales the confidence so it matches reality.

Image models use the same decision head as Laya, on top of ModernVBERT, a compact image+text encoder.

---

## More

- **Text benchmark demo:** `examples/typed_decisions/` fine-tunes Laya on Laya's own benchmark. Accuracy went 42% → 70.5% with 300 training examples.
- **Tests:** `pip install -e ".[test]" && pytest -q tests` (they run offline).
- **Not built yet:**
  - a stronger zero-shot image model, for when you have very few labels;
  - Kev training, for more than 20 options;
  - motion-aware video;
  - non-speech audio (e.g. alarms, engines);
  - ONNX export for image models;
  - multi-GPU training;
  - a web UI.
- **License:** Apache-2.0.
