# Decision models: Jev, Laya and the System-1 class (research notes, Oct 2026)

## 1. What a decision model is

A decision model (also called a "System-1" model) is a non-autoregressive network. It answers a **typed question** about a **state** in one forward pass and returns calibrated probabilities instead of generated text.

All of these models share three question primitives:

| Type | Returns | Example |
|---|---|---|
| `choice` | one option out of N, plus a probability for each option | route a ticket to billing / tech / sales |
| `score` | an ordinal level (Jev allows 2 to 10 levels), returned as the expected level plus the distribution over levels | urgency 0–3 |
| `noul` | P(true) for a statement | "the customer threatens to cancel" |

Wire format (Jev's `/v1/systemone`; Laya, Kev and openjev mirror it):

```json
{"state": {...}, "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}}}
→ {"answers": {"q": {"choice": "a", "probabilities": {...}, "confidence": 0.9}}}
```

**Why use one instead of an LLM:**
- latency of 30–500 ms;
- a cost about 100× lower;
- answers are schema-valid by construction, so no format hallucinations.

**The catch:**
- "a confidently wrong choice from the options it was given is still wrong";
- the probabilities only mean something once they are calibrated on *your* data.

## 2. The models

### Jev (TypeSafe AI, launched about 15 Sep 2026; closed)

**Access:**
- API only: `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest` (e.g. `jev-1.13.0`).
- Runware (`typesafe:jev@latest`) and OpenRouter (alpha) also serve it.

**Limits:**
- Up to **255 choice options** and 2–10 score levels.
- 64k tokens per request.

**Price and speed:**
- $0.042 per 1M input tokens; output is free.
- 70–500 ms per request.

**Training:**
- "RLCD: Reinforcement Learning for Calibrated Decisions" and a "parallel sampler".
- No mechanism is published.

**Not fine-tunable:** "the same weights serve every account."

**Text only:** "Images, audio, and video are not supported (yet)."

**Weak spots TypeSafe documents** ("jaggedness"):
- literal reading of instructions;
- math;
- dates;
- multi-hop reasoning;
- large amounts of irrelevant state;
- adversarial content;
- contradictory instructions;
- **choice option order**.

**Security:** Check Point broke Jev with *fabricated evidence* placed inside the state, in 25 of 27 runs at about $0.50 per break. Structured input and anti-injection instructions barely helped. Their takeaway: "Do not mistake an output format for a security boundary."

### Laya (Convai Innovations, 18 Sep 2026; Apache-2.0)

**Install and weights:** `pip install laya`, Hugging Face `convaiinnovations/laya`.

**Checkpoints:**

| Checkpoint | Encoder | Size | Context |
|---|---|---|---|
| English | ModernBERT-large | 421M | 512 |
| `multilingual` | mmBERT-base | 322M | 1024; 100+ languages |
| `typed-decisions` | ModernBERT-large | 421M | 1024 |

**Architecture:**
- The input sequence is `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`.
- Each option is scored at its own `[MASK]` marker, using 2 transformer layers plus an MLP. The scores are then softmaxed over that question's options.

**Training (RLCD):**
- REINFORCE over Gaussian-perturbed logits with 4 samples. The noise σ falls from 0.4 to 0.1.
- Reward: log score + 0.75 × spherical score − RPS (RPS applies to score questions only).
- Plus cross-entropy.
- Afterwards, one temperature is fitted per bucket of (question type, number of options).

**Speed:** 33–40 ms on a T4, or 7.2 ms per question when batched.

**Accuracy and calibration (published):**

| Benchmark | Laya | Jev |
|---|---|---|
| typed-decisions, zero-shot | 0.362 (below the 0.461 majority baseline) | — |
| typed-decisions, after fine-tuning | 0.766 | 0.727 |
| ECE, raw | 0.466 | — |
| ECE, after temperature fitting | 0.081 | 0.246 |
| AG News | 0.950 | 0.910 |
| Banking77 (77 labels) | 0.425 | 0.870 |

**Honest limits:**
- weak above about 20 options;
- score questions are its weakest type;
- noul answers sometimes follow the label text instead of the state;
- ships over-confident;
- **needs fine-tuning on your domain**.

### Kev (Jared Palmer; Apache-2.0)

**Checkpoints:**
- Qwen3.5-0.8B, 4B and 9B bases, with LoRA r16/α32 on the attention, MLP and DeltaNet layers.
- A 27B model, fully fine-tuned.

**Pointer head:** each option's `</opt>` hidden state is scored against the `<decide>` token, then softmaxed.

**Training:**
- About 12.5k records, with option permutation and "none of the above" augmentation.
- Ships fitted temperatures of 1.3–2.4.

**Results:** Kev-27B scores 0.851 vs Jev's 0.857, and its Brier score is 0.225 vs Jev's 0.211.

**API:** Jev-compatible, plus `/v1/systemone/permute` for testing option order.

**Fine-tuning:** `kev.train --init_from jaredpalmer/kev-4b`.

### Community and research projects
- **SemIf (formerly OpenJev) and AnyJev (Nokia):** read option-letter logits from any LLM, with no training.
  - AnyJev rotates the option order and divides out the label prior; the order-flip rate falls from 0.23 to 0.07.
  - An optional head fitted on 100–300 labels raises auto-decidable traffic at ≤5% error from 7.7% to 52%.
- **NanoJev:** Qwen3-0.6B with parallel heads, tested on games.
- **jevlike:** an option-attention trainer that can also score image patches.
- **Multimodal:**
  - **Jev-Omni:** Gemma 4 12B; text, image, audio and 16 video frames.
  - **openjev:** DiffusionGemma 26B; images.
  - **JEV-27B-VL**, **djev**: also handle images.
  - All of these need 24–80 GB GPUs.
- **JevAny (fine-tuning framework):**
  - 26 LLM backbones, with pointer, direct-token or choice-token readouts.
  - LoRA SFT plus RLCR.
  - Covers text, image and video.
  - **No Laya/ModernBERT support and no Apple Silicon support.**

## 3. Gap this framework fills (dmtune)

Nothing fine-tunes **Laya-class encoders** across **image, video and audio** on **4 GB laptop GPUs or Apple Silicon** and then produces an **honest before/after report**. That combination is dmtune's niche.

| Need | dmtune choice | Why |
|---|---|---|
| Text | Laya + LoRA | Most-used open decision model; 421M parameters (322M multilingual) |
| Image | ModernVBERT (250M, MIT) + Laya's option-marker head | Same ModernBERT family; already pretrained to fuse images and text; runs on CPU |
| Video | storyboard: N frames tiled into one image | Fixed token cost; no video-specific model needed |
| Audio (speech) | Whisper-base transcript → Laya | 74M parameters, CPU-friendly |
| Jev | evaluation baseline via `/v1/systemone` | Closed and can't be fine-tuned |

## 4. Metrics that matter for decision models (and what `dmtune compare` reports)

- **Accuracy:**
  - overall accuracy;
  - choice accuracy, macro-F1 and balanced accuracy;
  - noul AUROC and AUPRC;
  - score MAE, QWK and RPS.
- **Calibration:** ECE (15 bins), Brier, NLL and a reliability diagram. Calibration is the whole point of probabilistic answers.
- **Selective risk:**
  - AURC;
  - selective accuracy at 50% and 80% coverage;
  - **coverage at ≤5% risk**: the share of traffic you can automate.
- **Robustness:**
  - **permutation flip rate** (Jev's documented weak spot);
  - **media reliance**: blank the image and see whether answers change. This checks the model actually uses the media; the idea comes from JevJudge's blank-media controls.
  - Prompt injection: tag an adversarial slice and read its per-tag metrics.
- **Cost:** latency p50/p95, peak memory, training wall time and trainable parameters.

## Sources
- **Jev and decision models in general:**
  - [Runware: Jev, Laya and decision models](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
  - TypeSafe docs: [API](https://docs.typesafe.ai/api.md), [models](https://docs.typesafe.ai/models.md), [jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
  - [TypeSafe launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  - [Check Point: prompt injection against Jev](https://blog.checkpoint.com/ai-security/jev-is-not-a-language-model-but-it-breaks-like-one-prompt-injection-against-a-typed-decision-model/)
  - [OpenRouter: What is Jev](https://openrouter.ai/blog/insights/what-is-jev/)
- **Laya:**
  - [GitHub](https://github.com/NandhaKishorM/laya), [Hugging Face](https://huggingface.co/convaiinnovations/laya), [site](https://laya.convaiinnovations.com/)
  - [Flowtivity benchmark](https://flowtivity.ai/blog/laya-open-source-jev-alternative/)
  - [Layer3 guide](https://www.layer3labs.io/guides/system-one-decision-models)
- **Other open decision models:**
  - [Kev](https://github.com/jaredpalmer/kev), [SemIf](https://github.com/TheoLeeCJ/SemIf), [AnyJev](https://github.com/nokia-applied-research/AnyJev)
  - [NanoJev](https://github.com/TianyuCodings/NanoJev), [jevlike](https://github.com/vinnylarouge/jevlike)
  - [JevAny](https://github.com/SimpleJev/JevAny), [openjev](https://github.com/razorback16/openjev)
  - [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni), [JevJudge-Public](https://huggingface.co/datasets/HuanxinSheng/JevJudge-Public)
  - [Pinggy roundup](https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/)
- **Backbones:** [ModernVBERT](https://huggingface.co/ModernVBERT/modernvbert) ([paper](https://arxiv.org/abs/2510.01149)), [Whisper](https://huggingface.co/openai/whisper-base)
- **Calibration:** [RLCR: Beyond Binary Rewards](https://arxiv.org/abs/2507.16806), [PriDe (option-order bias)](https://arxiv.org/abs/2309.03882)

Not independently verified: RLCD internals, the exact Jev rate limits, the JevAny VLM backbone names, and Jev-Omni's self-reported numbers.
