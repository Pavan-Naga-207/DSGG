# PRESENTATION_FLOW.md

12-slide outline mapped to phases. Each slide has:

- **Title** (the slide title)
- **On the slide** (what should appear visually)
- **Say out loud** (the script - rehearse this verbatim)

Total target time: **10-12 minutes** for the talk + 3-5 minutes Q&A.

---

## Slide 1 - Title + Authors + GitHub

**On the slide**

- Title: *Modernizing Spatio-Temporal Scene Graph Generation: From CNN Backbones to Frozen Foundation Models on Action Genome*
- Names + SJSU CMPE 297 affiliation
- Big QR code linking to `https://github.com/Pavan-Naga-207/DSGG.git`

**Say out loud (~10s)**

> "We modernized Dynamic Scene Graph Generation by replacing the CNN backbone in the STTran framework first with a Vision Transformer and then with a frozen DINOv2 foundation model. Final result: we beat the original STTran paper on the headline SGCLS R@20 metric, 52.88 vs 47.5."

Open strong with the headline.

---

## Slide 2 - The Problem: Dynamic Scene Graph Generation

**On the slide**

- One image of an AG frame with overlaid scene graph: `person -- holding -- cup`, `person -- looking_at -- cup`
- Bullet: *Predict, for every frame of a video, a directed graph of (subject, predicate, object) triplets*
- Mini bullet: *Action Genome, 26 relationship classes, 167K training frames after filtering*

**Say out loud (~25s)**

> "Dynamic Scene Graph Generation, or DSGG, takes a video and produces a per-frame relationship graph. Nodes are detected objects, edges are relations like *person holding cup* or *person looking at cup*. We use the Action Genome benchmark, which is built on top of the Charades video dataset and has 26 relationship classes split across attention, spatial, and contact families."

Anchor the audience on what we're predicting.

---

## Slide 3 - Why Modernize? ResNet-101's Limits

**On the slide**

- Left: STTran's original architecture diagram with **ResNet-101** highlighted
- Right: bullets:
  - *Local convolutional receptive fields struggle with long-range scene context*
  - *Expensive end-to-end fine-tuning required for any backbone change*
  - *Foundation models (DINOv2) and ViTDet have made plain-ViT detectors viable*

**Say out loud (~30s)**

> "The dominant DSGG framework, STTran, uses Faster R-CNN with a ResNet-101 backbone. ResNet's local receptive fields struggle with the long-range spatial context that DSGG needs - the person on the left interacting with the cup on the right. We hypothesized that swapping in a Vision Transformer, and eventually a frozen DINOv2 foundation backbone, would give us stronger features without retraining the world."

This is the motivation. Don't oversell - the punchline comes later.

---

## Slide 4 - Phase 1: Reproduce the Baseline

**On the slide**

- One small parity table:
  | Metric | Paper | Ours |
  |---|---|---|
  | mAP@0.5 | 24.6 | 24.6 |
  | PredCLS R@20 | 71.8 | 71.8 |
  | SGCLS R@20 | 47.5 | 47.5 |
- Bullet: *Validates the harness; any later deviation is the backbone*

**Say out loud (~20s)**

> "Phase 1 was reproducing the original STTran ResNet-101 pipeline. We hit the published numbers exactly. That gave us a trustworthy harness so any later deviation could be cleanly attributed to the backbone."

Quick, no drama. This slide buys credibility for everything after.

---

## Slide 5 - Phase 2 Plan: Drop in a Plain ViT

**On the slide**

- Architecture diagram:
  - *AG frame -> ViT-B/16 -> ViTDet synthetic pyramid (p2/p3/p4/p5) -> Faster R-CNN heads -> STTran*
- Bullet: *Same Faster R-CNN shell, same STTran heads. Only the visual backbone changes.*
- Footnote: *Bridge code: `phase2_vitdet/simple_vitdet_fpn.py`*

**Say out loud (~25s)**

> "Phase 2 swapped the ResNet for a plain ViT-B/16. Plain ViTs only output a single feature map at stride 16, so we wrapped the ViT in a ViTDet-style synthetic pyramid - simple up- and down-sampling convolutions to produce the four pyramid levels Faster R-CNN expects. Critical design choice: keep RPN, RoI Align, the detector heads, and STTran exactly the same. Only swap the backbone."

Set up the bug slides.

---

## Slide 6 - Phase 2 Bug 1: BF16 Poison Pill

**On the slide**

- Title with subtle red accent: *Bug 1 - BF16 underflows inside CE/BCE*
- Equation:
  $$\hat y = \mathrm{softmax}(\mathrm{cast}_{\text{FP32}}(z)),\ \mathcal{L}_{\text{obj}} = -\sum_c y_c \log \hat y_c$$
- Bullet: *Backbone forward stays BF16 (fast). Loss math runs FP32 (stable).*
- Footnote: *Fix: `STTran/train.py:997-999`*

**Say out loud (~30s)**

> "First bug. On H100 we run mixed precision in BF16. BF16 has 8-bit exponent but only 7-bit mantissa - low precision. The negative-log term inside cross-entropy underflowed and the loss returned NaN. The whole batch was poisoned. Fix: explicitly cast logits to FP32 before softmax, sigmoid, and the loss. Backbone forward stays BF16 for speed; loss math runs FP32 for stability. Three lines of code, one critical bug fixed."

Use the word "poisoned" - it's memorable.

---

## Slide 7 - Phase 2 Bug 2: GroupNorm Consistency

**On the slide**

- Title: *Bug 2 - 209 detector keys silently dropped at eval*
- Equation: $\mathrm{GN}_{32}(x) = \gamma \odot \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \varepsilon}} + \beta,\ g=32$
- Bullet: *BatchNorm running buffers vs GroupNorm: train/eval mode mismatch dropped 209 keys -> mAP = 0.*
- Footnote: *Fix: `STTran/lib/object_detector.py:32-50, 187-206`*

**Say out loud (~25s)**

> "Second bug. Faster R-CNN ships with BatchNorm. ViT-based detectors typically use GroupNorm. When training with GroupNorm but evaluating with the env var unset, we silently dropped 209 BatchNorm running-mean and running-variance buffers at checkpoint load. Detector mAP went to zero, even though training looked fine. Fix: standardize on GroupNorm with 32 groups everywhere, train and eval. The launchers now default to it."

Specific number - "209 keys" - is memorable and hard to question.

---

## Slide 8 - Phase 2 Bug 3: The Mixed-Path Bug

**On the slide**

- Title: *Bug 3 - PredCLS/SGCLS using ResNet, SGDET using ViT*
- Diagram showing two divergent feature paths from one frame, then converging into the fix: *all modes call `_extract_base_features()`*
- Equation: $\forall m \in \{\text{PredCLS}, \text{SGCLS}, \text{SGDET}\},\ f_\theta^{(m)} \equiv f_\theta$
- Bullet: *SGCLS R@20: 0.013 -> 0.297 on the same checkpoint*

**Say out loud (~40s)**

> "Third bug, and the most important one. PredCLS and SGCLS were silently using the legacy ResNet feature path while SGDET used the new ViT path. They were not even the same model. Cross-mode comparisons were meaningless - that is why early Phase 2 SGCLS was 0.013 R@20, essentially zero. Fix: rewire all three modes through one feature extractor function. SGCLS jumped from 0.013 to 0.297 immediately on the same checkpoint. The methodological lesson - which we now treat as an invariant - is that all evaluation modes must share one backbone path. We even added runtime assertions in Phase 3 to enforce it."

This is the slide that sells the story. Take your time on it.

---

## Slide 9 - Phase 3: Frozen DINOv2 Foundation Backbone

**On the slide**

- Architecture diagram:
  - *Frame -> DINOv2 ViT-B/14 [FROZEN, snowflake icon] -> DINOv2Bridge -> Faster R-CNN heads -> STTran*
- Bullets:
  - *Backbone params frozen ($\nabla_\theta f_\theta \equiv 0$), only train bridge + heads*
  - *Single source of truth for preprocessing prevents Mixed-Path-style bugs*
  - *Path assertions + per-mode runtime logging*
- Footnote: *`phase3_dinov2/dinov2_bridge.py`, `preprocess.py`*

**Say out loud (~35s)**

> "Phase 3 swaps the trainable ViT for a frozen DINOv2 ViT-B/14. We made it a parallel branch so we couldn't break Phase 2's corrected scaffolding. The backbone is frozen - all 86 million parameters - and the forward runs inside torch.no_grad. We only train the synthetic pyramid, RPN, detector heads, and STTran graph head. Three anti-regression safeguards from Phase 2 lessons: a single preprocessing config shared by every loader, runtime assertions that the DINOv2 path is being used, and per-mode logging that prints backbone, bridge class, and feature shape the first time each mode runs."

Mention "snowflake" - frozen is a recurring concept worth a visual.

---

## Slide 10 - Results: The Headline Tables

**On the slide**

- Detector mAP table:
  | Model | mAP@0.5 |
  |---|---|
  | Base STTran | 24.60 |
  | Phase 2 ViT | 23.19 |
  | **Phase 3 DINOv2 (ours)** | **24.25** |
- Below: SGCLS / SGDET / PredCLS R@20 With-constraint comparison row, headline cells bold

**Say out loud (~30s)**

> "Headline results. Detector mAP is essentially a three-way tie - 24.6 base, 23.19 ViT, 24.25 DINOv2. SGCLS R@20: 47.5 base, 30.75 ViT, 52.88 DINOv2 - we beat the original paper by 5.4 points. SGDET R@20: 34.1 base, 23.07 ViT, 40.41 DINOv2 - we beat the paper by 6.3 points. PredCLS is essentially a tie with the base paper, which is expected because we did not modify the relation head."

Pause after the headline numbers. Let them land.

---

## Slide 11 - Key Insight: The mAP Gap is Small, the SGCLS Gap is Huge

**On the slide**

- Two-bar chart side by side:
  - Left: detector mAP gap = +0.06 between best and ours
  - Right: SGCLS R@20 gap = +5.4 over base, +22.1 over ViT
- One-line caption: *Same detector quality, much better per-region embedding -> SGCLS isolates exactly that.*

**Say out loud (~30s)**

> "The most interesting insight in the project: the detector mAP gap is small but the SGCLS gap is huge. SGCLS gives the model ground-truth boxes - so it bypasses the localizer entirely. What it measures is per-region embedding quality. DINOv2's frozen patch features carry stronger object semantics than a from-scratch ViT or a fine-tuned ResNet, and SGCLS isolates that exactly. The win is the embedding, not the detector."

This is the question-magnet slide. Practice this one out loud at least three times.

---

## Slide 12 - Conclusion + Future Work

**On the slide**

- Bullets:
  - *We re-built STTran's pipeline correctly, then swapped in DINOv2 frozen features*
  - *Beat base STTran on SGCLS R@20 by +5.4 and SGDET R@20 by +6.3*
  - *Future: DINOv3 + Gram anchoring, DETR-style detector swap, longer temporal windows*
- Bottom: GitHub QR code + repo URL

**Say out loud (~25s)**

> "To summarize: we modernized STTran across three phases. Phase 1 reproduced the baseline. Phase 2 corrected three serious bugs in ViT integration. Phase 3 dropped in a frozen DINOv2 backbone and beat the original paper on the hardest metrics. The clearest next step is DINOv3 with Gram anchoring, which is Meta's regularizer for preserving fine-grained patch geometry - it is the most direct candidate to close the residual PredCLS gap and probably push SGCLS and SGDET further. Code is on the GitHub repo on screen. Happy to take questions."

End on the next-step lever. It signals you understand where the field is going.

---

## Timing budget

| Slide | Topic | Approx duration |
|---|---|---|
| 1 | Title + headline | 10s |
| 2 | Problem | 25s |
| 3 | Motivation | 30s |
| 4 | Phase 1 baseline | 20s |
| 5 | Phase 2 plan | 25s |
| 6 | Bug 1 BF16 | 30s |
| 7 | Bug 2 GroupNorm | 25s |
| 8 | Bug 3 Mixed-Path | 40s |
| 9 | Phase 3 DINOv2 | 35s |
| 10 | Results tables | 30s |
| 11 | Key insight (mAP vs SGCLS) | 30s |
| 12 | Conclusion + future | 25s |
| **Total talk** | | **~5.5 min** |
| Buffer + transitions | | +2-3 min |
| Q&A | | 3-5 min |

If they cut you off, the slides you can drop in priority order:

1. Slide 4 (Phase 1) - just say "we hit parity, moved on"
2. Slide 7 (Bug 2 GroupNorm) - mention in one line on Slide 8
3. Slide 11 (key insight) - **DO NOT DROP THIS** - merge with Slide 10 if needed

Slides 1, 2, 8, 9, 10, 11, 12 are non-negotiable.

---

## One thing to do before the talk

Read each slide's "Say out loud" verbatim once, in front of a mirror or to your phone. The first time you hear yourself say "BF16 poison pill" out loud, you will realize you need it to be smoother. Do that pass tonight, not tomorrow morning.

Good luck.
