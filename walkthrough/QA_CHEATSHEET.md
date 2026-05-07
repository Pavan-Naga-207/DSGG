# QA_CHEATSHEET.md

20 likely Q&A items. Each: the question, a 2-3 sentence answer in your own voice, and a "fallback if cornered" line you can use to retreat to safe ground.

Read this twice tonight. Skim once tomorrow morning.

---

## Architecture and design choices

### Q1. "Why use ViTDet's synthetic pyramid instead of a native ViT detector like DETR?"

**Answer.** A plain ViT outputs a single feature map at stride 16, but Faster R-CNN's RPN and RoI Align expect a four-level pyramid. ViTDet showed that you can synthesize that pyramid from a single ViT scale with simple up- and down-sampling convolutions, which let us keep the entire Faster R-CNN shell unchanged. We didn't go DETR because that would have required rewriting the relation head's interface to STTran, which was out of scope.

**Fallback.** "We wanted to isolate the *backbone* as the only changing variable across phases. ViTDet was the smallest possible bridge to do that."

---

### Q2. "Why keep the Faster R-CNN detector at all? Why not go end-to-end?"

**Answer.** STTran's whole pipeline is built around a detector-first flow: train detector on Stage 1, freeze it, train graph head on Stage 2. Replacing Faster R-CNN would have forced us to redesign Stage 2 around set-prediction heads, which is a separate research project. Our contribution is a clean *backbone* ablation, not a detector ablation.

**Fallback.** "We mention DETR as future work in the report - it's the natural next step after this."

---

### Q3. "Why freeze DINOv2 instead of fine-tuning it?"

**Answer.** Two reasons. First, AG is a small corpus by foundation-model standards - fine-tuning 86M backbone params on a few hundred thousand frames risks degrading the features. Second, freezing tests a stronger version of the open-world hypothesis: if frozen general-purpose features beat task-specific fine-tuning, that's a much more interesting result for the field.

**Fallback.** "Empirically the frozen variant beat the fine-tuned ViT on every graph metric, so the result confirms the hypothesis."

---

### Q4. "Why DINOv2 specifically? Why not CLIP or MAE?"

**Answer.** We piloted MAE as a side branch and it did *not* beat the non-MAE detector on detector mAP, so we kept it out of the mainline. We didn't pursue CLIP because its features are aligned to language and we wanted dense, spatially-coherent visual features for box-level reasoning, which is exactly what DINOv2's patch-level self-distillation produces. DINOv3 + Gram anchoring is the next frozen-backbone we'd try.

**Fallback.** "DINOv2 was a deliberate choice for spatial granularity, not arbitrary."

---

### Q5. "Why ViT-B and not ViT-L or ViT-g for DINOv2?"

**Answer.** ViT-B/14 was the best size that fit comfortably on a single H100 at 1024 input with batch size 12. ViT-L/14 would have forced us to either drop batch size or input resolution, which would have hurt detector quality more than the bigger backbone would help. We mention scaling up as future work.

**Fallback.** "Compute budget; the trend is monotonic so larger DINOv2 should help."

---

## The three bugs

### Q6. "Why exactly did BF16 break the loss?"

**Answer.** BF16 has only 7 mantissa bits. Cross-entropy evaluates `-log p` for predicted probabilities, and the log of a small number underflows fast in low precision. The loss returned NaN, which propagated through the gradient and corrupted the optimizer state. Fix: cast logits to FP32 before softmax, sigmoid, and the loss. Backbone forward stays BF16 for speed, loss math runs FP32 for stability.

**Fallback.** "Standard mixed-precision-training pattern: keep the heavy compute in BF16, keep the numerically sensitive ops in FP32."

---

### Q7. "How does GroupNorm differ from BatchNorm? Why did 209 keys disappear?"

**Answer.** BatchNorm normalizes each channel using statistics over the batch dimension and stores `running_mean` and `running_var` buffers; GroupNorm normalizes within a group of channels per sample and stores nothing. When training used GroupNorm but eval defaulted to BatchNorm, every BN running-buffer pair across the backbone-neck-RPN stack - 209 of them - was silently dropped at `load_state_dict(strict=False)` time, leaving random-init BN buffers at eval. mAP went to zero. Fix: standardize on GroupNorm everywhere with 32 groups.

**Fallback.** "It was a config-drift bug, not a model-quality bug; once we matched train and eval norm modes, mAP came right back."

---

### Q8. "How did you find the Mixed-Path Bug?"

**Answer.** PredCLS looked good but SGCLS was 0.013 R@20, essentially zero. That asymmetry was the clue - PredCLS and SGCLS share the same object features, so they shouldn't behave that differently. We traced the GT-box branch in `object_detector.py` and found that the legacy ResNet feature path was still being called there, while only SGDET had been rewired to the new ViT path. After the fix, SGCLS R@20 jumped from 0.013 to 0.297 on the *same* checkpoint.

**Fallback.** "When two evaluation modes that should agree are wildly different, the bug is upstream of both - in our case, in the feature extractor."

---

### Q9. "What's the methodological lesson from the Mixed-Path Bug?"

**Answer.** Any time you change the backbone in a multi-mode evaluation pipeline, you have to make the mode-consistency invariant explicit and assert it at runtime. We now have an environment flag `PHASE3_ASSERT_DINOV2_PATH=1` that asserts the DINOv2 path is actually being used, and the bridge prints a per-mode log line the first time each mode runs. Any future regression of this kind would surface in epoch 0.

**Fallback.** "Treat assumptions as preconditions, not comments."

---

## Datasets and metrics

### Q10. "What's the difference between PredCLS, SGCLS, and SGDET?"

**Answer.** PredCLS gets ground-truth boxes *and* ground-truth object classes - it only predicts the relation. SGCLS gets ground-truth boxes but the model has to classify the objects, then predicts the relation. SGDET gives nothing - the model detects boxes, classifies them, and predicts relations. Difficulty goes PredCLS < SGCLS < SGDET. Our headline win is on SGCLS because that mode isolates per-region embedding quality - exactly what DINOv2 is best at.

**Fallback.** "Three modes, increasing difficulty. We win biggest on the embedding-bottleneck mode, which lines up with our hypothesis."

---

### Q11. "What's With vs Semi vs No constraint?"

**Answer.** With constraint allows at most one predicate per (subject, object) pair - strictest. Semi allows top-k per pair. No constraint ranks every (pair, predicate) independently - loosest. The official STTran headline anchor is *With-constraint R@20*. We win there too, but we report all 27 cells (3 modes x 3 constraints x 3 recalls) in the paper for transparency.

**Fallback.** "With is the strictest and most paper-comparable. We win across all three regimes."

---

### Q12. "Why recall and not precision or mAP for the relations?"

**Answer.** Scene-graph annotations are *incomplete* - the dataset cannot label every relation in every frame. A precision-based metric would penalize the model for predicting a real relation that wasn't annotated, which would be unfair. Recall@K side-steps that by only asking what fraction of ground-truth triplets appear in the top-K predictions. It's the standard for the field.

**Fallback.** "Annotation incompleteness; precision would be misleading."

---

### Q13. "How does Action Genome differ from Visual Genome?"

**Answer.** Visual Genome is *static images* with crowd-sourced scene graph annotations. Action Genome is *videos* (built on Charades) with per-frame scene graphs that focus specifically on human-object interactions. Action Genome adds temporal structure and predicate categories - attention, spatial, contact - that make sense for action understanding. We use AG because DSGG is a video task.

**Fallback.** "VG is images, AG is videos with HOI focus."

---

## Results

### Q14. "Why is the detector mAP gap small but the SGCLS gap large?"

**Answer.** Because they measure different things. Detector mAP measures how well the box localizer finds objects. SGCLS bypasses the localizer (boxes are given) and measures per-region embedding quality - how well the pooled feature carries object semantics. Our backbone change barely affects the localizer, but it strongly affects per-box embeddings - which is exactly what DINOv2's frozen features are best at. So you'd expect the SGCLS gap to dwarf the mAP gap, and that's what we see.

**Fallback.** "SGCLS isolates embedding quality; mAP measures localization. We won the embedding lever, not the localization lever."

---

### Q15. "Are you sure 5.4 points on SGCLS R@20 is significant? Could it be noise?"

**Answer.** A single-cell win could be noise, but we sweep all 9 SGCLS cells - three constraints, three recall thresholds - and DINOv2 wins every single one, with margins ranging from +5 to +14. We also win all 9 SGDET cells. A 18-for-18 sweep across two modes is not noise - it's a robust ordering.

**Fallback.** "It's the *pattern* across all 18 SGCLS+SGDET cells that's the evidence, not any one number."

---

### Q16. "Why does PredCLS not win? Doesn't that hurt your story?"

**Answer.** PredCLS gives the model ground-truth boxes *and* ground-truth object classes. The relation head is the only thing being tested, and we did not modify the relation head. So PredCLS *should* be roughly the same as the base paper. If it had jumped, that would mean we accidentally helped the relation head, which would be suspicious. The clean tie is actually a positive sanity check - it confirms our wins live in the backbone, not in some other component.

**Fallback.** "PredCLS being a tie is expected and is a sanity check, not a weakness."

---

### Q17. "How long did training take?"

**Answer.** Stage 1 detector pretraining was on the order of a few days per backbone on a single H100. Stage 2 graph training was shorter - around a day - because the detector was frozen and only the STTran head trained. Total wall-clock was dominated by Stage 1 sweeps and detector-checkpoint-ranking sweeps, not by Stage 2.

**Fallback.** "Most of the wall clock was Stage-1 detector pretraining. Stage 2 was relatively fast."

---

## Future work and limitations

### Q18. "What is Gram anchoring and why would DINOv3 help?"

**Answer.** DINOv3 (Meta, 2024-2025) extends DINOv2 with a regularizer called Gram anchoring that constrains the patch-feature Gram matrix during training, preserving the geometric and textural relationships between patches. The practical effect is more discriminative per-patch features without sacrificing semantic transfer. That's exactly the property our SGCLS metric rewards - per-region discriminability - so DINOv3 should be a direct lever for closing any residual gap and pushing SGCLS further.

**Fallback.** "It's a regularizer that improves per-patch discriminability, which is the property our headline metric measures."

---

### Q19. "What's the single most important limitation of the current result?"

**Answer.** Single-seed training. We didn't have the compute to repeat each phase across multiple seeds, so absolute numbers may shift by 1-2 recall points across seeds, although the *ordering* between phases is robust in the small repeats we did run. The other notable limitation is that our Phase 2 ViT was trained from scratch on AG only - a ViT initialized from ImageNet pretrained weights would probably look better, so our Phase 2 numbers are a lower bound on what fine-tuned ViT can do.

**Fallback.** "Single-seed training and Phase 2 lacking ImageNet-pretrained init are the two most honest limitations."

---

### Q20. "If you had two more weeks, what would you do?"

**Answer.** Three things in priority order. First, try DINOv3 + Gram anchoring as the next frozen backbone. Second, swap Faster R-CNN for a DETR-style set-prediction detector to remove the synthetic-pyramid bridge entirely - that would let the ViT operate at its native single scale. Third, extend STTran's temporal window beyond 2 frames to a clip-level encoder for long-horizon predicates like *wearing* and *holding*. The first one is the cheapest to try and most likely to give a clean numerical win.

**Fallback.** "DINOv3 first, then DETR-style detector, then a longer temporal window."

---

## "Save the day" answers if you go blank

If you completely freeze on a question, retreat to one of these:

> "Great question. The way I would think about it is that our entire project is structured as a clean backbone ablation - everything to the right of the backbone is held fixed across phases. So the answer almost always comes down to: what changes when we swap the backbone? In this case..."

(then improvise from your last clear thought)

> "I want to be careful not to over-claim. Our specific finding is that *frozen* DINOv2 features beat both ResNet-101 and a from-scratch ViT on the SGCLS and SGDET headline metrics, while detector mAP stays roughly the same. The implication is that the win comes from per-region embedding quality, not from the localizer."

(this is a safe pull-back to the headline result)

> "I don't want to guess. We didn't run that ablation, but my best hypothesis would be..." (then commit to one short hypothesis).

This is much better than making up a number. Instructors respect "I don't know but here's how I'd find out."

---

## The single highest-leverage thing to memorize

If only one full sentence comes out of your mouth tomorrow, make it this one:

> *"The detector mAP gap between our pipeline and the base paper is small, but the SGCLS gap is large, and that gap is exactly what you would expect if the backbone embedding is the lever - because SGCLS gives the model ground-truth boxes and isolates per-region embedding quality, which is what DINOv2's frozen features carry."*

Memorize that. It is the answer to the most important possible question, the one any sharp instructor will ask after seeing your tables.

Good luck.
