# Mitigation Summary

This document summarizes the most practical mitigation strategies across the entire system, with emphasis on student-scale execution and limited compute.

## 1. Data Mitigation Priorities

### What To Do First

* Build a small manually reviewed validation subset.
* Split by `product_id` or `listing_id`, not by random rows only.
* Track score bucket, category, language type, and image quality in the metadata.
* Create a small normalization dictionary for frequent Vietnamese-English slang and abbreviations.

### Why This Matters

* Better architecture will not rescue a weak evaluation setup.
* The largest hidden risk in this project is learning from noisy labels and leaking repeated samples across splits.

## 2. Model Mitigation Priorities

### What To Do First

* Start with `convnext_tiny` and `xlm-roberta-base`.
* Freeze most backbone layers for the first stage of training.
* Use early stopping on validation MAE.
* Use Huber or SmoothL1 if label noise is visibly high.

### Why This Matters

* These choices reduce the chance of overfitting and catastrophic forgetting while staying feasible on one GPU.

## 3. Fusion Mitigation Priorities

### What To Do First

* Use concatenation plus a small MLP as the default fusion model.
* Compare against image-only and text-only baselines under identical splits.
* Add modality dropout or simple branch masking during validation.
* Track disagreement cases explicitly.

### Why This Matters

* A multi-modal system is only worth the added complexity if both branches contribute measurable value.

## 4. Explainability Mitigation Priorities

### What To Do First

* Treat Grad-CAM and attention as diagnostic tools, not proof of reasoning.
* Generate heavy explanations offline or on demand.
* Review explanation outputs on a fixed sample bank.
* Keep slice-based error analysis next to qualitative explanation examples.

### Why This Matters

* Explanations are easy to over-trust, especially in academic demos.

## 5. AI Agent Mitigation Priorities

### What To Do First

* Use deterministic templates before any LLM-backed generation.
* Pass structured named fields, not free-form prompts.
* Include confidence and modality-disagreement information in the explanation input schema.
* Add basic explanation unit tests against fixed prediction cases.

### Why This Matters

* The explanation layer should not be allowed to undermine trust in an otherwise reasonable model.

## 6. System Mitigation Priorities

### What To Do First

* Keep API prediction and explanation modes separate.
* Pin dependency versions and save config snapshots with checkpoints.
* Reuse the same preprocessing code in training and serving.
* Add input validation for missing or corrupt images and empty text.

### Why This Matters

* Many deployment failures come from avoidable engineering drift rather than model quality alone.

## 7. Recommended Low-Cost Validation Checklist

Before calling the system stable, verify all of the following:

* image-only, text-only, and fusion baselines are all trained and compared
* train-validation leakage checks are complete
* validation metrics are reported by category, score bucket, and language slice
* at least one explanation review set has been manually inspected
* the agent cannot generate unsupported claims from fixed test inputs
* API preprocessing matches training preprocessing exactly

## 8. Optional Higher-Effort Improvements

These are useful, but they should come after the baseline system is reliable.

* layer-wise learning-rate decay for XLM-R and ConvNeXt
* learned gating between image and text branches
* cross-attention fusion after stable concatenation baselines
* active learning for ambiguous or conflicting samples
* calibration analysis and confidence-aware output schemas
* asynchronous explanation generation pipeline

## 9. Best Return On Time

If you can only invest effort in a few upgrades, prioritize them in this order:

1. better splits and label review
2. unimodal versus fusion ablations
3. stable fine-tuning and early stopping
4. deterministic explanation generation
5. slice-based evaluation on noisy and code-mixed samples