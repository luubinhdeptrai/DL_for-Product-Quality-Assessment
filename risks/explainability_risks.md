# Explainability Risks

This document focuses on risks in Grad-CAM, attention visualization, SHAP, and LIME when applied to a ConvNeXt + XLM-R multi-modal system.

## Risk 1: Misleading Grad-CAM Heatmaps

**Cause**

* Grad-CAM can highlight regions correlated with the prediction without proving those regions contain the true product defect.
* In e-commerce data, backgrounds, packaging layout, watermarks, or seller-specific photo styles may dominate the signal.

**Example**

* The heatmap highlights the corner watermark or packaging border rather than the damaged seam that a human would inspect.

**Mitigation**

* Inspect Grad-CAM on a fixed review set spanning good and bad examples.
* Compare highlighted regions against human judgment on a small manually reviewed subset.
* Use image crops and augmentation that reduce background shortcuts during training.
* Present Grad-CAM as supporting evidence, not proof of causal reasoning.

## Risk 2: Attention Maps Interpreted As True Reasoning

**Cause**

* XLM-R attention weights show what tokens interact strongly, not necessarily what caused the decision.
* In code-mixed and noisy text, visually salient tokens can be overemphasized during interpretation.

**Example**

* The explanation UI highlights `good` and `đẹp`, while the actual prediction is also influenced by nearby negative phrases or the image branch.

**Mitigation**

* Treat attention visualization as diagnostic, not definitive.
* Pair token highlights with ablation tests, such as masking top tokens and measuring score change.
* Prefer pooled token-importance summaries over raw attention matrices in the user-facing interface.
* Keep explanation language explicit: `model focused on`, not `model proved`.

## Risk 3: SHAP Or LIME Instability On High-Dimensional Multi-Modal Features

**Cause**

* SHAP and LIME can become unstable or extremely slow when applied directly to raw image pixels and token sequences.
* Small perturbations can create explanation artifacts that look precise but vary from run to run.

**Example**

* Running LIME twice on the same review-image pair yields noticeably different token or region importance outputs.

**Mitigation**

* Apply SHAP or LIME on fused embeddings or pooled modality features instead of raw inputs.
* Use a fixed background sample set for reproducibility.
* Limit SHAP or LIME to offline analysis rather than every API request.
* Save the explainer configuration with each experiment so outputs stay comparable.

## Risk 4: False Sense Of Transparency

**Cause**

* Showing three types of explanations can make the system appear more trustworthy than it actually is.
* Users may assume the model is well-understood just because it produces heatmaps and highlighted tokens.

**Example**

* A report includes Grad-CAM, top tokens, and SHAP contributions, but all are based on a poorly calibrated model that still fails on rare categories.

**Mitigation**

* Tie explanation outputs to model confidence and validation performance.
* Show explanations together with limitations and confidence notes.
* Keep failure-case examples in project documentation, not only success-case visualizations.
* Make slice-based error analysis part of the explanation review process.

## Risk 5: Explanation Cost Becomes Operationally Impractical

**Cause**

* Grad-CAM, SHAP, and LIME add nontrivial computation and storage overhead.
* On limited compute, full explanation generation can be slower than the prediction itself.

**Example**

* The `/predict` endpoint returns scores quickly, but the request becomes slow or times out when Grad-CAM and SHAP are always enabled.

**Mitigation**

* Separate `fast prediction` from `full explanation` modes.
* Generate Grad-CAM on demand or asynchronously.
* Cache explanation artifacts for demo samples and repeated queries.
* Use lightweight token importance summaries during normal inference and reserve SHAP or LIME for offline analysis.

## Optional Improvements

* Add token ablation or occlusion tests for a small evaluation subset.
* Compare explanation consistency across checkpoints instead of trusting a single run.
* Maintain a human-reviewed explanation benchmark with a few representative cases.