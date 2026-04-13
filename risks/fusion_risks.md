# Multi-Modal Fusion Risks

This document focuses on failure modes specific to combining ConvNeXt image features and XLM-R text features.

## Risk 1: Ineffective Fusion Where One Modality Is Ignored

**Cause**

* Concatenation is simple and stable, but it does not guarantee that both modalities are actually used.
* If one branch is consistently easier to optimize, the MLP head may learn to ignore the other branch.

**Example**

* The model behaves almost identically to the text-only baseline, even when product images contain obvious defects.
* Validation metrics barely improve over XLM-R alone.

**Mitigation**

* Always compare multi-modal results against image-only and text-only baselines.
* Track modality ablations during validation by zeroing or masking one branch at a time.
* Add modality dropout so training batches sometimes hide text or image features.
* Normalize both embeddings before concatenation and keep projection dimensions aligned.

## Risk 2: Conflicting Image And Text Signals

**Cause**

* Real reviews frequently contain disagreement between text and images.
* The review text may praise the product while the user-uploaded image shows damage, or the text may mention price dissatisfaction while the product looks visually fine.

**Example**

* The text says `good quality`, but the image shows peeling edges or missing accessories.
* A naive fusion layer averages the conflict away and predicts a safe mid-range score.

**Mitigation**

* Inspect disagreement samples explicitly during validation.
* Add a confidence or disagreement feature, such as the absolute gap between image-only and text-only predictions.
* Let the AI Agent mention conflicting evidence rather than forcing a single smooth narrative.
* Use a simple gating mechanism or disagreement-aware rule before attempting full cross-attention.

## Risk 3: Cross-Attention Instability On Small Data

**Cause**

* Cross-attention introduces many more trainable interactions than concatenation.
* With a small noisy dataset and single-GPU training, it can overfit, become hard to optimize, or exceed memory limits.

**Example**

* A cross-attention model improves training loss but gives unstable validation MAE across runs.
* GPU memory usage spikes because text tokens attend over flattened ConvNeXt spatial tokens.

**Mitigation**

* Use concatenation as the default baseline and only add cross-attention after baseline stability.
* Reduce image token count by using the last ConvNeXt stage only.
* Keep attention heads and fusion dimension small.
* Run repeated seeds on a smaller validation subset before adopting the more complex fusion design.

## Risk 4: Misaligned Image-Text Pairs

**Cause**

* In e-commerce data, the image associated with a review may be a listing image, a generic photo, or a user-uploaded picture that does not directly describe the review text.
* Fusion assumes some semantic relationship between modalities, but the actual alignment can be weak.

**Example**

* The review complains about sizing or late delivery, while the attached image is a clean catalog-style listing photo.
* The fusion model learns unreliable image-text correlations.

**Mitigation**

* Add a metadata flag for `listing image` versus `user-uploaded image` if available.
* Filter out clearly uninformative or mismatched image-text pairs from the first training set.
* Use the image branch primarily for visual quality factors and avoid forcing it to explain logistics or shipping complaints.
* Keep disagreement cases in an error-analysis bucket instead of training on them blindly.

## Risk 5: Fusion Complexity Without Real Performance Gain

**Cause**

* On limited data, fusion can add parameters and runtime while providing little benefit over the stronger single modality.
* This is common when review text already captures most of the quality signal or when images are low quality.

**Example**

* The multi-modal model is slower and harder to train, but validation MAE is not meaningfully better than XLM-R alone.

**Mitigation**

* Define a clear promotion rule for fusion, such as a minimum improvement in validation MAE or performance on low-information text samples.
* Evaluate multi-modal gains on slices where images should matter most, such as packaging damage or visual appearance.
* If fusion adds no measurable value, keep image reasoning as a secondary explanation path instead of a primary scoring input.

## Optional Improvements

* Add a learned gating network that weighs image and text contributions per sample.
* Use auxiliary unimodal losses so the fused representation stays grounded in each branch.
* Add a disagreement score to the final output schema for safer downstream explanation.