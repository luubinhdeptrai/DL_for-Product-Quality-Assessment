# Model-Related Risks

This document focuses on risks inside the ConvNeXt, XLM-R, and prediction-head design, independent of API or deployment concerns.

## Risk 1: Overfitting During Fine-Tuning

**Cause**

* `convnext_tiny` and `xlm-roberta-base` still have large capacity relative to a likely student-scale Shopee/Lazada dataset.
* A small number of noisy samples can cause the fusion head to memorize seller-specific styles, repeated review templates, or common product backgrounds.

**Example**

* Training MAE improves steadily, but validation MAE worsens after a few epochs.
* The model performs well on known clothing categories yet fails badly on held-out products with different image composition.

**Mitigation**

* Start with partial freezing of both backbones and train only projection layers and heads first.
* Use early stopping on validation MAE, not only training loss.
* Prefer `convnext_tiny` over `convnext_base` for the first stable baseline.
* Keep fusion dimensions modest, such as `256`, instead of using a very wide MLP.

## Risk 2: Catastrophic Forgetting Of Pretrained Knowledge

**Cause**

* If learning rates are too high or all layers are unfrozen too early, XLM-R may lose multilingual robustness and ConvNeXt may lose general visual representations.
* This is more likely with noisy labels and small batches.

**Example**

* After aggressive fine-tuning, XLM-R becomes good at a narrow set of Vietnamese product phrases but degrades on English fragments or mixed-language reviews.
* ConvNeXt begins to key on specific seller backgrounds instead of general product cues.

**Mitigation**

* Use differential learning rates: lowest for XLM-R base layers, moderate for ConvNeXt, highest for fusion and heads.
* Unfreeze progressively rather than end-to-end from the start.
* Track performance by language slice to detect loss of multilingual capability early.
* Consider layer-wise learning-rate decay if you need a stronger but still stable fine-tuning strategy.

## Risk 3: Modality Dominance Inside The Prediction Head

**Cause**

* One encoder may produce more stable or larger-magnitude features than the other.
* With short reviews, XLM-R can still dominate due to strong semantic priors, while with noisy images the system may learn to ignore ConvNeXt features.
* The reverse can also happen if text quality is poor and the images are visually consistent.

**Example**

* The review says `good quality` but the image clearly shows damaged packaging.
* The fused model still predicts a high overall score because the text branch overwhelms the visual branch.

**Mitigation**

* Project both modalities to the same hidden dimension and monitor feature norms.
* Add auxiliary image-only and text-only heads during training for ablation and debugging.
* Use modality dropout during training so the model cannot rely on just one branch.
* Compare image-only, text-only, and multi-modal predictions on the same samples.

## Risk 4: Poor Generalization To New Product Types

**Cause**

* ConvNeXt may learn category-specific appearance cues rather than reusable quality cues.
* XLM-R may memorize common review patterns from dominant categories such as cosmetics or fashion.
* The scoring head may not transfer well to categories with very different defect patterns.

**Example**

* A model trained mostly on fashion reviews performs poorly on electronics, where defects are described differently and visual problems are less obvious.

**Mitigation**

* Track metrics by category and by unseen-product split.
* Use product-group or category-aware validation rather than pure random splits.
* Keep the first deployment scope narrow if the dataset is category-skewed.
* Add category balancing or per-category caps during data sampling.

## Risk 5: Regression Collapse Toward The Mean

**Cause**

* Continuous score prediction under noisy supervision often encourages the model to predict a safe middle range.
* This is especially common when positive reviews dominate and low-score cases are rare.

**Example**

* The system predicts most samples between `6.5` and `8.0` regardless of obvious product differences.
* Overall MAE looks acceptable, but the model fails at identifying truly poor products.

**Mitigation**

* Plot prediction histograms against target histograms after every major run.
* Evaluate error separately for low, medium, and high score bands.
* Use weighted loss or sampling for rare low-quality cases.
* Standardize or normalize regression targets during training if optimization is unstable.

## Risk 6: Inconsistent Factor Head Behavior

**Cause**

* `quality`, `price`, and `appearance` are correlated but not identical.
* If factor labels are weak or sparse, the three-output head may learn inconsistent semantics.

**Example**

* A sample gets a high `appearance` score, low `quality` score, and an overall score that does not reflect either branch clearly.
* The AI Agent then struggles to explain the inconsistency.

**Mitigation**

* Keep factor definitions explicit in the annotation guide.
* Check factor correlations and review outlier samples manually.
* Use a smaller factor head first instead of a deep output stack.
* If factor labels are too noisy, prioritize overall quality first and add factor prediction later.

## Optional Improvements

* Use layer-wise learning-rate decay for XLM-R and ConvNeXt.
* Add simple calibration on validation outputs before deployment.
* Use robust loss such as Huber for overall score prediction when weak labels are noisy.