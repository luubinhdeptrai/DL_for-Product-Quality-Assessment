# Data-Related Risks

This document focuses on data risks specific to a Shopee/Lazada multi-modal system built on ConvNeXt and XLM-R for noisy Vietnamese-English review data.

## Risk 1: Low-Information Reviews

**Cause**

* Shopee/Lazada reviews are often extremely short, such as `ok`, `đẹp`, `good`, or emoji-only messages.
* XLM-R can tokenize them correctly, but there is still very little semantic content available for factor-level scoring.
* The model may over-rely on weak text signals or default to average predictions.

**Example**

* A review says `đẹp nha 👍` while the image shows dented packaging and poor stitching.
* The text branch pushes the score upward even though the visual evidence suggests a weaker overall quality score.

**Mitigation**

* Track token count and create a `short_text_flag` feature for analysis and debugging.
* Evaluate separately on `very short`, `short`, and `normal-length` reviews.
* Use modality-aware fallback logic during inference, such as lowering confidence when token count is below a threshold.
* Keep the baseline fusion model simple so you can tell whether text is helping or just adding noise.

## Risk 2: Code-Mixed Slang And Abbreviation Drift

**Cause**

* Vietnamese-English code mixing is common in e-commerce reviews.
* Slang, abbreviations, and seller-specific shorthand change quickly and are not consistently covered by pretrained multilingual corpora.
* Even XLM-R can fragment these expressions into subwords that are hard to learn from a small dataset.

**Example**

* Reviews such as `sp ok, form đẹp, auth nha, shop rep ib lẹ` combine Vietnamese shorthand, English, and platform slang.
* The tokenizer may technically process the text, but the semantic intent is still noisy and inconsistent across users.

**Mitigation**

* Build a small normalization dictionary from the most frequent abbreviations in your crawl.
* Preserve emojis and diacritics instead of aggressively cleaning them away.
* Inspect tokenization statistics on real samples, especially heavily code-mixed reviews.
* Add light text normalization augmentation so the model sees both raw and normalized forms during training.

## Risk 3: Label Noise From Weak Supervision

**Cause**

* Platform star ratings do not always reflect product quality alone.
* Weakly derived labels for `quality`, `price`, and `appearance` may be inconsistent with the actual review text or image.
* Human annotators may also disagree on subtle factors like `appearance` versus `overall quality`.

**Example**

* A user gives 5 stars because shipping was fast, but the text says the material is thin and the color differs from the listing.
* If you map the star rating directly to overall quality, the model learns a misleading signal.

**Mitigation**

* Keep `label_source` and `label_confidence` fields in the dataset.
* Create a small manually reviewed validation subset to anchor training decisions.
* Use Huber or SmoothL1 loss instead of pure MSE when label noise is obvious.
* Avoid deriving all factor scores directly from stars; use separate annotation or weak heuristics per factor.

## Risk 4: Score Imbalance And Category Bias

**Cause**

* Real e-commerce reviews skew positive, so low-quality examples are often underrepresented.
* Popular categories such as clothing or cosmetics may dominate the crawl, while electronics or household items remain sparse.
* The model can end up learning the majority score range rather than true quality distinctions.

**Example**

* The training set is dominated by 4-5 star fashion items with polished listing images.
* The system later underperforms on low-scoring electronics reviews where defects are visually subtle and language is more technical.

**Mitigation**

* Build stratified splits by score bucket and product category.
* Use weighted sampling or loss reweighting for underrepresented score ranges.
* Report MAE and RMSE by score bucket, not only overall averages.
* Cap the number of reviews per product or seller to reduce popularity bias.

## Risk 5: Train-Validation Leakage Through Near Duplicates

**Cause**

* Shopee/Lazada data often contains repeated listing images, resized copies, templated review text, or multiple reviews for the same product variant.
* A random row-level split can leak visually or semantically identical samples across train and validation.

**Example**

* The same product image appears in training and validation, with only slight crop or compression differences.
* ConvNeXt appears to generalize well, but the metric is inflated by memorization.

**Mitigation**

* Split by `product_id`, `listing_id`, or a stable group identifier instead of pure random row split.
* Use perceptual image hashing to detect near-duplicate images.
* Deduplicate templated or highly similar review texts before splitting.
* Keep a leakage audit report for each dataset version.

## Risk 6: Crawl Bias And Missing Negative Evidence

**Cause**

* Crawled data may overrepresent accessible pages, popular sellers, or surviving reviews while underrepresenting deleted, hidden, or rare negative cases.
* This creates a gap between research data and the real distribution encountered in deployment.

**Example**

* The crawl contains mostly public, highly engaged, positively rated listings, while genuinely poor-quality products are sparse.
* The model looks strong offline but misses failure cases once evaluated on a harsher sample.

**Mitigation**

* Track dataset composition by language, category, score bucket, and image quality.
* Add targeted collection for underrepresented slices such as low-score reviews or code-mixed cross-border listings.
* Maintain evaluation slices for `low-light images`, `blurry images`, `short reviews`, and `code-mixed reviews`.
* Explicitly document the deployment scope if the crawl remains skewed.

## Optional Improvements

* Use active learning to manually review the most uncertain or contradictory samples.
* Maintain a gold validation set of a few hundred samples instead of relying only on weak labels.
* Add a lightweight language-tag field and a text-noise score for better slice-based evaluation.