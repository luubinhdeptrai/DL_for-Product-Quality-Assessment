# Top 5 Most Critical Risks

This document ranks the risks that are most likely to damage project quality, credibility, or demo stability for this specific ConvNeXt + XLM-R Shopee/Lazada system.

## Priority Ranking

| Rank | Risk | Why It Is Critical | Immediate Mitigation |
| --- | --- | --- | --- |
| 1 | Label noise from weak supervision | If labels are unreliable, every downstream metric, explanation, and demo claim becomes questionable | Create a small manually reviewed validation subset and track label confidence |
| 2 | Train-validation leakage and crawl bias | Inflated offline metrics can make the system look better than it really is | Split by product or listing group and run deduplication checks |
| 3 | Overfitting and forgetting during fine-tuning | The model can look strong on training data while losing multilingual and visual generalization | Freeze lower layers first, use early stopping, and apply differential learning rates |
| 4 | Modality dominance and ineffective fusion | The multi-modal design fails if one branch is effectively ignored | Compare against unimodal baselines and add modality dropout or ablation checks |
| 5 | Explanation-agent mismatch or hallucination | A fluent but unfaithful explanation will damage trust faster than a modestly inaccurate score | Use structured inputs and deterministic templates before any LLM-backed agent |

## 1. Label Noise From Weak Supervision

**Why It Happens**

* Star ratings, user text, and visual evidence do not align consistently in Shopee/Lazada reviews.
* Factor labels such as `price` and `appearance` are especially vulnerable if derived from heuristics.

**Why It Matters Here**

* Your model predicts both overall and factor scores, so noisy labels do not just hurt one target.
* They also corrupt the explanations that depend on those scores.

**First Practical Fix**

* Build a manually reviewed subset for validation and failure analysis before optimizing architecture complexity.

## 2. Train-Validation Leakage And Crawl Bias

**Why It Happens**

* E-commerce data contains repeated listing images, near-duplicate reviews, and skewed category coverage.

**Why It Matters Here**

* ConvNeXt can memorize repeated visual patterns and XLM-R can memorize repeated review templates, making validation scores misleading.

**First Practical Fix**

* Split by `product_id` or `listing_id`, and run image-text deduplication before trusting any metric.

## 3. Overfitting And Forgetting During Fine-Tuning

**Why It Happens**

* `xlm-roberta-base` and ConvNeXt are strong backbones, but the dataset is likely noisy and not large.

**Why It Matters Here**

* The model can lose the multilingual priors that made XLM-R attractive in the first place.
* It can also overfit seller-specific photo styles rather than actual product quality cues.

**First Practical Fix**

* Start with partial freezing and progressive unfreezing, then monitor performance by language and category slice.

## 4. Modality Dominance And Ineffective Fusion

**Why It Happens**

* One branch can be easier to optimize, especially when text is short but semantically strong or images are visually repetitive.

**Why It Matters Here**

* If the multi-modal model behaves like the stronger unimodal baseline, the added system complexity is not justified.

**First Practical Fix**

* Run image-only, text-only, and fusion models under the same split and compare disagreement cases, not just average metrics.

## 5. Explanation-Agent Mismatch Or Hallucination

**Why It Happens**

* The explanation layer sits above the predictive model and can introduce its own errors even when the base model is correct.

**Why It Matters Here**

* A polished but false explanation is often more damaging to trust than a raw numeric score.

**First Practical Fix**

* Use deterministic templates driven by named score fields and evidence snippets before adding any free-form LLM behavior.

## Recommendation

If project time is limited, address these five items before investing in cross-attention, SHAP-heavy workflows, or a more advanced agent.