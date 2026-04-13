# Section 6: Practical Tips & Pitfalls

## 6.1 Do Not Over-Clean The Text

Common mistake:

* lowercasing everything
* removing emojis
* stripping stopwords
* deleting Vietnamese diacritics

Why it is risky:

* XLM-R already handles noisy multilingual text better than older pipelines.
* Emojis and repeated punctuation often carry useful sentiment or emphasis.
* Vietnamese diacritics can change meaning.

## 6.2 Watch For Label Noise

Ratings and written reviews may disagree. A user might give five stars but write a mixed comment, or vice versa.

Practical mitigation:

* manually inspect a sample of labels
* use robust regression loss if disagreement is frequent
* keep a `label_source` field if some labels are manual and others are weakly supervised

## 6.3 Avoid Data Leakage

If the same product appears in both train and validation with highly similar text or images, your metrics will be inflated.

Safer split rules:

* split by product group when possible
* deduplicate near-identical images
* keep copies or resized variants of the same image in only one split

## 6.4 Start With The Baseline Fusion Model

Cross-attention is attractive, but it is not the first thing to build.

Reason:

* concatenation is easier to debug
* it gives a reliable performance reference
* it helps you determine whether the extra complexity is justified

## 6.5 Keep Train-Time And Inference-Time Preprocessing Aligned

Mismatch examples:

* training with slang normalization but serving raw text
* training with one tokenizer version and serving another
* training with ConvNeXt normalization but serving plain resized images

Practical rule:

* centralize preprocessing functions and reuse them in both training and FastAPI inference

## 6.6 Be Careful With Explainability Claims

Grad-CAM and attention visualizations are useful, but they are not formal proof of causal reasoning.

Safe framing:

* use them as evidence about what the model focuses on
* combine them with error analysis and ablation results
* use SHAP or LIME at the fused-feature level for a more grounded view of contribution

## 6.7 Control Memory Usage Early

Likely bottlenecks on one GPU:

* `xlm-roberta-large`
* long sequence lengths
* large image resolution
* cross-attention over many image tokens

Practical fixes:

* use `xlm-roberta-base`
* keep `max_length` tight
* use AMP
* use gradient accumulation
* reduce image resolution only if accuracy loss is acceptable

## 6.8 Plan For Missing Or Corrupted Images

Real crawled datasets rarely stay clean.

Add explicit handling for:

* missing files
* unreadable images
* zero-byte downloads
* non-RGB formats

Choose one fallback policy and keep it consistent:

* skip the sample
* substitute a placeholder image
* route the sample to a text-only fallback model

## 6.9 Use The AI Agent Conservatively At First

For the first version, keep the explanation layer deterministic.

Reason:

* easier to test
* easier to debug
* no external dependency required

Then add a richer agent or LLM layer once the core predictions and evidence are stable.