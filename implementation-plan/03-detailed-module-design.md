# Section 3: Detailed Module Design

## 3.1 Data Processing Module

| Item | Design |
| --- | --- |
| Responsibility | Read raw review records, clean multilingual text, load image files, tokenize with XLM-R, apply ConvNeXt transforms, and return tensors plus metadata |
| Inputs | `review_id`, `product_id`, `image_path`, `review_text`, `overall_score`, `quality_score`, `price_score`, `appearance_score` |
| Outputs | `pixel_values`, `input_ids`, `attention_mask`, `overall_score`, `factor_scores`, `meta` |
| Suggested files | `src/data/cleaning.py`, `src/data/transforms.py`, `src/data/dataset.py`, `src/data/collate.py` |

### Text preprocessing pipeline

Recommended steps:

1. Normalize Unicode to NFC and strip control characters.
2. Remove or replace URLs, tracking strings, and non-semantic boilerplate.
3. Normalize whitespace.
4. Reduce repeated characters only when they are clearly noise.
5. Preserve Vietnamese diacritics.
6. Preserve emojis because they often carry sentiment and quality signals.
7. Expand slang and abbreviations through a small domain dictionary when possible.
8. Do not remove stopwords.
9. Do not lowercase blindly because XLM-R is pretrained on multilingual text with casing information.
10. Tokenize with `AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)`.

Recommended tokenizer settings:

* `padding="max_length"`
* `truncation=True`
* `max_length=96` for the first baseline
* increase to `128` only if review truncation becomes a real issue

### Image preprocessing pipeline

Recommended steps:

1. Read image as RGB.
2. Handle broken or missing files with a fallback policy.
3. Use the transform config expected by the selected ConvNeXt model.
4. Apply augmentation only during training.

Recommended implementation:

* use `timm.data.resolve_model_data_config()`
* use `timm.data.create_transform()` for both training and evaluation presets

Typical train-time augmentations:

* resized crop
* horizontal flip
* color jitter
* mild blur or JPEG compression simulation for low-quality uploads

### Data split strategy

Prefer splitting by product or review group rather than pure random row split when possible. This reduces leakage from nearly identical review samples.

## 3.2 Model Module

| Item | Design |
| --- | --- |
| Responsibility | Encode images and text, fuse both modalities, and output continuous quality scores |
| Inputs | batch tensors from the data module |
| Outputs | overall score, factor scores, image embedding, text embedding, optional attention or spatial features |
| Suggested files | `src/models/model_image.py`, `src/models/model_text.py`, `src/models/model_fusion.py`, `src/models/heads.py` |

### Image encoder design

Use `timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")`.

Recommended settings:

* start with `convnext_tiny`
* use `convnext_base` only if the baseline is underfitting and compute allows it
* project image features to a smaller fusion dimension such as `256`

Expected outputs:

* pooled image embedding for fusion
* spatial feature map for Grad-CAM and optional cross-attention

### Text encoder design

Use `AutoModel.from_pretrained("xlm-roberta-base")`.

Recommended settings:

* start with `xlm-roberta-base`
* use mean pooling over valid tokens instead of only the first token for noisy, short reviews
* keep `output_attentions=True` only when explanation artifacts are needed, because it adds overhead

Expected outputs:

* sentence embedding for fusion
* token-level hidden states for explanation and optional cross-attention

### Fusion design

Baseline fusion:

* project both image and text embeddings into the same hidden space
* concatenate them
* pass through an MLP

Optional cross-attention extension:

* flatten ConvNeXt spatial features into image tokens
* project XLM-R token features and image tokens to the same dimension
* let text tokens attend to image tokens before pooling
* use this only after the concatenation baseline is stable

### Prediction heads

Use two heads:

* `overall_head`: outputs one scalar per sample
* `factor_head`: outputs three values for `quality`, `price`, and `appearance`

Recommended output contract:

* `overall_score`: shape `[B]`
* `factor_scores`: shape `[B, 3]`

Optional enhancement:

* add auxiliary image-only and text-only heads during training to improve modality-specific supervision and simplify ablation studies

## 3.3 Training Pipeline Module

| Item | Design |
| --- | --- |
| Responsibility | Build loaders, run optimization, validate periodically, save checkpoints, and log results |
| Inputs | config, train or validation datasets, model, optimizer, scheduler |
| Outputs | best checkpoint, last checkpoint, metrics logs |
| Suggested files | `src/engine/train.py`, `src/engine/losses.py`, `src/engine/metrics.py`, `src/engine/checkpoint.py` |

Recommended training responsibilities:

* seed control
* model and optimizer construction
* AMP training
* gradient accumulation
* validation after each epoch
* early stopping on validation MAE or a composite metric
* checkpoint saving for both `best` and `last`

Recommended optimizer grouping:

* lowest learning rate for XLM-R base layers
* moderate learning rate for ConvNeXt fine-tuned layers
* highest learning rate for fusion and prediction heads

## 3.4 Evaluation Module

| Item | Design |
| --- | --- |
| Responsibility | Measure performance, compare ablations, and export reports |
| Inputs | checkpoint, test loader, predictions, labels |
| Outputs | metrics JSON, CSV, summary tables, plots |
| Suggested files | `src/engine/evaluate.py`, `src/engine/metrics.py` |

Required evaluation outputs:

* regression metrics: MAE, MSE, RMSE
* rank correlation: Pearson or Spearman
* optional classification metrics after bucketizing scores into bands
* ablation comparison across:

  * image-only
  * text-only
  * multi-modal fusion

Recommended report slices:

* short reviews only
* code-mixed reviews only
* low-quality images only

## 3.5 Inference API Module

| Item | Design |
| --- | --- |
| Responsibility | Load trained artifacts once, preprocess incoming requests, run inference, and return predictions in a stable schema |
| Inputs | `review_text`, uploaded image file, optional flags such as `include_explanations` |
| Outputs | JSON response with scores and optional explanation artifacts |
| Suggested files | `src/serving/api.py`, `src/serving/inference.py`, `src/serving/schemas.py` |

Recommended response contract:

```json
{
  "overall_score": 7.8,
  "factor_scores": {
    "quality": 8.1,
    "price": 6.0,
    "appearance": 8.4
  },
  "explanations": {
    "text_tokens": ["dep", "good", "gia"],
    "image_heatmap_path": "artifacts/explainability/sample_001.png",
    "summary": "The product looks good overall, but value for money is weaker."
  }
}
```

Keep API logic thin. All heavy preprocessing and model logic should live in reusable service functions.

## 3.6 Explainability Module

| Item | Design |
| --- | --- |
| Responsibility | Generate interpretable evidence for image, text, and final multi-modal predictions |
| Inputs | trained model, raw sample, intermediate features |
| Outputs | heatmaps, token importance, modality contributions |
| Suggested files | `src/explain/gradcam.py`, `src/explain/attention.py`, `src/explain/shap_lime.py` |

Recommended design:

* Grad-CAM should target the last spatial stage of ConvNeXt.
* Text explanation should visualize token importance from hidden states or attention summaries.
* SHAP or LIME should start on pooled embeddings or final features instead of raw high-dimensional tokens and pixels, to keep runtime manageable.

Important note:

* Attention visualization is useful for diagnostics, but it should not be described as causal proof.

## 3.7 AI Agent Module

| Item | Design |
| --- | --- |
| Responsibility | Convert scores and explanation artifacts into natural language for end users or analysts |
| Inputs | overall score, factor scores, top tokens, image evidence, modality contribution |
| Outputs | concise explanation string or structured explanation object |
| Suggested files | `src/agent/explanation_agent.py` |

Recommended two-stage design:

1. Stage one uses deterministic templates so the system is stable and offline-capable.
2. Stage two optionally calls an LLM or agent layer for more fluent explanations behind a feature flag.

Example template logic:

* if appearance is high and price is low, mention strong look but weak value
* if text and image disagree, mention conflicting evidence explicitly
* if confidence is low, add a caution note

This staged design avoids making the AI Agent a hard dependency for the first production-ready version.