# Section 1: System Breakdown

## 1.1 Target System

The system is an end-to-end multi-modal pipeline for product quality assessment from Shopee/Lazada review data. It consumes a product image and a short multilingual review, predicts an overall quality score and factor scores, then returns model evidence and a readable explanation.

Core goals:

* Support Vietnamese, English, and code-mixed reviews.
* Remain robust to slang, abbreviations, emojis, spelling noise, blur, compression, and inconsistent image quality.
* Train on a single GPU without introducing unnecessary model complexity too early.
* Expose a production-friendly inference API through FastAPI.

## 1.2 End-to-End Flow

1. Crawl or ingest Shopee/Lazada review records with text, image paths or URLs, and labels.
2. Normalize and validate the raw data.
3. Preprocess text with XLM-R tokenizer and images with ConvNeXt-compatible transforms.
4. Encode image and text separately.
5. Fuse the two embeddings.
6. Predict:

   * overall quality score
   * factor scores for quality, price, and appearance
7. Generate:

   * image explanation via Grad-CAM
   * text explanation via attention visualization
   * multi-modal explanation via SHAP or LIME
   * final user-facing summary via the AI Agent

## 1.3 Modules At A Glance

| Module | Responsibility | Main Output |
| --- | --- | --- |
| Data processing | Clean raw text and images, tokenize, transform, package tensors | batch-ready tensors |
| Image encoder | Extract ConvNeXt image embedding and spatial features | image embedding |
| Text encoder | Extract XLM-R sentence embedding and token features | text embedding |
| Fusion model | Combine modalities and predict scores | overall and factor predictions |
| Training pipeline | Optimize model, validate, checkpoint, log metrics | trained checkpoints |
| Evaluation | Compute regression and classification-style metrics, run ablations | reports and metrics |
| Explainability | Produce heatmaps, token importance, modality contribution | interpretable artifacts |
| Inference API | Serve `/predict` endpoint | JSON response |
| AI Agent | Convert evidence into natural language | explanation text |

## 1.4 Recommended Starting Configuration

For a single GPU, start with the smallest design that is still representative of the final system.

| Component | Recommended Start | Why |
| --- | --- | --- |
| Image backbone | `convnext_tiny` | Best quality-to-compute ratio for a first stable baseline |
| Text backbone | `xlm-roberta-base` | Strong multilingual performance without the memory cost of `large` |
| Fusion | Concatenation + MLP | Easy to debug, stable, and strong enough as a baseline |
| Image size | `224` | Compatible with ConvNeXt defaults and manageable on one GPU |
| Text length | `96` or `128` | Reviews are short, so longer sequences usually waste memory |
| Precision | AMP mixed precision | Reduces memory and improves throughput |
| Batch strategy | batch size `4-8` + gradient accumulation | Practical for limited VRAM |

## 1.5 Pretrained Model Trade-offs

### ConvNeXt Variants

| Model | Approx. Size | Pros | Cons | Recommendation |
| --- | --- | --- | --- | --- |
| `convnext_tiny` | ~29M params | Fastest, easier to fine-tune, strong baseline | Slightly lower ceiling on large datasets | Default choice |
| `convnext_base` | ~89M params | Better representation capacity | Much heavier on memory and training time | Use only after the baseline is stable |

### XLM-R Variants

| Model | Approx. Size | Pros | Cons | Recommendation |
| --- | --- | --- | --- | --- |
| `xlm-roberta-base` | ~270M params | Strong multilingual encoder, practical on one GPU | Still memory-heavy if sequence length is too high | Default choice |
| `xlm-roberta-large` | ~560M params | Better ceiling on large multilingual datasets | Often impractical on a single GPU | Use only with significant compute headroom |

## 1.6 Recommended Delivery Sequence

Implement the system in this order:

1. Data cleaning and dataset loader
2. Text-only and image-only baselines
3. Multi-modal concatenation model
4. Training and validation pipeline
5. Evaluation and ablation reporting
6. Explainability utilities
7. FastAPI inference service
8. AI Agent explanation layer

This order keeps debugging costs low and ensures the fusion model is only introduced after each modality works independently.