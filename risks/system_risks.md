# System And Deployment Risks

This document focuses on operational risks around training, serving, reproducibility, and limited compute.

## Risk 1: Single-GPU Memory Pressure

**Cause**

* XLM-R and ConvNeXt together already create a nontrivial memory footprint.
* Cross-attention, larger image sizes, and explanation artifacts can quickly exceed student-scale GPU limits.

**Example**

* Training works with `convnext_tiny` and `xlm-roberta-base`, but enabling cross-attention or longer sequence length causes out-of-memory failures.

**Mitigation**

* Keep the baseline at `convnext_tiny`, `xlm-roberta-base`, image size `224`, and short text length.
* Use AMP mixed precision and gradient accumulation.
* Freeze lower encoder layers early in training.
* Keep SHAP and LIME out of the training path entirely.

## Risk 2: Inference Latency Becomes Too High

**Cause**

* Serving ConvNeXt, XLM-R, fusion, and explanation logic in a single synchronous request can lead to slow response time.
* FastAPI will remain simple, but the underlying workload is still heavy.

**Example**

* The `/predict` endpoint is acceptable when only scores are returned, but response time becomes poor when Grad-CAM and explanation generation are always enabled.

**Mitigation**

* Separate `score-only` and `explain` modes in the API.
* Load models once at startup and reuse them across requests.
* Keep Grad-CAM and SHAP generation asynchronous or on-demand.
* Benchmark latency on representative review-image pairs before expanding the API contract.

## Risk 3: Training And Inference Preprocessing Drift

**Cause**

* It is easy to accidentally use different text cleaning rules, tokenization settings, or image normalization between training and serving.
* This silently degrades production behavior.

**Example**

* Training preserves emojis and uses `max_length=96`, but the API strips emojis or uses different truncation settings.
* The model underperforms in production even though offline validation looked good.

**Mitigation**

* Reuse the same preprocessing functions in both training and serving.
* Version tokenizer name, image transform config, and cleaning logic with the checkpoint.
* Add a small inference parity test that runs a fixed sample through both the training-side and API-side preprocessors.
* Keep configuration in YAML or structured config files instead of hardcoded constants spread across files.

## Risk 4: Fragile Handling Of Bad Inputs

**Cause**

* Real input data will include missing images, unreadable files, empty reviews, unsupported formats, and malformed requests.
* A demo system often works only on clean samples unless explicit guardrails are added.

**Example**

* A user uploads a corrupt image while sending a very short review such as `ok`.
* The API crashes or returns a misleading full-confidence score.

**Mitigation**

* Add explicit validation for empty text, unreadable images, and unsupported formats.
* Return structured error responses instead of silent failures.
* Define a fallback policy: reject, text-only inference, or placeholder image path.
* Log bad-input statistics so you know whether failures are rare or systematic.

## Risk 5: Scalability Bottlenecks In Artifact Storage And Serving

**Cause**

* Heatmaps, SHAP outputs, logs, and checkpoints can quickly become large.
* If explanation artifacts are generated per request, storage and request throughput degrade.

**Example**

* A demo generates Grad-CAM PNG files for every prediction and the artifact directory grows rapidly, making it harder to manage experiments.

**Mitigation**

* Store only selected explanation artifacts instead of every sample.
* Use retention rules for checkpoints and logs.
* Cache repeated demo outputs instead of recomputing them.
* Keep online inference lightweight and reserve heavy analysis for offline batches.

## Risk 6: Weak Reproducibility And Dependency Drift

**Cause**

* Small research projects often evolve quickly, and model behavior changes when library versions, tokenizer settings, or transforms change.
* This is especially risky when using `timm`, `transformers`, and explanation libraries together.

**Example**

* A checkpoint trained with one `transformers` version behaves differently after an environment update, and the explanation code no longer matches the saved artifacts.

**Mitigation**

* Pin dependency versions in `requirements.txt` or `pyproject.toml`.
* Save config snapshots with every checkpoint.
* Seed all training runs and log dataset versions.
* Treat model checkpoints, tokenizer configuration, and image transform settings as one deployable unit.

## Optional Improvements

* Add a background worker or queue for expensive explanation jobs.
* Add lightweight model monitoring with latency and input-quality logs.
* Add CPU fallback for demo inference if GPU access is unavailable.