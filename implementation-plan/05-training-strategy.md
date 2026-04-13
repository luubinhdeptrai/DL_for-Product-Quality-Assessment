# Section 5: Training Strategy

## 5.1 Training Order

Use a phased approach rather than training the most complex model first.

### Phase A: Data pipeline validation

Goal:

* verify dataset loading
* check text cleaning behavior on Vietnamese and code-mixed samples
* verify image transform output shape and value range

Deliverables:

* batch inspection script
* sample visualizations
* split statistics

### Phase B: Unimodal baselines

Train separately:

* image-only model with ConvNeXt
* text-only model with XLM-R

Purpose:

* establish sanity-check performance
* identify whether one modality is much stronger than the other
* simplify debugging before fusion

### Phase C: Multi-modal baseline

Train the concatenation model first.

Purpose:

* validate that fusion improves over each unimodal baseline
* keep memory and implementation complexity under control

### Phase D: Optional cross-attention extension

Only run this phase if:

* the baseline is stable
* you have enough training data
* the improvement target justifies the extra complexity

## 5.2 Recommended Hyperparameters For One GPU

| Item | Start Value | Notes |
| --- | --- | --- |
| Image model | `convnext_tiny` | Default baseline |
| Text model | `xlm-roberta-base` | Default baseline |
| Image size | `224` | Good trade-off for ConvNeXt |
| Max text length | `96` | Increase only if truncation is frequent |
| Batch size | `4-8` | Depends on VRAM |
| Gradient accumulation | `2-4` | Use if batch size is too small |
| Epochs | `6-12` | Stop early if validation plateaus |
| Early stopping patience | `3-5` | Monitor validation MAE |
| Optimizer | `AdamW` | Standard for transformer and CNN fine-tuning |
| Weight decay | `1e-2` | Good default |
| Mixed precision | enabled | Recommended |

## 5.3 Learning Rate Strategy

Use differential learning rates.

Recommended first pass:

* XLM-R lower layers: `1e-5`
* ConvNeXt fine-tuned layers: `3e-5` to `1e-4`
* fusion and prediction heads: `1e-3` to `2e-4`

If you prefer simpler setup, start with a single learning rate such as `2e-4` for new heads and partially frozen backbones, then move to parameter groups once the pipeline is working.

## 5.4 Freezing Strategy

For noisy data and limited compute, avoid fully unfreezing everything at the start.

Recommended schedule:

1. Freeze most of ConvNeXt and XLM-R for the first `1-2` epochs.
2. Train only projection layers, fusion module, and prediction heads.
3. Unfreeze upper encoder blocks once losses are stable.
4. Keep lower layers frozen if overfitting or memory usage becomes a problem.

## 5.5 Loss Design

### Basic option

* overall loss: MSE or MAE
* factor loss: MSE or MAE
* final loss: `overall_loss + alpha * factor_loss`

### Practical option for noisy labels

Use Huber or SmoothL1 for at least the overall score if label noise is high.

Suggested formula:

```text
total_loss = overall_loss + 0.5 * factor_loss
```

Tune `0.5` based on whether factor supervision is reliable.

## 5.6 Validation Strategy

Run validation after every epoch.

Track at minimum:

* validation loss
* validation MAE
* validation RMSE

Save:

* `best.pt` based on validation MAE
* `last.pt` after each epoch
* `history.json` for plotting and regression analysis

## 5.7 Required Ablations

To make the project convincing, compare these settings under the same split:

1. image-only ConvNeXt
2. text-only XLM-R
3. multi-modal concatenation
4. multi-modal cross-attention if implemented

Also compare:

* with and without slang normalization
* with and without image-quality augmentation
* short reviews versus longer reviews

## 5.8 Monitoring And Logging

Track these artifacts for every run:

* config snapshot
* training and validation metrics per epoch
* learning rate schedule
* confusion-style bucketized report for score bands
* sample explanations on a fixed validation subset

If possible, log a few representative review samples every epoch to detect preprocessing drift early.