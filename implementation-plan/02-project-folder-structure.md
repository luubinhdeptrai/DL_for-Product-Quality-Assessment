# Section 2: Project Folder Structure

## 2.1 Recommended Repository Layout

```text
DL_for-Product-Quality-Assessment/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ splits/
├─ artifacts/
│  ├─ checkpoints/
│  ├─ logs/
│  ├─ metrics/
│  └─ explainability/
├─ notebooks/
├─ src/
│  ├─ configs/
│  │  ├─ train.yaml
│  │  ├─ model.yaml
│  │  └─ inference.yaml
│  ├─ data/
│  │  ├─ cleaning.py
│  │  ├─ transforms.py
│  │  ├─ dataset.py
│  │  └─ collate.py
│  ├─ models/
│  │  ├─ model_image.py
│  │  ├─ model_text.py
│  │  ├─ model_fusion.py
│  │  └─ heads.py
│  ├─ engine/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ losses.py
│  │  ├─ metrics.py
│  │  └─ checkpoint.py
│  ├─ explain/
│  │  ├─ gradcam.py
│  │  ├─ attention.py
│  │  └─ shap_lime.py
│  ├─ serving/
│  │  ├─ api.py
│  │  ├─ schemas.py
│  │  └─ inference.py
│  ├─ agent/
│  │  └─ explanation_agent.py
│  └─ utils/
│     ├─ config.py
│     ├─ logging.py
│     ├─ seed.py
│     └─ device.py
├─ tests/
├─ implementation-plan/
├─ Ideas for project.md
└─ README.md
```

## 2.2 Folder Responsibilities

### `data/`

Stores raw crawled records, cleaned intermediate tables, processed metadata, and reproducible train or validation splits.

### `artifacts/`

Stores outputs created during experiments:

* checkpoints
* logs
* metrics JSON files
* Grad-CAM images and other explanation artifacts

### `src/data/`

Contains the logic that converts raw rows into model-ready tensors.

### `src/models/`

Contains modality encoders, fusion logic, and prediction heads.

### `src/engine/`

Contains training, validation, evaluation, loss calculation, and checkpointing logic.

### `src/explain/`

Contains explainability methods for image, text, and multi-modal outputs.

### `src/serving/`

Contains inference-time preprocessing, model loading, and FastAPI serving code.

### `src/agent/`

Contains deterministic or LLM-backed logic that turns predictions into human-readable explanations.

## 2.3 Minimal Viable Codebase

If you want the leanest version that still supports training and inference, start with:

```text
src/
├─ data/
│  └─ dataset.py
├─ models/
│  ├─ model_image.py
│  ├─ model_text.py
│  └─ model_fusion.py
├─ engine/
│  ├─ train.py
│  └─ evaluate.py
└─ serving/
   └─ api.py
```

Then expand into the fuller structure once the first training run is stable.

## 2.4 Naming And Packaging Rules

Use these conventions consistently:

* Keep one responsibility per file.
* Keep research notebooks outside `src/`.
* Put configuration in YAML instead of hardcoding paths and hyperparameters.
* Save artifacts outside the source tree.
* Keep inference-time preprocessing identical to training-time preprocessing except for augmentation.

## 2.5 Suggested Configuration Keys

Keep at least these keys in config files:

* dataset paths
* label columns
* image size
* tokenizer name
* max sequence length
* model names
* fusion type
* learning rates per module
* batch size and gradient accumulation
* checkpoint path
* explainability options