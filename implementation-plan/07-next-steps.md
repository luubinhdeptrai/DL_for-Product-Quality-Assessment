# Section 7: Next Steps

## 7.1 Immediate Build Order

Implement the system in this order:

1. Create the project folder structure under `src/`, `data/`, and `artifacts/`.
2. Build the raw-data inspection and cleaning scripts.
3. Implement `dataset.py` and verify one training batch end to end.
4. Train text-only and image-only baselines.
5. Train the multi-modal concatenation model.
6. Add evaluation reports and ablation comparison.
7. Add FastAPI `/predict` serving.
8. Add Grad-CAM, text visualization, and AI Agent summarization.

## 7.2 Suggested First Milestone

The first usable milestone should include:

* cleaned metadata file
* working dataset loader
* concatenation-based multi-modal model
* training loop with checkpointing
* evaluation script with MAE and RMSE

Do not block this milestone on cross-attention or SHAP.

## 7.3 Suggested Second Milestone

After the baseline is stable, add:

* ablation study across image-only, text-only, and fusion
* Grad-CAM outputs on validation samples
* top-token inspection for XLM-R
* deterministic explanation generation in the API

## 7.4 Suggested Third Milestone

Only after the previous steps are stable, consider:

* cross-attention fusion
* SHAP or LIME for fused features
* NestJS integration
* optional LLM-backed explanation agent

## 7.5 Recommended Acceptance Criteria

Before calling the system implementation-ready, verify:

* training and validation runs complete without manual intervention
* best and last checkpoints are saved correctly
* inference preprocessing matches training preprocessing
* multi-modal fusion beats at least one unimodal baseline
* explanation artifacts can be generated for held-out samples
* API returns stable JSON for valid and invalid inputs

## 7.6 What To Build Next If You Want More Help

The most natural follow-up tasks are:

1. generate the actual Python project scaffold under `src/`
2. create `requirements.txt` or `pyproject.toml`
3. write a real training config and run script
4. design the dataset schema for crawled Shopee/Lazada samples