# Model Card — Label_Trainer Intrusion Detector

## Model Description

- **Inputs.** Tabular feature vectors **X** derived from network flows/payload statistics loaded from `archive/Payload_data_UNSW.csv`. Each row corresponds to a session/flow with numeric features and a binary label `y`.
- **Outputs.**
  - **Score:** Predicted probability \( \hat p = P(y=1 \mid X) \).
  - **Class:** Decision \( \hat y = \mathbb{1}[\hat p \ge \tau] \) using a threshold **τ** selected to maximize **F1** on validation.
- **Model architectures.**
  - **Primary:** Gradient-boosted trees (**LightGBM** classifier) with class‑imbalance aware tuning.
  - **Alternative (optional):** CNN‑based classifier for derived representations; used as an experimental baseline.
- **Training setup.** Stratified train/validation/test split; **Bayesian optimization** (skopt) plus parallel grid search; artifacts checkpointed under `staging/` for exact resume.

## Performance

**Metrics reported (validation/test):** Precision, Recall/TPR, Specificity/TNR, F1 (at **τ**), and **AUPRC** under class imbalance.  
**Evaluation protocol:** 
- Select **τ** on the validation split by maximizing F1. 
- Report all metrics on the held‑out test split at the chosen **τ** and the **AUPRC** as a threshold‑free summary.

> Tip: After running the notebook, you can auto-fill the numbers here from `staging/bo_lgb/manifest.json` and `staging/manual_grid/manifest.json`.

**Example field template (replace after running):**
- AUPRC (test): `0.XXX`
- F1@τ (test): `0.XXX` (τ = `0.XXX`)
- Precision@τ / Recall@τ (test): `0.XXX` / `0.XXX`
- TNR@τ (test): `0.XXX`

## Limitations

- **Domain shift.** Performance may drop on networks unlike the training source; re‑calibration of **τ** is recommended per environment.
- **Label noise.** If labels are derived from heuristics/signatures, minority-class noise can bias thresholding and AUPRC.
- **Interpretability.** Tree ensembles are less interpretable than single trees; use SHAP/feature importance for analyst transparency.

## Trade‑offs

- **Precision vs. coverage.** Increasing **τ** reduces false positives (higher precision) but may miss stealthy attacks (lower recall).
- **Latency vs. accuracy.** Heavier feature pipelines and CNN baselines may improve accuracy but increase compute.
- **Generalization vs. specificity.** Tuning on one subnet/application mix may overfit to that context; prefer cross‑site validation when possible.
