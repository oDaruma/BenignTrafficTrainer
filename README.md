# Reflection: Optimising Network Intrusion Detection — **Benign-First Training for Non-Benign Identification**

## Executive summary (non-technical)

- **Goal.** Train detectors to **model benign traffic precisely** (high TNR) so we can **flag non-benign** when the predicted benign probability falls below a threshold τ. We **must** still maintain high attack detection (TPR ≥ 90%) and ensure low CPU latency. Corpora: **UNSW-NB15** (`Payload_data_UNSW.csv`) with flow stats and payload histograms.  
- **Why multiple optimisation methods.** Hyper-parameters, calibration, sampling ratios and τ are expensive to tune. We therefore used a **progression of methods**:  
  - **Grid Search** for small/logistic models.  
  - **Random Search** for tree ensembles.  
  - **Successive Halving** for large search spaces with early stopping.  
  - **Bayesian Optimisation** (optional, via `skopt`) for sample-efficient exploration.  
- **Outcome.** A CPU-deployable detector that **passes benign** reliably (minimises false alerts) and **signals non-benign** with high recall. We save the **trained pipeline**, **scaler**, **label encoder**, **metadata**, and **trials logs** for SOC deployment.

---

## 1) How optimisation skills/code transfer (technical)

### A. Direct applications (benign-first modelling)

| Optimisation skill | IDS use-case | **Target & constraints** | Implemented in notebook |
| --- | --- | --- | --- |
| **Grid Search** | Logistic Regression tuning | Objective: maximise AP (PR-AUC); constraints implicit via CV | ✅ |
| **Random Search** | Random Forest depth/leaf tuning | Broad coverage without exploding compute | ✅ |
| **Successive Halving** | SGD/linear models | Early stop bad configs; efficient on large data | ✅ |
| **Bayesian Opt** (if installed) | Gradient Boosting | Objective: maximise AP, constraint-aware search | ✅ |
| **Threshold optimisation** | τ selection post-fit | Pick τ for max-F1 or target precision | ✅ |
| **Calibration** | Reliable probabilities | Platt scaling (sigmoid); fallback isotonic | ✅ |

**Why benign-first?** In production SIEM/SOAR, benign dominates. If the model **knows benign well**, anything deviating is flagged as suspicious. We enforce a TPR floor to avoid missing true attacks.

### B. What is optimised

- **Features.** Full UNSW flow statistics and payload histograms, plus categorical protocol/service features.  
- **Models.** Baselines (LogReg, SGD), Random Forest, Gradient Boosting.  
- **Objective/constraints.** Maximise **PR-AUC**; evaluate TPR/TNR at chosen thresholds.  
- **Explainability.** Pipeline ready for SHAP/local feature importance (future extension).  

---

## 2) Questions addressed

- **Which optimiser suits which model?** Grid for small LR, Random for forests, Halving for SGD, Bayesian for boosting.  
- **What benign pass-rate (TNR) is achievable while maintaining TPR ≥ 0.90?**  
- **Do flow/payload histogram features improve discrimination under optimised τ?**  
- **How stable are thresholds/calibrations across folds?**  

---

## 3) Dataset

- **Primary:** `archive/Payload_data_UNSW.csv` (UNSW-NB15) with flow stats and payload histograms.  
- **Target column:** `label` → renamed to **`target`** (encoded 0 = Normal, 1 = Attack).  
- **Splitting:** stratified train/test split with cross-validation.  

---

## 4) Alignment

- **SOC/DFIR (must).** High TNR reduces benign alert noise; TPR floor reduces missed attacks.  
- **MLOps (should).** Multiple optimisers compared, unified evaluation metrics, trials logged to Parquet.  
- **Deployment (must).** CPU latency bounds, model/scaler/encoder/metadata saved to `ids_artifacts/`.  

---

## Project blueprint (implemented)

1. **Problem framing & KPIs.** Objective: maximise **PR-AUC**, report **TNR/TPR/F1** at chosen τ.  
2. **Data engineering.** Load UNSW CSV; retain flow/payload features; impute/scaling/encoding via `ColumnTransformer`.  
3. **Model space.** Logistic Regression, SGD, Random Forest, Gradient Boosting.  
4. **Optimisation.**  
   - Grid Search (LR)  
   - Random Search (RF)  
   - Successive Halving (SGD)  
   - Bayesian (GBC, optional)  
5. **Imbalance handling.** Class weights & stratified splits.  
6. **Calibration & τ.** Platt vs Isotonic; τ by max-F1 or target precision.  
7. **Results comparison.** Table (AP, TPR, TNR, F1); PR curves.  
8. **Artifacts.** Save model/scaler/encoder/metadata/trials to `ids_artifacts/`.  

---

## Kernel-crash avoidance & reliability controls

- **Row cap option** for very large datasets.  
- **Downcast dtypes, drop constants.**  
- **Thread limits** on BLAS/OpenMP.  
- **Safe plotting** with `plt.close()`.  
- **Checkpoint CV results** to `trials.parquet`.  
- **Skip Bayesian if `skopt` missing.**  

---

## Explanation (plain language)

*We train a computer to learn what “normal internet traffic” looks like. If a new connection doesn’t fit that pattern, we flag it as “non-benign.”*

1. We try different models and parameters.  
2. We use different search strategies (grid, random, halving, Bayesian) to find the best ones efficiently.  
3. We **calibrate** the model so probability scores are meaningful.  
4. We choose a **threshold τ** that keeps false alarms low but still catches >90% of attacks.  
5. We save everything (model, scaler, encoder, metadata, trials) for deployment in the SOC.  

---

## Mathematical/statistical notes (used in code)

- **Precision/Recall/F1/AP** as before.  
- **Sigmoid** \(σ(z)=1/(1+e^{-z})\) for logistic models & calibration.  
- **Calibration**: Platt = logistic fit, Isotonic = step-wise non-parametric.  
- **PR-AUC**: area under Precision–Recall curve, robust under imbalance.  
- **Threshold search**: τ chosen to maximise F1 or hit target precision.  
