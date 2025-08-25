# Reflection: Applying Bayesian Optimisation (BO) to Network Intrusion Detection — **Benign-First Training for Non-Benign Identification**

## Executive summary (non-technical)

* **Goal.** Learn a precise model of **benign** traffic (high **TNR**) and flag **non-benign** when the benign probability falls below a decision threshold **τ**. The detector **must** meet **TPR ≥ 0.90**, **FPR ≤ 0.005**, and **p95 CPU latency ≤ 10 ms**. Primary corpus: **UNSW-NB15** (`archive/Payload_data_UNSW.csv`) with optional cross-dataset checks on **CIC-IDS2017**. [1]–[3]
* **Why BO.** BO **must** tune hyperparameters and the operating threshold **τ** under **noisy, expensive** evaluation while honouring **hard constraints** (TPR/FPR/latency). The notebook uses a **Gaussian Process surrogate with Noisy Expected Improvement** and a **barrier objective** for constraints. [4]–[6]
* **Outcome.** A probability-calibrated, CPU-deployable detector that **passes benign** reliably (low false alerts) and **signals non-benign** at bounded **FPR** and **p95 latency**. All artefacts, trials, and **hyperparameter audit logs** are persisted to `ids_artifacts/` for SOC ingestion and reproducibility.

---

## 1) How the BO skills/code transfer (technical)

### A. Direct applications (benign-first modelling)

| BO skill                       | IDS use-case                                                    | **Optimisation target & constraints**                                                                                        | Implemented in notebook |
| ------------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| **GP surrogate + NoisyEI**     | Sample-efficient tuning of **LightGBM** hyperparameters **+ τ** | Objective: **maximise TNR**. Constraints: **TPR ≥ 0.90**, **FPR ≤ 0.005**, **p95 latency ≤ 10 ms** via **barrier objective** | ✅                       |
| **Constraint handling**        | Keep search “safe”                                              | **Must** reject/penalise infeasible configs; select **best feasible**                                                        | ✅                       |
| **Calibration before τ**       | Reliable scoring                                                | **Must** calibrate with **Isotonic CV** prior to τ optimisation                                                              | ✅                       |
| **Threshold (τ) optimisation** | Operating point                                                 | Constrained τ sweep to satisfy TPR/FPR; among feasible, **maximise TNR**                                                     | ✅                       |
| **Auditability**               | Governance                                                      | **Must** validate `HPO_SPACES` against estimator `.get_params()`; **must** log effective hyperparameters to JSONL            | ✅                       |

**Why benign-first?** Production traffic is predominantly benign. Tight benign modelling yields **low FPR / high TNR**, reducing alert fatigue; a **TPR floor** safeguards detection coverage.

### B. What is optimised

* **Features.** Flow and payload-derived features via a **ColumnTransformer** (scaled numerics + OHE categoricals). CICFlowMeter may be used for flow extraction in CIC-IDS2017 contexts. [3], [8]
* **Models.** Logistic Regression / SGD (baselines), **LightGBM** (primary); CatBoost optional; 1D-CNN optional for payloads.
* **Objective/constraints.** **Maximise TNR** at τ **subject to** TPR/FPR/latency; also report PR-AUC, ROC-AUC, F1.

---

## 2) Questions addressed

* **Benign pass-rate at fixed safety.** Which configurations achieve **highest TNR** under **TPR ≥ 0.90**, **FPR ≤ 0.005**, **p95 ≤ 10 ms**?
* **Effect of calibration/τ.** How much do **Isotonic calibration** and **constrained τ selection** reduce FPR at stable TPR?
* **Governance.** Can we **validate** HPO spaces and **audit** the exact hyperparameters used per training run?

---

## 3) Datasets

* **Primary:** **UNSW-NB15** (`archive/Payload_data_UNSW.csv`). [2]
* **Extension (may):** **CIC-IDS2017** for cross-dataset transfer. [1], [3], [9]
* **Labelling:** `normal` + `generic` → **benign (0)**; all other labels → **malicious (1)**.
* **Splitting:** stratified **train/validation/test**; **Isotonic CV** for calibration; final refit on train+val, report on test.
* **Tooling:** CICFlowMeter for flow features when processing CIC PCAPs. [8]

---

## 4) Alignment

* **SOC/DFIR (must).** High **TNR** reduces benign alert noise; **TPR floor** prevents coverage loss.
* **MLOps (must).** Deterministic artefacts, **model card**, **trials log**, **hyperparameter audit JSONL**, and **fail-fast space validation**.
* **Deployment (must).** Latency bounded on CPU; artefacts written to `ids_artifacts/`.

**Key artefacts**

| Artefact                                  | Path                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| Calibrated model                          | `ids_artifacts/model.pkl`                                              |
| Preprocessor / scaler (ColumnTransformer) | `ids_artifacts/scaler.pkl`                                             |
| Label encoder (if used)                   | `ids_artifacts/label_encoder.pkl`                                      |
| Metadata (incl. τ and metrics)            | `ids_artifacts/metadata.json`                                          |
| Trials log (BO/HPO)                       | `ids_artifacts/trials.parquet` *(CSV fallback if Parquet unavailable)* |
| Hyperparameter audit log (JSONL)          | `ids_artifacts/audit_hyperparams.jsonl`                                |
| Model card                                | `ids_artifacts/model_card.json`                                        |

---

## Proposed project blueprint (updated)

1. **Problem framing & KPIs (must).** **Objective:** maximise **TNR**. **Constraints:** **TPR ≥ 0.90**, **FPR ≤ 0.005**, **p95 ≤ 10 ms**; secondary: macro-F1, PR-AUC, ROC-AUC.
2. **Data engineering (must).** Load UNSW CSV; feature engineering; ColumnTransformer (scale + OHE). Optionally process CIC PCAPs with CICFlowMeter. [1], [3], [8]
3. **Model space (should).** LR, SGD, **LightGBM** primary; CatBoost/1D-CNN optional.
4. **HPO schemes (should).** Grid (compact baselines), Random (wider spaces), **BO** (LightGBM + τ). [4], [5]
5. **BO setup (must).** **GP (Matern-5/2)** surrogate, **NoisyEI** acquisition; **barrier objective** for TPR/FPR/latency; **best feasible** selection; **trials logged** to `TRIALS_PATH`. [4]–[6]
6. **Calibration & τ (must).** **Isotonic** calibration; constrained τ sweep to meet TPR/FPR; choose τ maximising TNR among feasible thresholds.
7. **Evaluation & diagnostics (should).** ROC/PR, calibration curve, confusion matrices at τ, threshold sweeps, **p95 latency**, feature importances.
8. **Audit & governance (must).**

   * **Validation cell:** `validate_hpo_spaces(HPO_SPACES)` checks keys match estimator parameters and `set_params` succeeds.
   * **Audit logger:** `log_effective_hyperparams(...)` prints and appends **effective hyperparameters** (BASE/HPO/BO context, τ, seed, CV, constraints) to JSONL.
9. **Export (must).** Save model, scaler, encoder, metadata, trials, audit to `ids_artifacts/`.

---

## Reliability & performance controls

* **Fail-fast HPO configuration.** Validation **must** pass before running HPO/BO.
* **Latency benchmarking.** **p95** per-sample latency measured on validation subsets; enforced in BO constraints.
* **Resource safety.** Optional row capping, dtype down-casting, thread limits, closed figures after plotting.
* **Resilient logging.** Trials saved to Parquet (CSV fallback) and **JSONL audit** for each fitted run.

---

## Explanation (plain language)

We first ensure the model’s probability scores are **well calibrated**, then find a **threshold τ** that keeps **false alarms** low while still catching **≥ 90%** of attacks and staying fast on CPU. **Bayesian Optimisation** chooses hyperparameters (and τ) **efficiently**, focusing compute on promising regions while **respecting constraints**. We persist a **model card**, artefacts, trials, and an **audit log** so operations can reproduce and govern the detector.

---

## Mathematical/statistical notes (used in code)

* **Calibration:** Isotonic regression on CV folds.
* **Operating metrics at τ:** **TPR**, **FPR**, **TNR**; plus PR-AUC, ROC-AUC, F1.
* **Constrained τ selection:** τ ∈ \[0,1] sweep; **feasible** if TPR ≥ 0.90 and FPR ≤ 0.005; among feasible, **maximise TNR**.
* **Bayesian Optimisation:** Gaussian Process surrogate with **Matern-5/2** kernel; **Noisy Expected Improvement** acquisition. [4], [5]
* **Constraint handling in BO:** **Barrier objective** assigns very large loss to infeasible points; among feasible, optimiser minimises $-\mathrm{TNR}$. [6], [7]

---

[1]: https://www.kaggle.com/datasets/yasiralifarrukh/unsw-and-cicids2017-labelled-pcap-data/code?utm_source=chatgpt.com
[2]: https://research.unsw.edu.au/projects/unsw-nb15-dataset?utm_source=chatgpt.com
[3]: https://www.unb.ca/cic/datasets/ids-2017.html?utm_source=chatgpt.com
[4]: https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf?utm_source=chatgpt.com
[5]: https://arxiv.org/pdf/1807.02811?utm_source=chatgpt.com
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485113/?utm_source=chatgpt.com
[7]: https://arxiv.org/abs/2403.12948?utm_source=chatgpt.com
[8]: https://github.com/ahlashkari/CICFlowMeter?utm_source=chatgpt.com
[9]: https://zenodo.org/records/7258579?utm_source=chatgpt.com
