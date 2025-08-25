# BenignTrafficTrainer
Train models to recognise benign network traffics accurately (high TNR/specificity)

# Reflection: Applying Bayesian Optimisation (BO) to Network Intrusion Detection — **Benign-First Training for Non-Benign Identification**

## Executive summary (non-technical)

- **Goal.** Train detectors to **model benign traffic precisely** (high TNR) so we can **flag non-benign** whenever the model’s benign probability falls below a threshold τ. We **must** still maintain high attack detection (TPR ≥ 90%) and low latency on CPU. Corpora: **UNSW-NB15** and **CIC-IDS2017** labelled PCAPs. ([Kaggle][1], [UNSW Sites][2], [unb.ca][3])  
- **Why BO.** BO **must** tune hyper-parameters, calibration, sampling ratios and τ under noisy/expensive evaluation with constraints (FPR cap, latency). It **should** prefer safe/constrained exploration and **may** reuse priors between datasets for sample efficiency. ([NeurIPS Proceedings][4], [arXiv][5], [PMC][6])  
- **Outcome.** A CPU-deployable detector that **passes benign** (minimises false alerts) and **signals non-benign** reliably at a bounded **FPR** and **p95 latency**, with explainable outputs ready for SOC ingestion.

---

## 1) How the BO skills/code transfer (technical)

### A. Direct applications (FN-aware but **benign-first** modelling)

| BO skill | IDS use-case | **Optimisation target & constraints** | Key refs |
| --- | --- | --- | --- |
| GP surrogate + EI/UCB/KG | Hyper-parameter tuning (LightGBM/CatBoost/SGD, optional 1D-CNN) | **Objective:** maximise **TNR** (benign pass-rate). **Constraints:** **TPR ≥ 0.90**, **FPR ≤ 0.5%**, **p95 latency ≤ 10 ms**. Use **Noisy-EI**. | [4],[5] |
| Noisy BO + replications | Stability under label noise & drift | 3–5× GroupKFold or repeated splits; fold variance → GP noise | [5] |
| **Constrained / Safe BO** | Keep search within SOC limits | **SafeOpt/feasible filters:** reject configs violating latency/FPR constraints | [6],[7] |
| Threshold & calibration BO | Operating point selection | Calibrate (Platt/Isotonic), then BO over τ with **FPR cap**; objective **TNR↑** under **TPR floor** | [5] |
| Mixed discrete/continuous | End-to-end pipeline choice | Search over model family, class-weights, sampling, features | [4] |
| Transfer/meta-BO | UNSW ↔ CIC generalisation | Warm-start from source dataset; compare TNR/TPR stability | [3] |

**Why benign-first?** In production SIEM/SOAR, benign dominates. If the model **knows benign well**, anything that deviates (low benign probability) is a **non-benign candidate**. We still **must** uphold an attack-TPR floor to avoid missing true attacks.

### B. What is optimised

- **Features.** CICFlowMeter-style flow stats, byte histograms, temporal aggregates; **time-grouped** CV to avoid leakage. ([3],[8])  
- **Models.** Baselines (SGD/LogReg), LightGBM/CatBoost; optional 1D-CNN on payload bytes if present.  
- **Objective/constraints.** **Maximise TNR** under **TPR ≥ 0.90**, **FPR ≤ 0.5%**, **p95 latency ≤ 10 ms**.  
- **Explainability.** SHAP stability on benign motifs; aide for triage playbooks.  

---

## 2) Questions addressed

- **Benign pass-rate at fixed safety.** Which configurations yield **highest TNR** while keeping **TPR ≥ 0.90**, **FPR ≤ 0.5%**, **p95 latency ≤ 10 ms**?  
- **Cross-dataset robustness.** Do UNSW-tuned τ and calibrations hold on CIC with acceptable TNR/TPR?  
- **Alert budget discipline.** What benign false-alert reduction is achieved at the set FPR cap?  

---

## 3) Datasets

- **Primary:** **UNSW-NB15 & CIC-IDS2017 Labelled PCAPs** (Kaggle/Zenodo). Integer matrices (e.g., **N×1504**) aligned to official CSVs. **Must** be used. ([1],[9])  
- **Metadata:** **UNSW** official description; **CIC-IDS2017** official page; **CICFlowMeter** tooling. ([2],[3],[8])  

**Splitting:** **GroupKFold by day/pcap/session** to prevent temporal leakage. UNSW↔CIC out-of-domain testing for generalisation.

---

## 4) Alignment

- **SOC/DFIR (must).** High TNR reduces benign alert noise; TPR floor reduces missed attacks.  
- **MLOps (should).** Constrained/safe BO, model cards, reproducible CV.  
- **Deployment (must).** CPU latency bounds, τ at FPR cap, calibration stability.  

---

## Proposed project blueprint (updated)

1. **Problem framing & KPIs (must).** **Objective:** maximise **TNR (benign pass-rate)**. **Constraints:** **TPR ≥ 0.90**, **FPR ≤ 0.5%**, **p95 latency ≤ 10 ms**; secondary: macro-F1, PR-AUC.  
2. **Data engineering (must).** Load labelled PCAPs; derive flow/payload features; de-dup; **time-grouped CV**. ([1],[3])  
3. **Model space (should).** SGD/LogReg; LightGBM/CatBoost; optional 1D-CNN (payload).  
4. **BO setup (must).** GP-Matern(5/2) + **Noisy-EI**; batch 4–8; **feasibility filters** for FPR/latency/TPR. ([4],[5],[6])  
5. **Imbalance handling (must).** BO over class-weights/sampling ratios with benign-first objective and TPR floor.  
6. **Calibration & τ (should).** Platt vs Isotonic; τ picked under **FPR cap**.  
7. **Cross-dataset (must).** UNSW↔CIC transfer; report ΔTNR/ΔTPR and τ drift.  
8. **Explainability (may).** SHAP motifs → defensive heuristics/rules.  
9. **Deployment (should).** CPU perf profile; CI job for small BO refresh on rolling data.  

---

## Kernel-crash avoidance & reliability controls

- **Row cap for BO:** subsample to `MAX_TRAIN_ROWS` with stratification when datasets are huge.  
- **Aggressive dtype down-cast & constant-column drop.**  
- **Thread control:** limit OpenMP/BLAS threads to avoid oversubscription.  
- **Safe plotting:** skip heavy skopt plots if memory constrained; free figures (`plt.close()`).  
- **Checkpoint trials:** write a light **trials.parquet** to resume analysis after interruption.  
- **Guard optional deps:** only search installed model families.  

---

## Explanation

*We teach a computer what “normal (benign) internet traffic” looks like. If a new connection looks **unlike** benign, we treat it as **non-benign**.*

1. We try different model settings. **Bayesian Optimisation** helps us *choose smartly* rather than guessing randomly.  
2. We **calibrate** model scores so “0.8” really means “~80% chance benign”.  
3. We pick a **threshold** (τ) so that **false alarms on normal traffic** stay under our limit.  
4. We also check that **at least 90% of attacks** are still caught and that **predictions are fast** on a normal CPU.  
5. We save the best settings and a small “model card” (a factsheet) for the SOC team.  

---

## Mathematical/statistical notes (used in code; quick glossary)

- **Sigmoid:** maps any number to 0–1:  \( \sigma(z)=1/(1+e^{-z}) \).  
- **Calibration:** learn a mapping \( g(p) \) so calibrated probabilities match observed frequencies (Platt = logistic; Isotonic = monotone step-wise).  
- **Confusion matrix:** TN/FP/FN/TP counts; **TNR** \(= \text{TN}/(\text{TN}+\text{FP})\) (benign passed), **TPR** \(= \text{TP}/(\text{TP}+\text{FN})\).  
- **Percentiles:** p95 latency = time under which 95% of single-row predictions finish.  
- **Bayesian Optimisation:** build a surrogate (GP with Matern-5/2 kernel) over hyper-params; pick next point by **Expected Improvement (EI)**; handle noise by modelling observation variance.  
- **Feasible set:** we keep only configs with **TPR ≥ 0.90** and **latency ≤ 10 ms**; τ is chosen to satisfy **FPR cap**.  

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

