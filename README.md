# Label_Trainer — Intrusion Detection Model Pipeline

## Non‑technical explanation
This project trains a machine‑learning model to spot suspicious network activity (like lateral movement or zero‑day‑style behavior) from flow/payload statistics. We feed past examples of **benign** vs **attack** traffic to the model so it can learn patterns, then set a **decision threshold (τ)** that balances catching more true attacks with keeping false alarms low.

Our model will be extended to an IDS system that identifies unknown cyber threats in network data, to detect suspicious activity on a computer network. It analyses patterns in the UNSW-NB15 dataset, distinguishing between normal (benign) traffic and attacks. By using Bayesian Optimization, we fine-tuned a LightGBM model to achieve high accuracy (PR-AUC: ~0.95), detecting most attacks while minimizing false alerts. A neural network (CNN) was also tested, but it performed slightly worse. The model’s results, visualized through charts, help security teams prioritize real threats, enhancing network safety with reliable, automated detection.

## Data
- Original dateset obtained from `UNSW-NB15 and CIC-IDS2017 Labelled PCAP Data` (https://www.kaggle.com/datasets/yasiralifarrukh/unsw-and-cicids2017-labelled-pcap-data/code/data)
- Source file expected at: `archive/Payload_data_UNSW.csv` (tabular flow/payload features with binary labels).
- Features are aggregated statistics per flow/session; no raw packet contents or personal identifiers are included.
- Train/validation/test splits are **stratified** to preserve class balance. Intermediate splits and artifacts are stored under `staging/` for reproducibility.

## Model
- **Primary:** LightGBM classifier (gradient‑boosted trees) chosen for strong tabular performance, speed, and native handling of non‑linear interactions.
- **Secondary (optional):** CNN baseline for derived representations.
- **Decision rule:** predict attack when \( $\hat p \ge \tau$ \); **τ** is chosen to **maximize F1** on the validation set.

## Hyperparameter optimisation
- **Manual grid (parallel):** Exploratory search with `joblib.Parallel` (multi‑threaded).
- **Bayesian optimization:** `skopt` (BayesSearchCV / ask‑tell) with checkpointing and resume.
- Global concurrency via `N_THREADS`; models set `n_jobs=-1` when supported.

## Results

- **AUPRC (test):** 0.9997634333419292
- **F1@τ (test):** 0.9997028988582828 (τ = 0.05)
- **Precision@τ / Recall@τ / TNR@τ:** … / … / …

Artifacts to consult:
- `staging/bo_lgb/manifest.json` — best LightGBM config, τ, and validation metrics.
- `staging/manual_grid/` — grid search history.
- `staging/cnn_best.keras` — best CNN weights (if used).

## Profile
https://www.linkedin.com/in/maxchowhk
