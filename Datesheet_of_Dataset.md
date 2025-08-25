# Datasheet for Dataset: UNSW-NB15 & CIC-IDS2017 Payload Data

## 1. Motivation
- **Purpose.** To support research and development of **intrusion detection systems (IDS)** that model benign network traffic and detect anomalies/attacks.  
- **Tasks enabled.** Classification (benign vs non-benign), anomaly detection, adversarial robustness studies.  
- **Why created?** Address the lack of **realistic, labeled intrusion traffic datasets** with both modern benign and attack flows.

---

## 2. Composition
- **UNSW-NB15**  
  - **Instances:** ~2,540,044 network flows.  
  - **Features:** 49 features + 1 label.  
  - **Disk size:** ~2.6 GB (CSV).  

- **CIC-IDS2017**  
  - **Instances:** ~2,830,743 network flows.  
  - **Features:** 80+ features + 1 label.  
  - **Disk size:** ~66 GB (PCAPs), ~13 GB (CSV).  

- **Each row:** One bidirectional flow with statistical and/or payload-based features.  
- **Labels:** `benign` vs specific attack categories (DoS, Botnet, Exploit, Reconnaissance, etc.).  
- **Errors/noise.** Labels derived via scripts + manual curation may contain mislabeled flows. Payload truncation can lose fine-grained context.  
- **Redundancy.** Benign flows dominate (>80% in CIC), leading to class imbalance.

---

## 3. Collection Process
- **Source.**  
  - UNSW: IXIA PerfectStorm traffic generator + tcpdump capture.  
  - CIC: Real user activity (browsing, email, chat, streaming) combined with scripted attacks.  
- **Collection period.**  
  - UNSW: 2015.  
  - CIC: 2017.  
- **Ethics.** No personally identifiable data (payloads are synthetic or anonymised). Institutional approvals documented.

---

## 4. Pre-Processing
- **Cleaning.**  
  - Non-numeric fields encoded (categorical → integers).  
  - Constant columns dropped.  
  - Payloads normalised to histograms.  
- **Splits.** GroupKFold by session/day to avoid temporal leakage.  
- **Tools.** CICFlowMeter, tcpdump, custom scripts.

---

## 5. Uses
- **Appropriate uses.**  
  - Intrusion detection training/benchmarking.  
  - Research on **Bayesian Optimisation** of IDS.  
  - Robustness and generalisation testing across datasets.  
- **Inappropriate uses.**  
  - User attribution or deanonymisation.  
  - Real-world deployment without retraining/validation.

---

## 6. Distribution
- **Availability & Download URLs.**  
  - **UNSW-NB15:**  
    - [Research UNSW site](https://research.unsw.edu.au/projects/unsw-nb15-dataset)  
    - [Kaggle mirror (UNSW-NB15 & CIC-IDS2017 PCAP Data)](https://www.kaggle.com/datasets/yasiralifarrukh/unsw-and-cicids2017-labelled-pcap-data)  
  - **CIC-IDS2017:**  
    - [Official UNB site](https://www.unb.ca/cic/datasets/ids-2017.html)  
    - [Zenodo mirror](https://zenodo.org/records/7258579)  
- **Licensing.** Free for academic/non-commercial use (cite source).

---

## 7. Maintenance
- **Hosting.** UNSW Canberra, Canadian Institute for Cybersecurity (UNB).  
- **Updates.** Static datasets; no new flows expected.  
- **Contact.** Dataset maintainers at UNSW/UNB.

---
