# Hyperparameter Optimisation Strategy

## Problem Context
Our task is binary classification: detecting whether a network flow is benign (0) or non-benign/attack (1) using datasets like UNSW-NB15 and CIC-IDS2017.  
Model performance depends heavily on hyperparameters (learning rate, max depth, class weights, threshold τ, etc.). Choosing them correctly is critical to balance true positive rate (TPR) and false positive rate (FPR) while ensuring CPU efficiency.

---

## Approaches Considered

### 1. Grid Search
- How it works: Exhaustively tries all combinations of predefined hyperparameter values.  
- Benefits: Simple, guarantees coverage of specified parameter space.  
- Drawbacks: Computationally expensive; scales poorly with many parameters.  
- Fit here: Useful for categorical/discrete hyperparameters (for example, model type: LightGBM, CatBoost, or LogisticRegression).  

### 2. Random Search
- How it works: Randomly samples hyperparameter combinations from distributions.  
- Benefits: More efficient than grid search; can find good regions quickly.  
- Drawbacks: Noisy results; may miss optimal regions.  
- Fit here: Good for quick exploration, but less sample-efficient than Bayesian Optimisation.  

### 3. Bayesian Optimisation (BO) (preferred in Benign_Trainer.ipynb)
- How it works: Uses a surrogate model (Gaussian Process with Matern kernel) to model the performance surface. Iteratively selects promising hyperparameters via Expected Improvement (EI).  
- Benefits:  
  - Sample-efficient, requires fewer evaluations.  
  - Can incorporate constraints (for example, FPR ≤ 0.5%, latency ≤ 10ms).  
  - Handles noisy metrics (cross-validation results).  
- Drawbacks: More complex implementation; sensitive to kernel/prior assumptions.  
- Fit here: Best for continuous hyperparameters like learning rate, regularisation strength, or threshold τ. Already implemented in Benign_Trainer_3.ipynb.

---

## Combined Strategy
- Categorical hyperparameters (model type, sampling strategy): Grid Search.  
- Continuous hyperparameters (learning rate, τ, class weights): Bayesian Optimisation.  
- This hybrid ensures we explore model families broadly while fine-tuning numeric parameters efficiently.

---

## Number of Tunable Hyperparameters
- LightGBM/CatBoost models in this workflow typically expose 5–7 tunable hyperparameters:  
  - learning_rate  
  - max_depth  
  - num_leaves (LightGBM)  
  - min_child_samples  
  - subsample  
  - colsample_bytree  
  - class_weight  
- Logistic Regression / SGD has 2–3 tunable hyperparameters:  
  - C (inverse regularisation)  
  - penalty  
  - class_weight  

This falls within the recommended range (2–8 hyperparameters), making Bayesian Optimisation a strong choice.

---

## Final Decision
We adopt a Bayesian Optimisation pipeline with feasibility constraints, supplemented by grid search for categorical options.  
This balances:  
- Efficiency (fewer evaluations needed)  
- Robustness (safe exploration within FPR/latency constraints)  
- Interpretability (clear record of best hyperparameters and trade-offs).

---

# Hyperparameter Optimisation: Decision Matrix

| Method              | When To Use                                        | Benefits                                                   | Drawbacks                                                    | Typical Use In Our IDS |
|---------------------|-----------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------|------------------------|
| Grid Search         | Small, discrete spaces; few parameters (<= 4)       | Simple, exhaustive on a small grid; reproducible           | Scales poorly; wastes trials on unimportant dimensions       | Sweep categorical choices (e.g., model family, penalty) |
| Random Search       | Medium/large spaces; quick baseline                 | Covers wide spaces fast; good early wins                   | Noisy; may miss narrow optima; less sample efficient         | Coarse pass before BO for RF/GB params                   |
| Bayesian Optimisation (BO) | Expensive models/metrics; continuous params; limited budget | Sample efficient; focuses on promising regions; handles noise | More complex; needs surrogate choice; some overhead          | Main tuner for LR/SGD/GB continuous params, threshold tau |

## Notes
- Categorical hyperparameters (e.g., model type, penalty, sampling strategy) are best handled by **Grid Search** or a small manual list.
- Continuous hyperparameters (e.g., learning_rate, C, alpha, max_depth, num_leaves, tau) are best tuned with **Bayesian Optimisation**.
- A practical hybrid: **Grid over categorical** x **BO over continuous**.
- Keep total tunables in the range **2 to 8** to maintain a tractable search.

## Recommended Workflow
1. Fix categorical choices with a tiny grid (1–3 options each).
2. Run BO on continuous parameters (budgeted iterations).
3. Validate with cross-validation PR-AUC; finalize threshold tau on test set for target precision or max-F1.

