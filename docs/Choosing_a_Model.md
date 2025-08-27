# Model Choice for Benign Traffic Trainer

## 1) Problem Type
- Binary classification.
- Goal: predict whether a network flow is benign (0) or non-benign/attack (1).

## 2) What the notebook implements
- Data pipeline: ColumnTransformer with imputation, scaling for numeric features, and one-hot encoding for categoricals.
- Class imbalance handling: stratified splits; models use class_weight="balanced" (where supported).
- Evaluation: cross-validation scored by Average Precision (PR-AUC); test-time threshold selection for F1 or a target precision.

## 3) Candidate models in the notebook

### Logistic Regression (with L2 or elastic-net penalty)
- Benefits: fast CPU inference; well-calibrated probabilities; interpretable coefficients.
- Drawbacks: linear decision boundary can underfit complex patterns.
- Typical hyperparameters (2–5):
  - penalty (l2 or elasticnet), C (inverse regularization), l1_ratio (if elasticnet), solver.

### SGDClassifier (log loss, linear model trained with SGD)
- Benefits: scalable to large data; supports class weighting; streaming-friendly.
- Drawbacks: sensitive to learning-rate and regularization settings; may need feature scaling and tuning.
- Typical hyperparameters (3–6):
  - loss (log_loss), alpha (regularization), penalty, learning_rate schedule, eta0, max_iter.

### RandomForestClassifier
- Benefits: models non-linearities and interactions; robust to outliers; limited preprocessing needs.
- Drawbacks: larger memory footprint; slower inference than linear models; less calibrated by default.
- Typical hyperparameters (3–6):
  - n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.

### Gradient Boosting (e.g., GradientBoostingClassifier)
- Benefits: strong accuracy on tabular data; captures complex interactions; good bias–variance control.
- Drawbacks: more tuning needed; slower to train than linear models; probability calibration sometimes required.
- Typical hyperparameters (4–8):
  - n_estimators, learning_rate, max_depth (or max_leaf_nodes), subsample, min_samples_leaf, min_samples_split.

### (Optional) Bayesian Optimization wrapper (BayesSearchCV) around the above
- Benefits: sample-efficient model selection across hyperparameters.
- Drawbacks: adds a dependency and orchestration overhead.
- Tunables: the same model hyperparameters; BO controls like n_iter, init_points.

## 4) Benefits and drawbacks summary
- Linear (Logistic, SGD): fast, simple, explainable; may underfit; best when features are informative after scaling and encoding.
- Tree ensembles (RF, GB): higher accuracy on complex tabular patterns; more tunables; heavier at inference; may need calibration.

## 5) Model selection approach used
- Compare Logistic, SGD, Random Forest, and Gradient Boosting with cross-validation on PR-AUC.
- Pick the best by validation PR-AUC; then calibrate (if configured) and choose a decision threshold on the test set:
  - Either max-F1 threshold, or the smallest threshold achieving a target precision (for alert-budget control).

## 6) Hyperparameter count guidance (keeping within 2–8 tunables)
- Logistic Regression: 2–3 primary tunables (penalty, C, optionally l1_ratio).
- SGDClassifier: 3–6 tunables (alpha, penalty, learning-rate schedule, eta0, max_iter).
- Random Forest: 3–5 tunables (n_estimators, max_depth, min_samples_leaf, max_features).
- Gradient Boosting: 5–7 tunables (n_estimators, learning_rate, max_depth or leaves, min_samples_leaf, subsample).
- This keeps searches tractable while giving enough capacity to improve performance.

## 7) Recommended choice and rationale
- Start with Gradient Boosting as the main candidate for best accuracy on tabular IDS features, within a 5–7 hyperparameter budget.
- Keep Logistic Regression and SGD as light baselines for speed and interpretability.
- Use model selection (cross-validated PR-AUC) to confirm the winner on your dataset slice, then finalize thresholding to satisfy precision/recall objectives.
