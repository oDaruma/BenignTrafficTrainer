# Bayesian Optimisation (BO) for Hyperparameter Tuning — Math & Visual Guide (with Acquisition Strategy Comparison)

This guide explains, step by step, how Bayesian Optimisation (BO) tunes hyperparameters in models like LightGBM for intrusion detection. It references maths introduced in the Imperial PCMLAI modules:
- Module 3: Probabilistic Modelling (Gaussian processes, kernels, Gaussian pdf/cdf).
- Module 10: Ensembles and Optimisation (practical tuning, black-box optimisation).
- RL to Multi-armed Bandits: exploration vs exploitation framing.

The examples align with the Benign_Trainer notebooks (LightGBM/LogReg/SGD, PR-AUC, threshold selection).

---

## 1. Problem Setup

We choose hyperparameters θ (for example, learning_rate, max_depth, num_leaves) to maximise a validation objective:
J(θ) = PR-AUC from cross-validation.

Why this is hard:
- Expensive: each evaluation trains a model and runs CV.
- Noisy: CV folds vary; scores have variance.
- Non-convex: many local optima.
- Small budget: typically 25–50 evaluations.

This is a black-box optimisation problem (course capstone framing).

---

## 2. BO in Plain Words

BO repeats the following loop:
1) Evaluate a few configurations θ and record scores J(θ).
2) Fit a surrogate model that predicts J(θ) for unseen θ, with uncertainty.
3) Use an acquisition function to choose the next θ, trading off exploration vs exploitation.
4) Repeat until the evaluation budget is exhausted.

Result: Stronger hyperparameters with fewer evaluations than grid or random search.

---

## 3. Core Maths (Module 3, Module 10)

### 3.1 Gaussian Process (GP) Surrogate

Given data D_t = { (θ_i, y_i) } with y_i = J(θ_i) + ε_i, a GP posterior at θ provides:
- Posterior mean (predicted score): μ_t(θ)
- Posterior variance (uncertainty): σ_t^2(θ)

A Matérn kernel is commonly used (Module 3). Intuition: μ_t(θ) is the best guess, σ_t(θ) is how unsure we are.

### 3.2 Acquisition Functions (Module 10; Bandits framing)

Given current best f*:
- Expected Improvement (EI):
  EI(θ) = (μ_t − f* − ξ) Φ(z) + σ_t φ(z),
  z = (μ_t(θ) − f* − ξ) / σ_t(θ)
- Upper Confidence Bound (UCB):
  UCB(θ) = μ_t(θ) + κ σ_t(θ)
- Probability of Improvement (PI):
  PI(θ) = Φ( (μ_t(θ) − f* − ξ) / σ_t(θ) )

Here Φ and φ are the standard normal CDF and PDF. ξ ≥ 0 encourages exploration; κ > 0 weights uncertainty. These encode exploration vs exploitation mathematically (reinforcement learning bandits perspective).

---

## 4. Visualising Surrogate and EI

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Toy 1D "true" objective (unknown to BO)
def f(x):
    return np.sin(3*x) + 0.5*np.cos(5*x)

X = np.linspace(0, 2, 400)
Y_true = f(X)

# Pretend surrogate predictions (mean and uncertainty)
mu = 0.8*np.sin(3*X)
sigma = 0.2 + 0.3*np.abs(np.cos(2*X))

# Acquisition: Expected Improvement
f_best = 0.5
z = (mu - f_best) / np.maximum(sigma, 1e-9)
EI = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(X, Y_true, 'k--', label="True function (unknown)")
plt.plot(X, mu, 'b', label="Surrogate mean μ")
plt.fill_between(X, mu - sigma, mu + sigma, color='b', alpha=0.2, label="Uncertainty σ")
plt.axhline(f_best, color='r', ls='--', label="Best so far f*")
plt.legend()
plt.title("GP Surrogate: mean μ and uncertainty σ")

plt.subplot(2,1,2)
plt.plot(X, EI, 'g', label="Expected Improvement (EI)")
x_next = X[np.argmax(EI)]
plt.axvline(x_next, color='g', ls='--', label=f"Next θ (EI argmax) ≈ {x_next:.3f}")
plt.legend()
plt.title("Acquisition Function (EI) and next suggested point")
plt.xlabel("Hyperparameter θ (toy 1D)")
plt.tight_layout()
plt.show()
````

Interpretation:

* Surrogate mean μ approximates the unknown objective; σ widens where the model is uncertain.
* EI is large either where μ is high (exploitation) or σ is large (exploration).
* BO picks the next θ where EI is maximal.

---

## 5. Comparing EI, UCB, and PI on the Same Surrogate

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

X = np.linspace(0, 2, 400)
mu = 0.8*np.sin(3*X)
sigma = 0.2 + 0.3*np.abs(np.cos(2*X))
f_best = 0.5

# EI
z = (mu - f_best) / np.maximum(sigma, 1e-9)
EI = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)

# UCB with kappa
kappa = 1.5
UCB = mu + kappa * sigma

# PI with xi
xi = 0.0
PI = norm.cdf((mu - f_best - xi) / np.maximum(sigma, 1e-9))

# Argmax locations
x_ei = X[np.argmax(EI)]
x_ucb = X[np.argmax(UCB)]
x_pi = X[np.argmax(PI)]

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(X, mu, 'b', label="μ")
plt.fill_between(X, mu - sigma, mu + sigma, color='b', alpha=0.2, label="σ band")
plt.axhline(f_best, color='r', ls='--', label="f*")
plt.legend()
plt.title("Surrogate mean μ and uncertainty σ")

plt.subplot(3,1,2)
plt.plot(X, EI, 'g', label="EI")
plt.axvline(x_ei, color='g', ls='--', label=f"EI argmax ≈ {x_ei:.3f}")
plt.legend()
plt.title("Expected Improvement (EI)")

plt.subplot(3,1,3)
plt.plot(X, UCB, 'm', label=f"UCB (κ={kappa})")
plt.plot(X, PI, 'c', label=f"PI (ξ={xi})")
plt.axvline(x_ucb, color='m', ls='--', label=f"UCB argmax ≈ {x_ucb:.3f}")
plt.axvline(x_pi, color='c', ls='--', label=f"PI argmax ≈ {x_pi:.3f}")
plt.legend()
plt.title("UCB and PI (different acquisition preferences)")
plt.xlabel("Hyperparameter θ (toy 1D)")
plt.tight_layout()
plt.show()
```

What to notice:

* EI often selects points with a good mix of high μ and high σ.
* UCB prefers more optimistic points as κ increases (more exploration weight).
* PI is sensitive to the current best f\*; it can cluster near regions already “good enough” unless ξ is increased.

Course linkage:

* Module 3 provides the Gaussian pdf/cdf background used in EI and PI.
* Module 10 covers practical optimisation trade-offs.
* Bandits module gives the explore-exploit intuition that κ and ξ encode.

---

## 6. IDS-Specific Objective and Thresholding

After BO suggests hyperparameters and the model is fitted:

* Compute test probabilities.
* Choose threshold τ:

Max-F1 rule:
F1(τ) = 2 \* Precision(τ) \* Recall(τ) / (Precision(τ) + Recall(τ))

Target-precision rule:
Find the smallest τ such that Precision(τ) ≥ P\_target (for example, 0.95).

Report TPR (recall), TNR, Precision, and F1 at the chosen τ.

---

## 7. Example: LightGBM with BayesSearchCV

```python
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from sklearn.metrics import precision_recall_curve

param_space = {
    "num_leaves": (24, 96),
    "max_depth": (3, 10),
    "learning_rate": (1e-3, 0.2, "log-uniform"),
    "n_estimators": (100, 500),
    "min_child_samples": (20, 200)
}

opt = BayesSearchCV(
    estimator=LGBMClassifier(random_state=42),
    search_spaces=param_space,
    n_iter=30,
    cv=3,
    n_jobs=-1,
    random_state=42,
    scoring="average_precision"
)

opt.fit(X_train, y_train)
print("Best params:", opt.best_params_)
print("Best CV PR-AUC:", opt.best_score_)

# Threshold selection by max-F1
y_proba = opt.best_estimator_.predict_proba(X_test)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, y_proba)
f1 = 2*prec[:-1]*rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
tau = float(thr[f1.argmax()])
print("tau (max-F1) =", tau)
```

---

## 8. Checklist

* Tune 2 to 8 hyperparameters (keeps search tractable).
* Use PR-AUC for imbalanced IDS.
* BO for continuous parameters; tiny grid for categorical choices.
* Refit and choose τ; report TPR, TNR, Precision, F1.

---

## 9. Summary

BO = GP surrogate (μ, σ) + acquisition (EI, UCB, PI).

* Module 3: Gaussian processes, kernels, Gaussian pdf/cdf.
* Module 10: practical optimisation of models.
* Bandits module: exploration vs exploitation.
  The result is sample-efficient hyperparameter search that improves IDS detection with fewer evaluations.


