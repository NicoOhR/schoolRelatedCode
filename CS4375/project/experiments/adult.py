import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.criteria import LogLoss, LogLossHess
from models.gradient_boost import GradientBoost
from models.xg_boost import XGBoost

SEED        = 42
LR          = 0.1
ADULT_ITERS = 50
ADULT_DEPTH = 6   # deep enough to overfit visibly
ADULT_ROWS  = 5000
np.random.seed(SEED)

_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))
classify  = lambda lo: (_sigmoid(lo) >= 0.5).astype(int)


# ── data ──────────────────────────────────────────────────────────────────────
print("Loading Adult Census Income...")
raw = fetch_openml("adult", version=2, as_frame=True, parser="auto")
df = raw.data.copy()
for col in df.select_dtypes(["category", "object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df["target"] = (raw.target.astype(str).str.strip().str.startswith(">")).astype(int)

train, test = train_test_split(df, test_size=0.2, random_state=SEED)
train = train.sample(n=ADULT_ROWS, random_state=SEED).reset_index(drop=True)
X_train = train.drop(columns=["target"])
X_test  = test.drop(columns=["target"])
y_train = train["target"].to_numpy()
y_test  = test["target"].to_numpy()

print(f"Train: {len(train)} rows  |  Test: {len(test)} rows")
print(f"Class balance (test): {y_test.mean():.2%} positive")

# ── models ────────────────────────────────────────────────────────────────────
# λ=0 vs λ=10 gives the starkest leaf-weight difference; γ=2 shows pruning
models = {
    "Custom GBM":        GradientBoost("target", iterations=ADULT_ITERS, learning_rate=LR,
                                        tree_depth=ADULT_DEPTH, loss=LogLoss(), verbose=True),
    "Custom XGB (λ=0)":  XGBoost("target", iterations=ADULT_ITERS, learning_rate=LR,
                                  tree_depth=ADULT_DEPTH, loss=LogLossHess(lambda_reg=0.0),
                                  verbose=True),
    "Custom XGB (λ=10)": XGBoost("target", iterations=ADULT_ITERS, learning_rate=LR,
                                  tree_depth=ADULT_DEPTH, loss=LogLossHess(lambda_reg=10.0),
                                  verbose=True),
    "Custom XGB (γ=2)":  XGBoost("target", iterations=ADULT_ITERS, learning_rate=LR,
                                  tree_depth=ADULT_DEPTH, loss=LogLossHess(lambda_reg=0.0),
                                  gamma=2.0, verbose=True),
    "sklearn GBM":       GradientBoostingClassifier(n_estimators=ADULT_ITERS,
                                                     learning_rate=LR, max_depth=ADULT_DEPTH,
                                                     random_state=SEED),
}

# ── fit ───────────────────────────────────────────────────────────────────────
def _fit_task(args):
    name, model, fit_args = args
    model.fit(*fit_args)
    print(f"  {name}: done", flush=True)
    return model


tasks = [
    ("Custom GBM",        models["Custom GBM"],        (train,)),
    ("Custom XGB (λ=0)",  models["Custom XGB (λ=0)"],  (train,)),
    ("Custom XGB (λ=10)", models["Custom XGB (λ=10)"], (train,)),
    ("Custom XGB (γ=2)",  models["Custom XGB (γ=2)"],  (train,)),
    ("sklearn GBM",       models["sklearn GBM"],        (X_train, y_train)),
]

print("Fitting all models in parallel...")
with ProcessPoolExecutor() as executor:
    fitted = list(executor.map(_fit_task, tasks))

(
    models["Custom GBM"],
    models["Custom XGB (λ=0)"],
    models["Custom XGB (λ=10)"],
    models["Custom XGB (γ=2)"],
    models["sklearn GBM"],
) = fitted
print("Done.")

# ── metrics ───────────────────────────────────────────────────────────────────
accs_train = {}
accs_test  = {}
probs_test = {}

for name, model in models.items():
    if name == "sklearn GBM":
        accs_train[name] = (model.predict(X_train) == y_train).mean()
        accs_test[name]  = (model.predict(X_test)  == y_test).mean()
        probs_test[name] = model.predict_proba(X_test)[:, 1]
    else:
        accs_train[name] = (classify(model.predict(train)) == y_train).mean()
        accs_test[name]  = (classify(model.predict(test))  == y_test).mean()
        probs_test[name] = _sigmoid(model.predict(test))

print("\nAdult Census Income — Train / Test Accuracy")
for name in models:
    gap = accs_train[name] - accs_test[name]
    print(f"  {name:20s}  train={accs_train[name]:.4f}  test={accs_test[name]:.4f}  gap={gap:+.4f}")

# ── plots ─────────────────────────────────────────────────────────────────────
os.makedirs("plots/adult", exist_ok=True)

_pal = ["#4878cf", "#e24a33", "#6acc65", "#b47cc7", "#999999"]

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for (name, proba), color in zip(probs_test.items(), _pal):
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves  (Adult Census Income, depth={ADULT_DEPTH})")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("plots/adult/roc.png", dpi=150, bbox_inches="tight")
plt.close()

# Precision-Recall curves (more informative for imbalanced classes)
fig, ax = plt.subplots(figsize=(7, 6))
for (name, proba), color in zip(probs_test.items(), _pal):
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(rec, prec, lw=2, color=color, label=f"{name}  (AP = {ap:.3f})")
baseline = y_test.mean()
ax.axhline(baseline, color="black", lw=1, ls="--", alpha=0.4, label=f"baseline ({baseline:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision–Recall Curves  (Adult Census Income, depth={ADULT_DEPTH})")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig("plots/adult/precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()

# Train vs Test accuracy — overfitting gap
fig, ax = plt.subplots(figsize=(9, 5))
names = list(models.keys())
x = np.arange(len(names))
w = 0.35
bars_tr = ax.bar(x - w/2, [accs_train[n] for n in names], w, label="train", color="#6baed6")
bars_te = ax.bar(x + w/2, [accs_test[n]  for n in names], w, label="test",  color="#fd8d3c")
# annotate gap on each pair
for i, name in enumerate(names):
    gap = accs_train[name] - accs_test[name]
    y_top = max(accs_train[name], accs_test[name])
    ax.text(i, y_top + 0.003, f"Δ{gap:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8, rotation=10)
ax.set_ylim(0.78, 1.03)
ax.set_title(f"Train vs Test Accuracy  (Adult Census Income, depth={ADULT_DEPTH})")
ax.set_ylabel("Accuracy")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/adult/train_vs_test.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nPlots saved to plots/adult/")
