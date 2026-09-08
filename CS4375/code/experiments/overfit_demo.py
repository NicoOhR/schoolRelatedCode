import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from models.criteria import MSEHessLoss
from models.xg_boost import XGBoost

SEED      = 42
LR        = 0.1
OVF_ITERS = 30
OVF_DEPTH = 6
np.random.seed(SEED)

# Last run results (depth=6, n=100, iters=30, normalised target):
#   No reg  train=0.0560  test=0.6715  (12× train/test gap — massive overfit)
#   λ=3     train=0.2268  test=0.6308  (leaf shrinkage reduces variance)
#   α=1     train=0.3354  test=0.6145  (L1 prunes low-gradient nodes)
#   γ=1     train=0.4184  test=0.5530  (min-gain threshold, also fastest per iter)


def rmse(y_true, y_pred):
    return np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean())


# ── data ──────────────────────────────────────────────────────────────────────
X, y = make_regression(
    n_samples=100, n_features=20, n_informative=5, noise=20, random_state=SEED
)
y = (y - y.mean()) / y.std()

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y
train, test = train_test_split(df, test_size=0.2, random_state=SEED)
y_train = train["target"].to_numpy()
y_test  = test["target"].to_numpy()

# ── models ────────────────────────────────────────────────────────────────────
models = {
    "No reg": XGBoost("target", iterations=OVF_ITERS, learning_rate=LR, tree_depth=OVF_DEPTH, loss=MSEHessLoss(lambda_reg=0.0, alpha_reg=0.0), gamma=0.0, verbose=True),
    "λ=3":    XGBoost("target", iterations=OVF_ITERS, learning_rate=LR, tree_depth=OVF_DEPTH, loss=MSEHessLoss(lambda_reg=3.0, alpha_reg=0.0), gamma=0.0, verbose=True),
    "α=1":    XGBoost("target", iterations=OVF_ITERS, learning_rate=LR, tree_depth=OVF_DEPTH, loss=MSEHessLoss(lambda_reg=0.0, alpha_reg=1.0), gamma=0.0, verbose=True),
    "γ=1":    XGBoost("target", iterations=OVF_ITERS, learning_rate=LR, tree_depth=OVF_DEPTH, loss=MSEHessLoss(lambda_reg=0.0, alpha_reg=0.0), gamma=1.0, verbose=True),
}

# ── fit ───────────────────────────────────────────────────────────────────────
for name, model in models.items():
    print(f"\nFitting {name}...")
    model.fit(train)

# ── metrics ───────────────────────────────────────────────────────────────────
rmses_train = {n: rmse(y_train, m.predict(train)) for n, m in models.items()}
rmses_test  = {n: rmse(y_test,  m.predict(test))  for n, m in models.items()}

print("\nRegularization Comparison — Train / Test RMSE")
for name in models:
    print(f"  {name:8s}  train={rmses_train[name]:.4f}  test={rmses_test[name]:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
os.makedirs("plots/overfit", exist_ok=True)

names = list(models.keys())
x = np.arange(len(names))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w / 2, [rmses_train[n] for n in names], w, label="train", color="#6baed6")
ax.bar(x + w / 2, [rmses_test[n]  for n in names], w, label="test",  color="#fd8d3c")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_title(f"Regularization Comparison  (depth={OVF_DEPTH}, n=300, synthetic)")
ax.set_ylabel("RMSE  (normalised target)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/overfit/train_vs_test.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved plots/overfit/train_vs_test.png")
