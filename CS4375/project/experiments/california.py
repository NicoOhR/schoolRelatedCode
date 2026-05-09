import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from models.criteria import MSEHessLoss
from models.gradient_boost import GradientBoost
from models.xg_boost import XGBoost

SEED      = 42
LR        = 0.1
CAL_ITERS = 30
CAL_DEPTH = 4
CAL_ROWS  = 5000
np.random.seed(SEED)


def rmse(y_true, y_pred):
    return np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean())


def annotate_bars(ax, bars, vals, fmt="{:.2f}", pad=None):
    for bar, val in zip(bars, vals):
        if pad is None:
            pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + pad,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ── data ──────────────────────────────────────────────────────────────────────
cal = fetch_california_housing()
df_cal = pd.DataFrame(cal.data, columns=cal.feature_names)
df_cal["target"] = cal.target
train_cal, test_cal = train_test_split(df_cal, test_size=0.2, random_state=SEED)
train_cal = train_cal.sample(n=CAL_ROWS, random_state=SEED).reset_index(drop=True)
X_train_cal = train_cal.drop(columns=["target"])
X_test_cal  = test_cal.drop(columns=["target"])
y_train_cal = train_cal["target"].to_numpy()
y_test_cal  = test_cal["target"].to_numpy()

# ── models ────────────────────────────────────────────────────────────────────
our_xgb_cal = XGBoost(
    "target",
    iterations=CAL_ITERS,
    learning_rate=LR,
    tree_depth=CAL_DEPTH,
    loss=MSEHessLoss(lambda_reg=1.0),
    subsample=0.8,
    verbose=True,
)
our_xgb_cal_unreg = XGBoost(
    "target",
    iterations=CAL_ITERS,
    learning_rate=LR,
    tree_depth=CAL_DEPTH,
    loss=MSEHessLoss(lambda_reg=0.0),
    verbose=True,
)
our_gbm_cal = GradientBoost(
    "target", iterations=CAL_ITERS, learning_rate=LR, tree_depth=CAL_DEPTH, verbose=True
)
sk_gbm_cal = GradientBoostingRegressor(
    n_estimators=CAL_ITERS, learning_rate=LR, max_depth=CAL_DEPTH, random_state=SEED
)

# ── fit ───────────────────────────────────────────────────────────────────────
print("Fitting GBM..."); our_gbm_cal.fit(train_cal)
print("Fitting XGB (λ=0)..."); our_xgb_cal_unreg.fit(train_cal)
print("Fitting XGB (λ=1)..."); our_xgb_cal.fit(train_cal)
print("Fitting sklearn GBM..."); sk_gbm_cal.fit(X_train_cal, y_train_cal)

# ── metrics ───────────────────────────────────────────────────────────────────
cal_models = {
    "Custom GBM":      (our_gbm_cal.predict(train_cal),        our_gbm_cal.predict(test_cal)),
    "Custom XGB (λ=0)":(our_xgb_cal_unreg.predict(train_cal),  our_xgb_cal_unreg.predict(test_cal)),
    "Custom XGB (λ=1)":(our_xgb_cal.predict(train_cal),        our_xgb_cal.predict(test_cal)),
    "sklearn GBM":     (sk_gbm_cal.predict(X_train_cal),        sk_gbm_cal.predict(X_test_cal)),
}
rmses_cal_train = {n: rmse(y_train_cal, tr) for n, (tr, _) in cal_models.items()}
rmses_cal_test  = {n: rmse(y_test_cal,  te) for n, (_, te) in cal_models.items()}

print("\nCalifornia Housing — Train / Test RMSE")
for name in cal_models:
    print(f"  {name:18s}  train={rmses_cal_train[name]:.4f}  test={rmses_cal_test[name]:.4f}")

# ── plots ─────────────────────────────────────────────────────────────────────
os.makedirs("plots/california", exist_ok=True)

# Predicted vs actual: 2×2 hexbin grid — handles dense point clouds without overplotting
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()
_lo = min(y_test_cal.min(), min(p.min() for _, (_, p) in cal_models.items())) - 0.1
_hi = max(y_test_cal.max(), max(p.max() for _, (_, p) in cal_models.items())) + 0.1
for ax, (name, (_, test_pred)) in zip(axes, cal_models.items()):
    hb = ax.hexbin(y_test_cal, test_pred, gridsize=40, cmap="Blues", mincnt=1)
    ax.plot([_lo, _hi], [_lo, _hi], "r--", lw=1, zorder=3)
    ax.set_xlim(_lo, _hi)
    ax.set_ylim(_lo, _hi)
    ax.set_title(f"{name}  (RMSE={rmses_cal_test[name]:.3f})", fontsize=10)
    ax.set_xlabel("actual  ($100k)")
    ax.set_ylabel("predicted  ($100k)")
    fig.colorbar(hb, ax=ax, label="count")
fig.suptitle("Predicted vs Actual  (California Housing)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/california/predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()

# Residuals: step-outline + light fill — no opacity mud
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-3, 3, 80)
_pal = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"]
for (name, (_, test_pred)), color in zip(cal_models.items(), _pal):
    residuals = test_pred - y_test_cal
    ax.hist(residuals, bins=bins, histtype="stepfilled", density=True,
            alpha=0.15, color=color)
    ax.hist(residuals, bins=bins, histtype="step", density=True,
            lw=2, color=color, label=name)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_title("Residual Distribution  (California Housing)")
ax.set_xlabel("predicted − actual  ($100k)")
ax.set_ylabel("density")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/california/residuals.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
names = list(cal_models.keys())
x = np.arange(len(names))
w = 0.35
ax.bar(x - w / 2, [rmses_cal_train[n] for n in names], w, label="train", color="#6baed6")
ax.bar(x + w / 2, [rmses_cal_test[n]  for n in names], w, label="test",  color="#fd8d3c")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_title(f"Train vs Test RMSE  (California Housing, depth={CAL_DEPTH})")
ax.set_ylabel("RMSE")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/california/train_vs_test.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    rmses_cal_test.keys(),
    rmses_cal_test.values(),
    color="#4878cf",
    edgecolor="white",
    width=0.5,
)
ax.set_title("Test RMSE  (California Housing)")
ax.set_ylabel("RMSE")
ax.tick_params(axis="x", rotation=15)
annotate_bars(ax, bars, list(rmses_cal_test.values()), fmt="{:.3f}")
plt.tight_layout()
plt.savefig("plots/california/rmse.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nPlots saved to plots/california/")
