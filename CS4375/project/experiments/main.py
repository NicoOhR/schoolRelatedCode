import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_california_housing, fetch_openml
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from models.tree import Tree
from models.criteria import MSEHessLoss, LogLoss, LogLossHess
from models.gradient_boost import GradientBoost
from models.xg_boost import XGBoost

SEED = 42
ITERS = 60
LR = 0.1
CAL_ITERS  = 30
CAL_DEPTH  = 4
CAL_ROWS   = 5000
SPAM_ITERS = 30
SPAM_DEPTH = 4
np.random.seed(SEED)


def rmse(y_true, y_pred):
    return np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean())


def _fit_task(args):
    name, model, fit_args = args
    model.fit(*fit_args)
    print(f"  {name}: done", flush=True)
    return model


# ── 1-D toy: noisy sinusoid ───────────────────────────────────────────────────
x = np.linspace(-3, 3, 250)
y = np.sin(x) + np.random.normal(0, 0.25, len(x))
df_1d = pd.DataFrame({"x": x, "y": y})
train_1d, test_1d = train_test_split(df_1d, test_size=0.2, random_state=SEED)
test_1d = test_1d.sort_values("x")

our_xgb_1d = XGBoost(
    "y", iterations=ITERS, learning_rate=LR, tree_depth=4, verbose=False
)
our_tree_1d = Tree("y", max_depth=4)
our_gbm_1d = GradientBoost("y", iterations=ITERS, learning_rate=LR, verbose=False)
sk_tree_1d = DecisionTreeRegressor(max_depth=4, random_state=SEED)
sk_gbm_1d = GradientBoostingRegressor(
    n_estimators=ITERS, learning_rate=LR, max_depth=1, random_state=SEED
)

# ── multi-feature: diabetes ───────────────────────────────────────────────────
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
train_d, test_d = train_test_split(df, test_size=0.2, random_state=SEED)
X_train_d = train_d.drop(columns=["target"])
X_test_d = test_d.drop(columns=["target"])
y_test_d = test_d["target"].to_numpy()

our_xgb_d = XGBoost(
    "target", iterations=ITERS, learning_rate=LR, tree_depth=4, verbose=False
)
our_tree_d = Tree("target", max_depth=4)
our_gbm_d = GradientBoost("target", iterations=ITERS, learning_rate=LR, verbose=False)
sk_tree_d = DecisionTreeRegressor(max_depth=4, random_state=SEED)
sk_gbm_d = GradientBoostingRegressor(
    n_estimators=ITERS, learning_rate=LR, max_depth=1, random_state=SEED
)

# ── California Housing ────────────────────────────────────────────────────────
cal = fetch_california_housing()
df_cal = pd.DataFrame(cal.data, columns=cal.feature_names)
df_cal["target"] = cal.target
train_cal, test_cal = train_test_split(df_cal, test_size=0.2, random_state=SEED)
train_cal = train_cal.sample(n=CAL_ROWS, random_state=SEED).reset_index(drop=True)
X_train_cal = train_cal.drop(columns=["target"])
X_test_cal = test_cal.drop(columns=["target"])
y_train_cal = train_cal["target"].to_numpy()
y_test_cal = test_cal["target"].to_numpy()

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

# ── Spambase ─────────────────────────────────────────────────────────────────
spam_raw = fetch_openml("spambase", version=1, as_frame=True, parser="auto")
df_spam = spam_raw.data.copy()
df_spam["target"] = spam_raw.target.astype(float).astype(int)
train_spam, test_spam = train_test_split(df_spam, test_size=0.2, random_state=SEED)
X_train_spam = train_spam.drop(columns=["target"])
X_test_spam  = test_spam.drop(columns=["target"])
y_train_spam = train_spam["target"].to_numpy()
y_test_spam  = test_spam["target"].to_numpy()

our_gbm_spam      = GradientBoost("target", iterations=SPAM_ITERS, learning_rate=LR, tree_depth=SPAM_DEPTH, loss=LogLoss(), verbose=True)
our_xgb_spam      = XGBoost("target", iterations=SPAM_ITERS, learning_rate=LR, tree_depth=SPAM_DEPTH, loss=LogLossHess(lambda_reg=0.0), verbose=True)
our_xgb_spam_reg  = XGBoost("target", iterations=SPAM_ITERS, learning_rate=LR, tree_depth=SPAM_DEPTH, loss=LogLossHess(lambda_reg=1.0), verbose=True)
sk_gbm_spam       = GradientBoostingClassifier(n_estimators=SPAM_ITERS, learning_rate=LR, max_depth=SPAM_DEPTH, random_state=SEED)

# ── parallel fitting ──────────────────────────────────────────────────────────
print("Fitting all models in parallel...")
tasks = [
    ("XGBoost 1D", our_xgb_1d, (train_1d,)),
    ("Tree 1D", our_tree_1d, (train_1d,)),
    ("GBM 1D", our_gbm_1d, (train_1d,)),
    ("sklearn Tree 1D", sk_tree_1d, (train_1d[["x"]], train_1d["y"])),
    ("sklearn GBM 1D", sk_gbm_1d, (train_1d[["x"]], train_1d["y"])),
    ("XGBoost Diabetes", our_xgb_d, (train_d,)),
    ("Tree Diabetes", our_tree_d, (train_d,)),
    ("GBM Diabetes", our_gbm_d, (train_d,)),
    ("sklearn Tree Diabetes", sk_tree_d, (X_train_d, train_d["target"])),
    ("sklearn GBM Diabetes", sk_gbm_d, (X_train_d, train_d["target"])),
    ("XGBoost Cal (λ=1)", our_xgb_cal, (train_cal,)),
    ("XGBoost Cal (λ=0)", our_xgb_cal_unreg, (train_cal,)),
    ("GBM Cal", our_gbm_cal, (train_cal,)),
    ("sklearn GBM Cal",      sk_gbm_cal,      (X_train_cal, y_train_cal)),
    ("GBM Spam",             our_gbm_spam,    (train_spam,)),
    ("XGBoost Spam (λ=0)",   our_xgb_spam,    (train_spam,)),
    ("XGBoost Spam (λ=1)",   our_xgb_spam_reg,(train_spam,)),
    ("sklearn GBM Spam",     sk_gbm_spam,     (X_train_spam, y_train_spam)),
]

with ProcessPoolExecutor() as executor:
    fitted = list(executor.map(_fit_task, tasks))

(
    our_xgb_1d,
    our_tree_1d,
    our_gbm_1d,
    sk_tree_1d,
    sk_gbm_1d,
    our_xgb_d,
    our_tree_d,
    our_gbm_d,
    sk_tree_d,
    sk_gbm_d,
    our_xgb_cal,
    our_xgb_cal_unreg,
    our_gbm_cal,
    sk_gbm_cal,
    our_gbm_spam,
    our_xgb_spam,
    our_xgb_spam_reg,
    sk_gbm_spam,
) = fitted

print("Done.")

# ── metrics ───────────────────────────────────────────────────────────────────
y_test_1d = test_1d["y"].to_numpy()
preds_1d = {
    "Custom Tree": our_tree_1d.predict(test_1d).to_numpy(),
    "Custom GBM": our_gbm_1d.predict(test_1d),
    "Custom XGBoost": our_xgb_1d.predict(test_1d),
    "sklearn Tree": sk_tree_1d.predict(test_1d[["x"]]),
    "sklearn GBM": sk_gbm_1d.predict(test_1d[["x"]]),
}
rmses_1d = {name: rmse(y_test_1d, p) for name, p in preds_1d.items()}

preds_d = {
    "Custom Tree": our_tree_d.predict(test_d).to_numpy(),
    "Custom GBM": our_gbm_d.predict(test_d),
    "Custom XGBoost": our_xgb_d.predict(test_d),
    "sklearn Tree": sk_tree_d.predict(X_test_d),
    "sklearn GBM": sk_gbm_d.predict(X_test_d),
}
rmses_d = {name: rmse(y_test_d, p) for name, p in preds_d.items()}

cal_models = {
    "Custom GBM": (our_gbm_cal.predict(train_cal), our_gbm_cal.predict(test_cal)),
    "Custom XGB (λ=0)": (
        our_xgb_cal_unreg.predict(train_cal),
        our_xgb_cal_unreg.predict(test_cal),
    ),
    "Custom XGB (λ=1)": (our_xgb_cal.predict(train_cal), our_xgb_cal.predict(test_cal)),
    "sklearn GBM": (sk_gbm_cal.predict(X_train_cal), sk_gbm_cal.predict(X_test_cal)),
}
rmses_cal_train = {n: rmse(y_train_cal, tr) for n, (tr, _) in cal_models.items()}
rmses_cal_test = {n: rmse(y_test_cal, te) for n, (_, te) in cal_models.items()}

print("\n1-D Sinusoid — Test RMSE")
for name, val in rmses_1d.items():
    print(f"  {name:14s}: {val:.4f}")

print("\nDiabetes — Test RMSE")
for name, val in rmses_d.items():
    print(f"  {name:14s}: {val:.2f}")

print("\nCalifornia Housing — Test RMSE")
for name, val in rmses_cal_test.items():
    print(f"  {name:16s}: {val:.4f}")

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))

def classify(log_odds):
    return (_sigmoid(log_odds) >= 0.5).astype(int)

spam_models = {
    "Custom GBM":       (our_gbm_spam.predict(train_spam),     our_gbm_spam.predict(test_spam)),
    "Custom XGB (λ=0)": (our_xgb_spam.predict(train_spam),     our_xgb_spam.predict(test_spam)),
    "Custom XGB (λ=1)": (our_xgb_spam_reg.predict(train_spam), our_xgb_spam_reg.predict(test_spam)),
    "sklearn GBM":      (sk_gbm_spam.predict(X_train_spam),    sk_gbm_spam.predict(X_test_spam)),
}
# custom models output log-odds; sklearn outputs class labels
accs_spam_train = {}
accs_spam_test  = {}
for name, (tr, te) in spam_models.items():
    if name.startswith("sklearn"):
        accs_spam_train[name] = (tr == y_train_spam).mean()
        accs_spam_test[name]  = (te == y_test_spam).mean()
    else:
        accs_spam_train[name] = (classify(tr) == y_train_spam).mean()
        accs_spam_test[name]  = (classify(te) == y_test_spam).mean()

print("\nSpambase — Test Accuracy")
for name, val in accs_spam_test.items():
    print(f"  {name:18s}: {val:.4f}")


# ── plots ─────────────────────────────────────────────────────────────────────
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


os.makedirs("plots/1d", exist_ok=True)
os.makedirs("plots/diabetes", exist_ok=True)
os.makedirs("plots/california", exist_ok=True)

# ── 1-D ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    test_1d["x"], test_1d["y"], s=12, c="black", alpha=0.35, label="data", zorder=3
)
ax.plot(test_1d["x"], preds_1d["Custom Tree"], lw=1.8, label="Custom Tree")
ax.plot(test_1d["x"], preds_1d["Custom GBM"], lw=1.8, label="Custom GBM")
ax.plot(test_1d["x"], preds_1d["Custom XGBoost"], lw=1.8, label="Custom XGBoost")
ax.plot(test_1d["x"], preds_1d["sklearn Tree"], lw=1.5, ls="--", label="sklearn Tree")
ax.plot(test_1d["x"], preds_1d["sklearn GBM"], lw=1.5, ls="--", label="sklearn GBM")
ax.set_title("1-D Fit  (sin + noise)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("plots/1d/fit.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4878cf"] * 3 + ["#e24a33"] * 2
bars = ax.bar(
    rmses_1d.keys(), rmses_1d.values(), color=colors, edgecolor="white", width=0.5
)
ax.set_title("Test RMSE  (1-D Sinusoid)")
ax.set_ylabel("RMSE")
ax.tick_params(axis="x", rotation=15)
annotate_bars(ax, bars, list(rmses_1d.values()), fmt="{:.3f}")
plt.tight_layout()
plt.savefig("plots/1d/rmse.png", dpi=150, bbox_inches="tight")
plt.close()

# ── diabetes ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
lo, hi = y_test_d.min() - 10, y_test_d.max() + 10
ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
ax.scatter(y_test_d, preds_d["Custom GBM"], s=18, alpha=0.6, label="Custom GBM")
ax.scatter(
    y_test_d,
    preds_d["Custom XGBoost"],
    s=18,
    alpha=0.6,
    label="Custom XGBoost",
    marker="s",
)
ax.scatter(
    y_test_d, preds_d["sklearn GBM"], s=18, alpha=0.6, label="sklearn GBM", marker="^"
)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_title("Predicted vs Actual  (Diabetes)")
ax.set_xlabel("actual")
ax.set_ylabel("predicted")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/diabetes/predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4878cf"] * 3 + ["#e24a33"] * 2
bars = ax.bar(
    rmses_d.keys(), rmses_d.values(), color=colors, edgecolor="white", width=0.5
)
ax.set_title("Test RMSE  (Diabetes)")
ax.set_ylabel("RMSE")
ax.tick_params(axis="x", rotation=15)
annotate_bars(ax, bars, list(rmses_d.values()), pad=0.5)
plt.tight_layout()
plt.savefig("plots/diabetes/rmse.png", dpi=150, bbox_inches="tight")
plt.close()

# ── california housing ────────────────────────────────────────────────────────
# Predicted vs actual: 2×2 hexbin grid — handles dense point clouds without overplotting
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()
_lo = min(y_test_cal.min(), min(p.min() for _, (_, p) in cal_models.items())) - 0.1
_hi = max(y_test_cal.max(), max(p.max() for _, (_, p) in cal_models.items())) + 0.1
for ax, (name, (_, test_pred)) in zip(axes, cal_models.items()):
    hb = ax.hexbin(y_test_cal, test_pred, gridsize=40, cmap="Blues", mincnt=1)
    ax.plot([_lo, _hi], [_lo, _hi], "r--", lw=1, zorder=3, label="perfect fit")
    ax.set_xlim(_lo, _hi)
    ax.set_ylim(_lo, _hi)
    test_rmse = rmses_cal_test[name]
    ax.set_title(f"{name}  (RMSE={test_rmse:.3f})", fontsize=10)
    ax.set_xlabel("actual  ($100k)")
    ax.set_ylabel("predicted  ($100k)")
    fig.colorbar(hb, ax=ax, label="count")
fig.suptitle("Predicted vs Actual  (California Housing)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/california/predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()

# Residuals: step-outline + light fill per model — no opacity mud
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
ax.bar(
    x - w / 2, [rmses_cal_train[n] for n in names], w, label="train", color="#6baed6"
)
ax.bar(x + w / 2, [rmses_cal_test[n] for n in names], w, label="test", color="#fd8d3c")
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

print("\nPlots saved to plots/1d/, plots/diabetes/, plots/california/")

# ── spam ──────────────────────────────────────────────────────────────────────
os.makedirs("plots/spam", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
names = list(accs_spam_test.keys())
x = np.arange(len(names))
w = 0.35
ax.bar(x - w / 2, [accs_spam_train[n] for n in names], w, label="train", color="#6baed6")
ax.bar(x + w / 2, [accs_spam_test[n]  for n in names], w, label="test",  color="#fd8d3c")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.8, 1.01)
ax.set_title(f"Train vs Test Accuracy  (Spambase, depth={SPAM_DEPTH})")
ax.set_ylabel("Accuracy")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/spam/train_vs_test.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(accs_spam_test.keys(), accs_spam_test.values(), color="#4878cf", edgecolor="white", width=0.5)
ax.set_ylim(0.8, 1.01)
ax.set_title("Test Accuracy  (Spambase)")
ax.set_ylabel("Accuracy")
ax.tick_params(axis="x", rotation=15)
annotate_bars(ax, bars, list(accs_spam_test.values()), fmt="{:.3f}")
plt.tight_layout()
plt.savefig("plots/spam/accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

print("Plots saved to plots/spam/")
