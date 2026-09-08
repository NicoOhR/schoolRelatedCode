import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from models.criteria import LogLoss, LogLossHess
from models.gradient_boost import GradientBoost
from models.xg_boost import XGBoost

SEED       = 42
LR         = 0.1
SPAM_ITERS = 80
SPAM_DEPTH = 12
np.random.seed(SEED)

_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))
classify  = lambda lo: (_sigmoid(lo) >= 0.5).astype(int)


# ── data ──────────────────────────────────────────────────────────────────────
print("Loading Spambase...")
spam_raw = fetch_openml("spambase", version=1, as_frame=True, parser="auto")
df_spam = spam_raw.data.copy()
df_spam["target"] = spam_raw.target.astype(float).astype(int)
train_spam, test_spam = train_test_split(df_spam, test_size=0.2, random_state=SEED)
X_train = train_spam.drop(columns=["target"])
y_train = train_spam["target"].to_numpy()
y_test  = test_spam["target"].to_numpy()

print(f"Train: {len(train_spam)} rows  |  Test: {len(test_spam)} rows")
print(f"Class balance (test): {y_test.mean():.2%} spam")

# ── models ────────────────────────────────────────────────────────────────────
models = {
    "Custom GBM": GradientBoost(
        "target", iterations=SPAM_ITERS, learning_rate=LR,
        tree_depth=SPAM_DEPTH, loss=LogLoss(), verbose=True,
    ),
    "Custom XGB (λ=0.1)": XGBoost(
        "target", iterations=SPAM_ITERS, learning_rate=LR,
        tree_depth=SPAM_DEPTH, loss=LogLossHess(lambda_reg=0.1), verbose=True,
    ),
    "Custom XGB (λ=1)": XGBoost(
        "target", iterations=SPAM_ITERS, learning_rate=LR,
        tree_depth=SPAM_DEPTH, loss=LogLossHess(lambda_reg=1.0), verbose=True,
    ),
}

# ── fit ───────────────────────────────────────────────────────────────────────
def _fit_task(args):
    name, model, fit_args = args
    model.fit(*fit_args)
    print(f"  {name}: done", flush=True)
    return model


tasks = [
    ("Custom GBM",         models["Custom GBM"],         (train_spam,)),
    ("Custom XGB (λ=0.1)", models["Custom XGB (λ=0.1)"], (train_spam,)),
    ("Custom XGB (λ=1)",   models["Custom XGB (λ=1)"],   (train_spam,)),
]

print("Fitting all models in parallel...")
with ProcessPoolExecutor() as executor:
    fitted = list(executor.map(_fit_task, tasks))

models["Custom GBM"], models["Custom XGB (λ=0.1)"], models["Custom XGB (λ=1)"] = fitted
print("Done.")

# ── metrics ───────────────────────────────────────────────────────────────────
accs_train = {}
accs_test  = {}

for name, model in models.items():
    accs_train[name] = (classify(model.predict(train_spam)) == y_train).mean()
    accs_test[name]  = (classify(model.predict(test_spam))  == y_test).mean()

print(f"\nSpambase — Train / Test Accuracy  (depth={SPAM_DEPTH}, iters={SPAM_ITERS})")
for name in models:
    gap = accs_train[name] - accs_test[name]
    print(f"  {name:22s}  train={accs_train[name]:.4f}  test={accs_test[name]:.4f}  gap={gap:+.4f}")

# ── plots ─────────────────────────────────────────────────────────────────────
os.makedirs("plots/spam", exist_ok=True)

_pal = ["#4878cf", "#e24a33", "#6acc65"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(accs_test.keys(), accs_test.values(), color=_pal,
              edgecolor="white", width=0.5)
ax.set_ylim(0.8, 1.01)
for bar, val in zip(bars, accs_test.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_title(f"Test Accuracy  (Spambase, depth={SPAM_DEPTH})")
ax.set_ylabel("Accuracy")
ax.tick_params(axis="x", rotation=10)
plt.tight_layout()
plt.savefig("plots/spam/accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
names = list(models.keys())
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, [accs_train[n] for n in names], w, label="train", color="#6baed6")
ax.bar(x + w/2, [accs_test[n]  for n in names], w, label="test",  color="#fd8d3c")
for i, name in enumerate(names):
    gap = accs_train[name] - accs_test[name]
    y_top = max(accs_train[name], accs_test[name])
    ax.text(i, y_top + 0.002, f"Δ{gap:.3f}", ha="center", va="bottom",
            fontsize=8, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.8, 1.05)
ax.set_title(f"Train vs Test Accuracy  (Spambase, depth={SPAM_DEPTH})")
ax.set_ylabel("Accuracy")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/spam/train_vs_test.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nPlots saved to plots/spam/")
