import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["Decision Tree\n(depth 4)", "Random Forest\n(10 trees)", "Gradient Boosting\n(10, lr=0.3)"]
bias2    = [0.0015, 0.0017, 0.0480]
variance = [0.0316, 0.0197, 0.0102]
mse      = [0.0331, 0.0213, 0.0582]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width, bias2,    width, label="Bias²",    color="steelblue")
ax.bar(x,         variance, width, label="Variance", color="tomato")
ax.bar(x + width, mse,      width, label="MSE",      color="seagreen")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Value")
ax.legend()
ax.set_title("Bias², Variance, and MSE by Ensemble Method")

fig.tight_layout()
fig.savefig("ensemble_barplot.png", bbox_inches="tight", dpi=150)
