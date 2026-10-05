from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve, TunedThresholdClassifierCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, PrecisionRecallDisplay
from sklearn.metrics import (classification_report, balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay)
import numpy as np
import pandas as pd
from pathlib import Path
from report_tables import write_tables


# fetch dataset
taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)

# data (as pandas dataframes)
X = taiwanese_bankruptcy_prediction.data.features
y = taiwanese_bankruptcy_prediction.data.targets.squeeze()  # Series, avoids column-vector warnings from the ensembles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


FIG_DIR = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "figures"
FIG_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"random_forest_{name}.png", dpi=200, bbox_inches="tight")


clf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print(f"f1_score: {f1_score(y_test, y_pred)}")
print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
print(f"recall_score: {recall_score(y_test, y_pred)}")
print(f"precision_score: {precision_score(y_test, y_pred)}")

disp = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
save_fig(disp.figure_, "default_pr_curve")

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10, 20, None],
    "class_weight": [None, "balanced"],
}

grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), cv=5, n_jobs=5, param_grid=param_grid, scoring='f1', return_train_score=True, verbose=1)
grid.fit(X_train, y_train)

# Elbow plot 1: F1 vs max_depth alone (other params default)
depths = np.arange(2, 31, 2)

train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train, y_train,
    param_name="max_depth",
    param_range=depths,
    scoring="f1",
    cv=5,
    n_jobs=5,
)

train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(depths, train_mean, marker="o", label="Train")
ax.fill_between(depths, train_mean - train_std, train_mean + train_std, alpha=0.2)
ax.plot(depths, val_mean, marker="o", label="Cross-validation")
ax.fill_between(depths, val_mean - val_std, val_mean + val_std, alpha=0.2)

best = depths[val_mean.argmax()]
ax.axvline(best, ls="--", c="gray", label=f"Best depth = {best}")

ax.set_xlabel("max_depth")
ax.set_ylabel("F1 score")
ax.set_title("Random forest: F1 vs. max_depth")
ax.set_xticks(depths)
ax.legend()
save_fig(fig, "f1_vs_max_depth")

# Elbow plot 2: best F1 per max_depth from the grid search (max over the other params)
res = pd.DataFrame(grid.cv_results_)
res = res[res["param_max_depth"].notna()]
res["param_max_depth"] = res["param_max_depth"].astype(int)
per_depth = res.groupby("param_max_depth")[["mean_test_score", "mean_train_score"]].max()

ax = per_depth.plot(marker="o", xlabel="max_depth", ylabel="F1",
                    title="Best CV F1 per max_depth (over n_estimators, class_weight)")
save_fig(ax.figure, "grid_f1_vs_max_depth")

# Final estimator: evaluate on the held-out test set
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print(f"Best CV F1: {grid.best_score_:.3f}")

y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=3))
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"MCC:               {matthews_corrcoef(y_test, y_pred):.3f}")
print(f"ROC AUC:           {roc_auc_score(y_test, y_score):.3f}")
print(f"Average precision: {average_precision_score(y_test, y_score):.3f}")

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.ax_.set_title("Confusion matrix")
save_fig(disp.figure_, "confusion_matrix")

disp = PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test, plot_chance_level=True)
disp.ax_.set_title("Precision-recall curve")
save_fig(disp.figure_, "pr_curve")

disp = RocCurveDisplay.from_estimator(best_model, X_test, y_test, plot_chance_level=True)
disp.ax_.set_title("ROC curve")
save_fig(disp.figure_, "roc_curve")

# Threshold tuning: choose the probability cutoff that maximizes F1 (via CV on the training set)
tuned = TunedThresholdClassifierCV(best_model, scoring="f1", cv=5, n_jobs=5).fit(X_train, y_train)
y_pred_tuned = tuned.predict(X_test)
print(f"Tuned threshold: {tuned.best_threshold_:.3f} (CV F1 {tuned.best_score_:.3f})")
print(classification_report(y_test, y_pred_tuned, digits=3))
print(f"MCC (tuned):       {matthews_corrcoef(y_test, y_pred_tuned):.3f}")

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tuned)
disp.ax_.set_title(f"Confusion matrix (threshold = {tuned.best_threshold_:.3f})")
save_fig(disp.figure_, "tuned_confusion_matrix")

# LaTeX tables for report.tex
write_tables("random_forest", "Random forest", grid, y_test, y_pred, y_pred_tuned, y_score, tuned.best_threshold_)
