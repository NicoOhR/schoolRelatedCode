from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve, TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import (classification_report, balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay)
import numpy as np
import pandas as pd
from pathlib import Path
from report_tables import write_tables
  
taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572) 
  
X = taiwanese_bankruptcy_prediction.data.features 
y = taiwanese_bankruptcy_prediction.data.targets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


FIG_DIR = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "figures"
FIG_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"descision_tree_{name}.png", dpi=200, bbox_inches="tight")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
f1_score(y_test, y_pred)
accuracy_score(y_test, y_pred)
recall_score(y_test, y_pred)
precision_score(y_test, y_pred)

disp = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
save_fig(disp.figure_, "unpruned_pr_curve")

ccp_alphas = clf.cost_complexity_pruning_path(X_train, y_train).ccp_alphas[:-1][::10]

param_grid = {
    "max_depth": list(range(1, clf.get_depth())) + [None],
    "ccp_alpha": ccp_alphas
}

grid = GridSearchCV(estimator= tree.DecisionTreeClassifier(), cv=5, n_jobs=5, param_grid=param_grid, scoring='f1', return_train_score=True, verbose=1)
grid.fit(X_train, y_train)

# Elbow plot 1: F1 vs max_depth alone (no pruning)
depths = np.arange(1, clf.get_depth() + 1)

train_scores, val_scores = validation_curve(
    tree.DecisionTreeClassifier(),
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
ax.set_title("Decision tree: F1 vs. max_depth")
ax.set_xticks(depths)
ax.legend()
save_fig(fig, "f1_vs_max_depth")

# Elbow plot 2: best F1 per max_depth from the grid search (max over ccp_alpha)
res = pd.DataFrame(grid.cv_results_)
res = res[res["param_max_depth"].notna()]
res["param_max_depth"] = res["param_max_depth"].astype(int)
per_depth = res.groupby("param_max_depth")[["mean_test_score", "mean_train_score"]].max()

ax = per_depth.plot(marker="o", xlabel="max_depth", ylabel="F1",
                    title="Best CV F1 per max_depth (over ccp_alpha)")
save_fig(ax.figure, "grid_f1_vs_max_depth")

# Final estimator: evaluate on the held-out test set
best_tree = grid.best_estimator_
print("Best params:", grid.best_params_)
print(f"Best CV F1: {grid.best_score_:.3f}")

y_pred = best_tree.predict(X_test)
y_score = best_tree.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=3))
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"MCC:               {matthews_corrcoef(y_test, y_pred):.3f}")
print(f"ROC AUC:           {roc_auc_score(y_test, y_score):.3f}")
print(f"Average precision: {average_precision_score(y_test, y_score):.3f}")

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.ax_.set_title("Confusion matrix")
save_fig(disp.figure_, "confusion_matrix")

disp = PrecisionRecallDisplay.from_estimator(best_tree, X_test, y_test, plot_chance_level=True)
disp.ax_.set_title("Precision-recall curve")
save_fig(disp.figure_, "pr_curve")

disp = RocCurveDisplay.from_estimator(best_tree, X_test, y_test, plot_chance_level=True)
disp.ax_.set_title("ROC curve")
save_fig(disp.figure_, "roc_curve")

# Threshold tuning: choose the probability cutoff that maximizes F1 (via CV on the training set)
tuned = TunedThresholdClassifierCV(best_tree, scoring="f1", cv=5, n_jobs=5).fit(X_train, y_train)
y_pred_tuned = tuned.predict(X_test)
print(f"Tuned threshold: {tuned.best_threshold_:.3f} (CV F1 {tuned.best_score_:.3f})")
print(classification_report(y_test, y_pred_tuned, digits=3))
print(f"MCC (tuned):       {matthews_corrcoef(y_test, y_pred_tuned):.3f}")

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tuned)
disp.ax_.set_title(f"Confusion matrix (threshold = {tuned.best_threshold_:.3f})")
save_fig(disp.figure_, "tuned_confusion_matrix")

# LaTeX tables for report.tex
write_tables("descision_tree", "Decision tree", grid, y_test, y_pred, y_pred_tuned, y_score, tuned.best_threshold_)
