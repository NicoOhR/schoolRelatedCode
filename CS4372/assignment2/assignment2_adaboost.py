from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve, TunedThresholdClassifierCV
from sklearn.base import clone
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
    fig.savefig(FIG_DIR / f"adaboost_{name}.png", dpi=200, bbox_inches="tight")


# Default base learner is a depth-1 decision tree (a stump)
clf = AdaBoostClassifier(random_state=42)
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
    "n_estimators": [50, 100, 200, 400],
    "learning_rate": [0.1, 0.5, 1.0],
}

grid = GridSearchCV(estimator=AdaBoostClassifier(random_state=42), cv=5, n_jobs=5, param_grid=param_grid, scoring='f1', return_train_score=True, verbose=1)
grid.fit(X_train, y_train)

# Elbow plot 1: F1 vs n_estimators alone (learning_rate default)
n_estimators = np.array([25, 50, 100, 200, 300, 400, 600])

train_scores, val_scores = validation_curve(
    AdaBoostClassifier(random_state=42),
    X_train, y_train,
    param_name="n_estimators",
    param_range=n_estimators,
    scoring="f1",
    cv=5,
    n_jobs=5,
)

train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_estimators, train_mean, marker="o", label="Train")
ax.fill_between(n_estimators, train_mean - train_std, train_mean + train_std, alpha=0.2)
ax.plot(n_estimators, val_mean, marker="o", label="Cross-validation")
ax.fill_between(n_estimators, val_mean - val_std, val_mean + val_std, alpha=0.2)

best = n_estimators[val_mean.argmax()]
ax.axvline(best, ls="--", c="gray", label=f"Best n_estimators = {best}")

ax.set_xlabel("n_estimators")
ax.set_ylabel("F1 score")
ax.set_title("AdaBoost: F1 vs. n_estimators")
ax.set_xticks(n_estimators)
ax.legend()
save_fig(fig, "f1_vs_n_estimators")

# Elbow plot 2: best F1 per n_estimators from the grid search (max over learning_rate)
res = pd.DataFrame(grid.cv_results_)
res["param_n_estimators"] = res["param_n_estimators"].astype(int)
per_n = res.groupby("param_n_estimators")[["mean_test_score", "mean_train_score"]].max()

ax = per_n.plot(marker="o", xlabel="n_estimators", ylabel="F1",
                title="Best CV F1 per n_estimators (over learning_rate)")
save_fig(ax.figure, "grid_f1_vs_n_estimators")

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

# Score after each boosting round, on a validation split carved out of the training set
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
staged_model = clone(best_model).fit(X_tr, y_tr)

train_ap = [average_precision_score(y_tr, p[:, 1]) for p in staged_model.staged_predict_proba(X_tr)]
val_ap = [average_precision_score(y_val, p[:, 1]) for p in staged_model.staged_predict_proba(X_val)]
rounds = np.arange(1, len(train_ap) + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rounds, train_ap, label="Train")
ax.plot(rounds, val_ap, label="Validation")
ax.axvline(rounds[np.argmax(val_ap)], ls="--", c="gray", label=f"Best round = {rounds[np.argmax(val_ap)]}")
ax.set_xlabel("Boosting round")
ax.set_ylabel("Average precision")
ax.set_title("AdaBoost: average precision per boosting round")
ax.legend()
save_fig(fig, "staged_average_precision")

# Learning rate vs number of rounds, from the grid search results
per_lr = res.pivot_table(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score", aggfunc="max")
ax = per_lr.plot(marker="o", xlabel="n_estimators", ylabel="CV F1",
                 title="AdaBoost: CV F1 vs n_estimators per learning rate")
ax.legend(title="learning_rate")
save_fig(ax.figure, "learning_rate_vs_n_estimators")

# Weight and weighted error of each stump in the final model
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
n_fit = len(best_model.estimators_)
ax1.plot(np.arange(1, n_fit + 1), best_model.estimator_weights_[:n_fit])
ax1.set_ylabel("Estimator weight")
ax2.plot(np.arange(1, n_fit + 1), best_model.estimator_errors_[:n_fit])
ax2.axhline(0.5, ls="--", c="gray", label="Random guessing")
ax2.set_ylabel("Weighted error")
ax2.set_xlabel("Boosting round")
ax2.legend()
ax1.set_title("AdaBoost: per-round estimator weights and errors")
save_fig(fig, "estimator_weights_errors")

# Top 15 features by importance
importances = pd.Series(best_model.feature_importances_, index=X.columns).nlargest(15)
fig, ax = plt.subplots(figsize=(8, 6))
importances.sort_values().plot.barh(ax=ax)
ax.set_xlabel("Importance")
ax.set_title("AdaBoost: top 15 feature importances")
save_fig(fig, "feature_importance")

# LaTeX tables for report.tex
write_tables("adaboost", "AdaBoost", grid, y_test, y_pred, y_pred_tuned, y_score, tuned.best_threshold_)
