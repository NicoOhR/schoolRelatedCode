from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
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
    fig.savefig(FIG_DIR / f"xgboost_{name}.png", dpi=200, bbox_inches="tight")


# n_jobs=1 inside the model so it doesn't fight GridSearchCV's own parallel workers
clf = XGBClassifier(random_state=42, n_jobs=1)
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print(f"f1_score: {f1_score(y_test, y_pred)}")
print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
print(f"recall_score: {recall_score(y_test, y_pred)}")
print(f"precision_score: {precision_score(y_test, y_pred)}")

disp = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
save_fig(disp.figure_, "default_pr_curve")

# Up-weights the rare bankrupt class by the negative/positive ratio (~30 here)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 6],
    "learning_rate": [0.1, 0.3],
    "scale_pos_weight": [1, pos_weight],
}

grid = GridSearchCV(estimator=XGBClassifier(random_state=42, n_jobs=1), cv=5, n_jobs=5, param_grid=param_grid, scoring='f1', return_train_score=True, verbose=1)
grid.fit(X_train, y_train)

# Elbow plot 1: F1 vs n_estimators alone (other params default)
n_estimators = np.array([25, 50, 100, 200, 300, 400, 600])

train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, n_jobs=1),
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
ax.set_title("XGBoost: F1 vs. n_estimators")
ax.set_xticks(n_estimators)
ax.legend()
save_fig(fig, "f1_vs_n_estimators")

# Elbow plot 2: best F1 per n_estimators from the grid search (max over the other params)
res = pd.DataFrame(grid.cv_results_)
res["param_n_estimators"] = res["param_n_estimators"].astype(int)
per_n = res.groupby("param_n_estimators")[["mean_test_score", "mean_train_score"]].max()

ax = per_n.plot(marker="o", xlabel="n_estimators", ylabel="F1",
                title="Best CV F1 per n_estimators (over max_depth, learning_rate, scale_pos_weight)")
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

# Score after each boosting round with early stopping, on a validation split carved out of the training set
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
staged_model = clone(best_model).set_params(n_estimators=1000, eval_metric="aucpr", early_stopping_rounds=50, n_jobs=-1)
staged_model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)

history = staged_model.evals_result()
train_ap = history["validation_0"]["aucpr"]
val_ap = history["validation_1"]["aucpr"]
rounds = np.arange(1, len(train_ap) + 1)
best_round = staged_model.best_iteration + 1
print(f"Early stopping: best round = {best_round}, validation AUCPR = {staged_model.best_score:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rounds, train_ap, label="Train")
ax.plot(rounds, val_ap, label="Validation")
ax.axvline(best_round, ls="--", c="gray", label=f"Best round = {best_round}")
ax.set_xlabel("Boosting round")
ax.set_ylabel("AUCPR")
ax.set_title("XGBoost: AUCPR per boosting round (early stopping)")
ax.legend()
save_fig(fig, "staged_aucpr")

# Learning rate vs number of rounds, from the grid search results (max over max_depth, scale_pos_weight)
per_lr = res.pivot_table(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score", aggfunc="max")
ax = per_lr.plot(marker="o", xlabel="n_estimators", ylabel="CV F1",
                 title="XGBoost: CV F1 vs n_estimators per learning rate")
ax.legend(title="learning_rate")
save_fig(ax.figure, "learning_rate_vs_n_estimators")

# Top 15 features by gain (average loss reduction from splits on that feature)
gain = pd.Series(best_model.get_booster().get_score(importance_type="gain")).nlargest(15)
fig, ax = plt.subplots(figsize=(8, 6))
gain.sort_values().plot.barh(ax=ax)
ax.set_xlabel("Gain")
ax.set_title("XGBoost: top 15 features by gain")
save_fig(fig, "feature_importance")

# LaTeX tables for report.tex
write_tables("xgboost", "XGBoost", grid, y_test, y_pred, y_pred_tuned, y_score, tuned.best_threshold_)
