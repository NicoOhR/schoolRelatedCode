"""CS4372 Assignment 2: tree-based models on the Taiwanese bankruptcy dataset.

Trains and evaluates a decision tree, random forest, AdaBoost and XGBoost, saving
figures to figures/ and the LaTeX tables that report.tex pulls in to tables/.
"""
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve, TunedThresholdClassifierCV
from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, PrecisionRecallDisplay
from sklearn.metrics import (classification_report, balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay)
import numpy as np
import pandas as pd
from pathlib import Path


# fetch dataset
taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)

# data (as pandas dataframes)
X = taiwanese_bankruptcy_prediction.data.features
y = taiwanese_bankruptcy_prediction.data.targets.squeeze()  # Series, avoids column-vector warnings from the ensembles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


def save_fig(fig, prefix, name):
    fig.savefig(FIG_DIR / f"{prefix}_{name}.png", dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# LaTeX tables for report.tex
# ---------------------------------------------------------------------------

def _tex(value):
    if isinstance(value, float):
        value = f"{value:.3g}"
    return str(value).replace("_", r"\_")


def write_tables(name, label, grid, y_test, y_pred, y_pred_tuned, y_score, threshold):
    """name: file prefix (e.g. "xgboost"), label: display name for the summary table."""
    # Best hyperparameters from the grid search
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Hyperparameter & Value \\", r"\midrule"]
    lines += [f"{_tex(k)} & {_tex(v)} \\\\" for k, v in grid.best_params_.items()]
    lines += [r"\midrule", f"Mean CV F1 & {grid.best_score_:.3f} \\\\", r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / f"{name}_best_params.tex").write_text("\n".join(lines) + "\n")

    # Test-set metrics at the default 0.5 threshold and at the tuned threshold
    metrics = [
        ("Accuracy", accuracy_score),
        ("Balanced accuracy", balanced_accuracy_score),
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("F1", f1_score),
        ("MCC", matthews_corrcoef),
    ]
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             f"Metric & Threshold 0.5 & Tuned ($t = {threshold:.3f}$) \\\\", r"\midrule"]
    lines += [f"{m} & {fn(y_test, y_pred):.3f} & {fn(y_test, y_pred_tuned):.3f} \\\\" for m, fn in metrics]
    lines += [r"\midrule",
              f"ROC AUC & \\multicolumn{{2}}{{c}}{{{roc_auc_score(y_test, y_score):.3f}}} \\\\",
              f"Average precision & \\multicolumn{{2}}{{c}}{{{average_precision_score(y_test, y_score):.3f}}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / f"{name}_metrics.tex").write_text("\n".join(lines) + "\n")

    # One row for the model comparison table
    row = [label,
           f"{f1_score(y_test, y_pred):.3f}", f"{f1_score(y_test, y_pred_tuned):.3f}",
           f"{matthews_corrcoef(y_test, y_pred):.3f}", f"{matthews_corrcoef(y_test, y_pred_tuned):.3f}",
           f"{roc_auc_score(y_test, y_score):.3f}", f"{average_precision_score(y_test, y_score):.3f}"]
    (TABLE_DIR / f"{name}_summary_row.tex").write_text(" & ".join(row) + " \\\\\n")


# ---------------------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------------------

def baseline(clf, prefix, pr_name):
    """Fit a model with default hyperparameters and report its test-set scores."""
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"f1_score: {f1_score(y_test, y_pred)}")
    print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
    print(f"recall_score: {recall_score(y_test, y_pred)}")
    print(f"precision_score: {precision_score(y_test, y_pred)}")

    disp = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
    save_fig(disp.figure_, prefix, pr_name)
    return clf


def plot_validation_curve(estimator, param_name, param_range, title, prefix):
    """Elbow plot: train and CV F1 as a single hyperparameter varies."""
    train_scores, val_scores = validation_curve(
        estimator,
        X_train, y_train,
        param_name=param_name,
        param_range=param_range,
        scoring="f1",
        cv=5,
        n_jobs=5,
    )

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_range, train_mean, marker="o", label="Train")
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(param_range, val_mean, marker="o", label="Cross-validation")
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2)

    best = param_range[val_mean.argmax()]
    label = "Best depth" if param_name == "max_depth" else f"Best {param_name}"
    ax.axvline(best, ls="--", c="gray", label=f"{label} = {best}")

    ax.set_xlabel(param_name)
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax.set_xticks(param_range)
    ax.legend()
    save_fig(fig, prefix, f"f1_vs_{param_name}")


def evaluate(grid, prefix, label):
    """Evaluate the grid search's best model on the test set, tune its threshold, and write the tables."""
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
    save_fig(disp.figure_, prefix, "confusion_matrix")

    disp = PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test, plot_chance_level=True)
    disp.ax_.set_title("Precision-recall curve")
    save_fig(disp.figure_, prefix, "pr_curve")

    disp = RocCurveDisplay.from_estimator(best_model, X_test, y_test, plot_chance_level=True)
    disp.ax_.set_title("ROC curve")
    save_fig(disp.figure_, prefix, "roc_curve")

    # Threshold tuning: choose the probability cutoff that maximizes F1 (via CV on the training set)
    tuned = TunedThresholdClassifierCV(best_model, scoring="f1", cv=5, n_jobs=5).fit(X_train, y_train)
    y_pred_tuned = tuned.predict(X_test)
    print(f"Tuned threshold: {tuned.best_threshold_:.3f} (CV F1 {tuned.best_score_:.3f})")
    print(classification_report(y_test, y_pred_tuned, digits=3))
    print(f"MCC (tuned):       {matthews_corrcoef(y_test, y_pred_tuned):.3f}")

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tuned)
    disp.ax_.set_title(f"Confusion matrix (threshold = {tuned.best_threshold_:.3f})")
    save_fig(disp.figure_, prefix, "tuned_confusion_matrix")

    write_tables(prefix, label, grid, y_test, y_pred, y_pred_tuned, y_score, tuned.best_threshold_)
    return best_model


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

def decision_tree():
    prefix = "descision_tree"
    clf = baseline(tree.DecisionTreeClassifier(), prefix, "unpruned_pr_curve")

    ccp_alphas = clf.cost_complexity_pruning_path(X_train, y_train).ccp_alphas[:-1][::10]

    param_grid = {
        "max_depth": list(range(1, clf.get_depth())) + [None],
        "ccp_alpha": ccp_alphas
    }

    grid = GridSearchCV(estimator=tree.DecisionTreeClassifier(), cv=5, n_jobs=5, param_grid=param_grid, scoring='f1', return_train_score=True, verbose=1)
    grid.fit(X_train, y_train)

    # Elbow plot 1: F1 vs max_depth alone (no pruning)
    plot_validation_curve(tree.DecisionTreeClassifier(), "max_depth", np.arange(1, clf.get_depth() + 1),
                          "Decision tree: F1 vs. max_depth", prefix)

    # Elbow plot 2: best F1 per max_depth from the grid search (max over ccp_alpha)
    res = pd.DataFrame(grid.cv_results_)
    res = res[res["param_max_depth"].notna()]
    res["param_max_depth"] = res["param_max_depth"].astype(int)
    per_depth = res.groupby("param_max_depth")[["mean_test_score", "mean_train_score"]].max()

    ax = per_depth.plot(marker="o", xlabel="max_depth", ylabel="F1",
                        title="Best CV F1 per max_depth (over ccp_alpha)")
    save_fig(ax.figure, prefix, "grid_f1_vs_max_depth")

    evaluate(grid, prefix, "Decision tree")


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------

def random_forest():
    prefix = "random_forest"
    baseline(RandomForestClassifier(random_state=42, n_jobs=-1), prefix, "default_pr_curve")

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [5, 10, 20, None],
        "class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), cv=5, n_jobs=5, param_grid=param_grid, scoring='f1', return_train_score=True, verbose=1)
    grid.fit(X_train, y_train)

    # Elbow plot 1: F1 vs max_depth alone (other params default)
    plot_validation_curve(RandomForestClassifier(random_state=42), "max_depth", np.arange(2, 31, 2),
                          "Random forest: F1 vs. max_depth", prefix)

    # Elbow plot 2: best F1 per max_depth from the grid search (max over the other params)
    res = pd.DataFrame(grid.cv_results_)
    res = res[res["param_max_depth"].notna()]
    res["param_max_depth"] = res["param_max_depth"].astype(int)
    per_depth = res.groupby("param_max_depth")[["mean_test_score", "mean_train_score"]].max()

    ax = per_depth.plot(marker="o", xlabel="max_depth", ylabel="F1",
                        title="Best CV F1 per max_depth (over n_estimators, class_weight)")
    save_fig(ax.figure, prefix, "grid_f1_vs_max_depth")

    evaluate(grid, prefix, "Random forest")


# ---------------------------------------------------------------------------
# AdaBoost
# ---------------------------------------------------------------------------

def adaboost():
    prefix = "adaboost"
    # Default base learner is a depth-1 decision tree (a stump)
    baseline(AdaBoostClassifier(random_state=42), prefix, "default_pr_curve")

    param_grid = {
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [0.1, 0.5, 1.0],
    }

    grid = GridSearchCV(estimator=AdaBoostClassifier(random_state=42), cv=5, n_jobs=5, param_grid=param_grid, scoring='f1', return_train_score=True, verbose=1)
    grid.fit(X_train, y_train)

    # Elbow plot 1: F1 vs n_estimators alone (learning_rate default)
    plot_validation_curve(AdaBoostClassifier(random_state=42), "n_estimators", np.array([25, 50, 100, 200, 300, 400, 600]),
                          "AdaBoost: F1 vs. n_estimators", prefix)

    # Elbow plot 2: best F1 per n_estimators from the grid search (max over learning_rate)
    res = pd.DataFrame(grid.cv_results_)
    res["param_n_estimators"] = res["param_n_estimators"].astype(int)
    per_n = res.groupby("param_n_estimators")[["mean_test_score", "mean_train_score"]].max()

    ax = per_n.plot(marker="o", xlabel="n_estimators", ylabel="F1",
                    title="Best CV F1 per n_estimators (over learning_rate)")
    save_fig(ax.figure, prefix, "grid_f1_vs_n_estimators")

    best_model = evaluate(grid, prefix, "AdaBoost")

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
    save_fig(fig, prefix, "staged_average_precision")

    # Learning rate vs number of rounds, from the grid search results
    per_lr = res.pivot_table(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score", aggfunc="max")
    ax = per_lr.plot(marker="o", xlabel="n_estimators", ylabel="CV F1",
                     title="AdaBoost: CV F1 vs n_estimators per learning rate")
    ax.legend(title="learning_rate")
    save_fig(ax.figure, prefix, "learning_rate_vs_n_estimators")

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
    save_fig(fig, prefix, "estimator_weights_errors")

    # Top 15 features by importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns).nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("AdaBoost: top 15 feature importances")
    save_fig(fig, prefix, "feature_importance")


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def xgboost():
    prefix = "xgboost"
    # n_jobs=1 inside the model so it doesn't fight GridSearchCV's own parallel workers
    baseline(XGBClassifier(random_state=42, n_jobs=1), prefix, "default_pr_curve")

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
    plot_validation_curve(XGBClassifier(random_state=42, n_jobs=1), "n_estimators", np.array([25, 50, 100, 200, 300, 400, 600]),
                          "XGBoost: F1 vs. n_estimators", prefix)

    # Elbow plot 2: best F1 per n_estimators from the grid search (max over the other params)
    res = pd.DataFrame(grid.cv_results_)
    res["param_n_estimators"] = res["param_n_estimators"].astype(int)
    per_n = res.groupby("param_n_estimators")[["mean_test_score", "mean_train_score"]].max()

    ax = per_n.plot(marker="o", xlabel="n_estimators", ylabel="F1",
                    title="Best CV F1 per n_estimators (over max_depth, learning_rate, scale_pos_weight)")
    save_fig(ax.figure, prefix, "grid_f1_vs_n_estimators")

    best_model = evaluate(grid, prefix, "XGBoost")

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
    save_fig(fig, prefix, "staged_aucpr")

    # Learning rate vs number of rounds, from the grid search results (max over max_depth, scale_pos_weight)
    per_lr = res.pivot_table(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score", aggfunc="max")
    ax = per_lr.plot(marker="o", xlabel="n_estimators", ylabel="CV F1",
                     title="XGBoost: CV F1 vs n_estimators per learning rate")
    ax.legend(title="learning_rate")
    save_fig(ax.figure, prefix, "learning_rate_vs_n_estimators")

    # Top 15 features by gain (average loss reduction from splits on that feature)
    gain = pd.Series(best_model.get_booster().get_score(importance_type="gain")).nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    gain.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost: top 15 features by gain")
    save_fig(fig, prefix, "feature_importance")


if __name__ == "__main__":
    for run in (decision_tree, random_forest, adaboost, xgboost):
        print(f"\n===== {run.__name__} =====")
        run()
        plt.close("all")
