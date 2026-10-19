"""Writes the LaTeX tables that report.tex pulls in with \\input."""
from pathlib import Path
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score, average_precision_score)

TABLE_DIR = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "tables"


def _tex(value):
    if isinstance(value, float):
        value = f"{value:.3g}"
    return str(value).replace("_", r"\_")


def write_tables(name, label, grid, y_test, y_pred, y_pred_tuned, y_score, threshold):
    """name: file prefix (e.g. "xgboost"), label: display name for the summary table."""
    TABLE_DIR.mkdir(exist_ok=True)

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
