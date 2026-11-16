# Run inside the live session AFTER cnn.py has finished:  %run -i save_outputs.py
# Uses the session's variables (model, history, histories, acc, ...) and writes
# figures/ and tables/ for report.tex.
import io
import itertools
import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

FIG_DIR = "figures"
TAB_DIR = "tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

class_names = sorted(os.listdir(train_dir))


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close()


def save_tab(name, text):
    with open(os.path.join(TAB_DIR, f"{name}.tex"), "w") as f:
        f.write(text)


def count_images(d):
    return {c: len(os.listdir(os.path.join(d, c))) for c in class_names}


# ---------------------------------------------------------------- data
train_counts = count_images(train_dir)
val_counts = count_images(validation_dir)
rows = "\n".join(
    f"{c} & {train_counts[c]} & {val_counts[c]} \\\\" for c in class_names
)
save_tab(
    "class_counts",
    "\\begin{tabular}{lrr}\n\\toprule\nClass & Train & Validation \\\\\n\\midrule\n"
    f"{rows}\n\\midrule\nTotal & {sum(train_counts.values())} & "
    f"{sum(val_counts.values())} \\\\\n\\bottomrule\n\\end{{tabular}}\n",
)

images, labels = next(iter(train_dataset))
plt.figure(figsize=(9, 9))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[int(labels[i])])
    plt.axis("off")
save_fig("sample_images")

# ---------------------------------------------------------- base model
n_init = initial_epochs
init_acc, init_val_acc = acc[:n_init], val_acc[:n_init]
init_loss, init_val_loss = loss[:n_init], val_loss[:n_init]

buf = io.StringIO()
with redirect_stdout(buf):
    model.summary()
with open(os.path.join(TAB_DIR, "model_summary.txt"), "w") as f:
    f.write(buf.getvalue())

def out_shape(layer):
    try:
        return str(tuple(layer.output.shape[1:]))
    except Exception:
        return "--"


rows = "\n".join(
    f"{l.name.replace('_', chr(92) + '_')} & {l.__class__.__name__} & "
    f"{out_shape(l)} & {l.count_params():,} \\\\"
    for l in model.layers[1:]
)
save_tab(
    "architecture",
    "\\begin{tabular}{llll}\n\\toprule\nLayer & Type & Output shape & Parameters \\\\\n"
    f"\\midrule\n{rows}\n\\bottomrule\n\\end{{tabular}}\n",
)

save_tab(
    "base_model_results",
    "\\begin{tabular}{lcc}\n\\toprule\n & Accuracy & Loss \\\\\n\\midrule\n"
    f"Before training (validation) & {accuracy0:.3f} & {loss0:.3f} \\\\\n"
    f"Epoch {n_init} (train) & {init_acc[-1]:.3f} & {init_loss[-1]:.3f} \\\\\n"
    f"Epoch {n_init} (validation) & {init_val_acc[-1]:.3f} & {init_val_loss[-1]:.3f} \\\\\n"
    "\\bottomrule\n\\end{tabular}\n",
)

epochs = np.arange(1, n_init + 1)
plt.figure(figsize=(6, 4))
plt.plot(epochs, init_acc, label="Training")
plt.plot(epochs, init_val_acc, label="Validation")
plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.legend(loc="lower right")
save_fig("base_accuracy")

plt.figure(figsize=(6, 4))
plt.plot(epochs, init_loss, label="Training")
plt.plot(epochs, init_val_loss, label="Validation")
plt.xlabel("Epoch"), plt.ylabel("Cross entropy"), plt.legend(loc="upper right")
save_fig("base_loss")

# --------------------------------------------------------- fine tuning
combos = list(itertools.product(fine_tune_at, learning_rate_divider))
rows = "\n".join(
    f"{f} & $10^{{{-4 - int(np.log10(r))}}}$ & {h.history['accuracy'][-1]:.3f} & "
    f"{h.history['val_accuracy'][-1]:.3f} & {h.history['val_loss'][-1]:.3f} \\\\"
    for (f, r), h in zip(combos, histories)
)
save_tab(
    "grid_search",
    "\\begin{tabular}{rcccc}\n\\toprule\nFrozen layers & Learning rate & "
    "Train acc. & Val.\\ acc. & Val.\\ loss \\\\\n\\midrule\n"
    f"{rows}\n\\bottomrule\n\\end{{tabular}}\n",
)
save_tab(
    "chosen_params",
    f"\\begin{{tabular}}{{lc}}\n\\toprule\nParameter & Value \\\\\n\\midrule\n"
    f"Frozen layers & {best_f} of {len(base_model.layers)} \\\\\n"
    f"Learning rate & $10^{{{-4 - int(np.log10(best_r))}}}$ \\\\\n"
    f"Fine-tune epochs & {fine_tune_epochs} \\\\\n\\bottomrule\n\\end{{tabular}}\n",
)

all_epochs = np.arange(1, len(acc) + 1)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
a1.plot(all_epochs, acc, label="Training")
a1.plot(all_epochs, val_acc, label="Validation")
a1.axvline(n_init + 0.5, color="gray", ls="--", label="Start fine tuning")
a1.set_ylabel("Accuracy"), a1.legend(loc="lower right")
a2.plot(all_epochs, loss, label="Training")
a2.plot(all_epochs, val_loss, label="Validation")
a2.axvline(n_init + 0.5, color="gray", ls="--", label="Start fine tuning")
a2.set_ylabel("Cross entropy"), a2.set_xlabel("Epoch"), a2.legend(loc="upper right")
save_fig("fine_tune_curves")

# Final evaluation on the validation set. Labels and predictions are taken
# from the same pass since validation_dataset reshuffles every iteration.
y_true, y_pred = [], []
for x, y in validation_dataset:
    y_true.append(y.numpy())
    y_pred.append(np.argmax(model(x, training=False).numpy(), axis=1))
y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy()
final_acc = (y_true == y_pred).mean()

prec = np.diag(cm) / cm.sum(axis=0).clip(min=1)
rec = np.diag(cm) / cm.sum(axis=1).clip(min=1)
f1 = 2 * prec * rec / (prec + rec).clip(min=1e-12)
rows = "\n".join(
    f"{c} & {p:.3f} & {r:.3f} & {f:.3f} & {cm[i].sum()} \\\\"
    for i, (c, p, r, f) in enumerate(zip(class_names, prec, rec, f1))
)
save_tab(
    "final_metrics",
    "\\begin{tabular}{lcccr}\n\\toprule\nClass & Precision & Recall & F1 & Support \\\\\n"
    f"\\midrule\n{rows}\n\\midrule\nAccuracy & \\multicolumn{{4}}{{c}}{{{final_acc:.3f}}} \\\\\n"
    "\\bottomrule\n\\end{tabular}\n",
)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted"), plt.ylabel("True"), plt.colorbar()
save_fig("confusion_matrix")

print(f"Saved figures to {FIG_DIR}/ and tables to {TAB_DIR}/ "
      f"(final val acc {final_acc:.3f}, best combo f={best_f}, r={best_r})")
