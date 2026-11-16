# Run inside the live session:  %run -i save_pred_grid.py
# 5x5 grid of seg_test images with predicted vs. actual labels.
import os

import matplotlib.pyplot as plt
import numpy as np

class_names = sorted(os.listdir(train_dir))
os.makedirs("figures", exist_ok=True)

xs, ys = [], []
for x, y in validation_dataset.unbatch().take(25):
    xs.append(x.numpy())
    ys.append(int(y))
xs = np.stack(xs)
preds = np.argmax(model.predict(xs, verbose=0), axis=1)

plt.figure(figsize=(12, 13))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(xs[i].astype("uint8"))
    ok = preds[i] == ys[i]
    plt.title(
        f"pred: {class_names[preds[i]]}\nactual: {class_names[ys[i]]}",
        color="green" if ok else "red",
        fontsize=10,
    )
    plt.axis("off")
plt.tight_layout()
plt.savefig("figures/prediction_grid.pdf", bbox_inches="tight")
plt.close()
print(f"PRED_GRID_DONE {int((preds == np.array(ys)).sum())}/25 correct")
