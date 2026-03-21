import numpy as np
from scipy.spatial import KDTree, kdtree
import scipy.spatial.kdtree
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def knn(df, label, k):
    # build a kd-tree out of input data frame
    # query for each point the nearest neighbors classify
    X = df.drop(columns=[label])
    tree = KDTree(X.to_numpy())
    predicted = []

    for idx, row in df.iterrows():
        _, neighbors = tree.query(X.iloc[idx], k=k)

        values, counts = np.unique(
            df.iloc[neighbors][label].tolist(), return_counts=True
        )

        label_idx = np.argmax(counts)
        predicted_label = values[label_idx]
        predicted.append(predicted_label)

    df["predicted"] = predicted
    return df


def main():
    iris = load_iris()
    feature_names = [
        name.replace(" (cm)", "").replace(" ", "_") for name in iris.feature_names
    ]
    df = pd.DataFrame(iris.data, columns=feature_names)
    df["label"] = iris.target

    k_values = [1, 5, 15]
    for k in k_values:
        result = knn(df.copy(), "label", k=k)
        accuracy = (result["label"] == result["predicted"]).mean()
        print(f"kNN accuracy on iris (k={k}): {accuracy:.4f}")

    feature_x = feature_names[0]
    feature_y = feature_names[1]
    fig, axes = plt.subplots(
        1, len(k_values), figsize=(12, 4), sharex=True, sharey=True
    )
    for ax, k in zip(axes, k_values):
        result = knn(df.copy(), "label", k=k)
        ax.scatter(
            result[feature_x],
            result[feature_y],
            c=result["predicted"],
            cmap="viridis",
            s=18,
            alpha=0.85,
        )
        ax.set_title(f"k={k}")
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)

    fig.suptitle("kNN predictions on iris (first two features)")
    fig.tight_layout()
    fig.savefig("knn_iris_k_plots.png", dpi=150)


if __name__ == "__main__":
    main()
