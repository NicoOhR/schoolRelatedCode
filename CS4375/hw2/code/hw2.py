import numpy as np
from scipy.spatial import KDTree, kdtree
import scipy.spatial.kdtree
import pandas as pd
from sklearn.datasets import load_iris


def knn(df, label, k):
    # build a kd-tree out of input data frame
    # query for each point the nearest neighbors classify
    X = df.drop(columns=[label])
    tree = KDTree(X.to_numpy())
    predicted = []

    for idx, row in df.iterrows():
        _, neighbors = tree.query(X.iloc[idx], k=k)

        values, counts = np.unique(
            df.iloc[neighbors][label].to_list(), return_counts=True
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

    result = knn(df, "label", k=3)
    accuracy = (result["label"] == result["predicted"]).mean()
    print(f"kNN accuracy on iris: {accuracy:.4f}")


if __name__ == "__main__":
    main()
