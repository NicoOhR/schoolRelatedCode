from unicodedata import normalize
import numpy as np
import pandas as pd
from pandas._libs.hashtable import value_count
import matplotlib.pyplot as plt


class TreeNode:
    def __init__(self, cond, feature, lvl):
        # condition is a number value which split on
        # left is the true path
        self.condition = cond
        self.condition_feature = feature
        self.level = lvl
        self.prediction = None
        self.left = None
        self.right = None

    def next(self, x_i):
        # test with condition and return the next node approprietly
        return (
            self.left if (x_i[self.condition_feature] < self.condition) else self.right
        )


def find_split_ig(df, label):
    if len(df) == 0:
        return None, None, None
    X = df.drop(columns=[label])
    candidates = []
    for col in X.columns:
        sorted_vals = np.sort(df[col].to_numpy())
        midpoints = [(a + b) / 2 for a, b in zip(sorted_vals, sorted_vals[1:])]
        if not midpoints:
            continue
        ig = []
        for point in midpoints:
            left_mask = df[col] < point
            right_mask = ~left_mask
            p_left = (left_mask).mean()
            p_right = 1.0 - p_left
            p_y_left = df.loc[left_mask, label].value_counts(normalize=True)
            p_y_right = df.loc[right_mask, label].value_counts(normalize=True)
            h_left = -(p_y_left * np.log2(p_y_left)).sum()
            h_right = -(p_y_right * np.log2(p_y_right)).sum()
            h_y_x = p_left * h_left + p_right * h_right
            p_y = (df[label]).value_counts(normalize=True)
            h_y = -(p_y * np.log2(p_y)).sum()
            ig.append(h_y - h_y_x)
        best_idx = int(np.argmax(ig))
        candidates.append((midpoints[best_idx], ig[best_idx], col))
    if not candidates:
        return None, None, None
    return max(candidates, key=lambda item: item[1])


def build_tree(df, label, curr):
    if curr is None or curr.condition is None or curr.condition_feature is None:
        return
    curr.prediction = df[label].mode().iat[0]
    if curr.level == 2:
        return
    else:
        left_subset = df.loc[df[curr.condition_feature] < curr.condition]
        left_cond, _, left_feature = find_split_ig(left_subset, label)
        if left_cond is not None and left_feature is not None:
            curr.left = TreeNode(left_cond, left_feature, curr.level + 1)
            build_tree(left_subset, label, curr.left)

        right_subset = df.loc[df[curr.condition_feature] >= curr.condition]
        right_cond, _, right_feature = find_split_ig(right_subset, label)
        if right_cond is not None and right_feature is not None:
            curr.right = TreeNode(right_cond, right_feature, curr.level + 1)
            build_tree(right_subset, label, curr.right)


def predict_one(node, x_i):
    while node and (node.left or node.right):
        next_node = node.next(x_i)
        if next_node is None:
            break
        node = next_node
    return None if node is None else node.prediction


def mse_report(root, df, label):
    preds = df.apply(lambda row: predict_one(root, row), axis=1).to_numpy()
    y_true = df[label].to_numpy()
    return np.mean((preds - y_true) ** 2)


def print_conditions(node, indent=""):
    if node is None:
        return
    if node.condition is None or node.condition_feature is None:
        print(f"{indent}<no split> (level={node.level}, pred={node.prediction})")
        return
    print(
        f"{indent}{node.condition_feature} < {node.condition:.4f} "
        f"(level={node.level}, pred={node.prediction})"
    )
    if node.left or node.right:
        print_conditions(node.left, indent + "  ")
        print_conditions(node.right, indent + "  ")


def main():
    rng = np.random.default_rng(0)
    n = 200
    x1a = rng.normal(loc=[-1.2, -0.8], scale=0.9, size=(n // 4, 2))
    x1b = rng.normal(loc=[0.8, -1.1], scale=0.9, size=(n // 4, 2))
    x2a = rng.normal(loc=[1.2, 1.0], scale=0.9, size=(n // 4, 2))
    x2b = rng.normal(loc=[-0.9, 1.1], scale=0.9, size=(n // 4, 2))
    X = np.vstack([x1a, x1b, x2a, x2b])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    noise_idx = rng.choice(n, size=int(0.08 * n), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=18, alpha=0.85)
    plt.title("Synthetic 2D data (with noise)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig("decision_tree_data.png", dpi=150)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    label = "y"
    cond, _, feature = find_split_ig(df, label)
    root = TreeNode(cond, feature, 0)
    build_tree(df, label, root)
    print_conditions(root)
    print(f"MSE: {mse_report(root, df, label):.4f}")


if __name__ == "__main__":
    main()
