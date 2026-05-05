import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class TreeNode:
    def __init__(self, cond, feature, lvl):
        self.condition = cond
        self.condition_feature = feature
        self.level = lvl
        self.prediction = None
        self.left = None
        self.right = None

    def next(self, x_i):
        return (
            self.left if (x_i[self.condition_feature] < self.condition) else self.right
        )


class DecisionTree:
    def __init__(self, label, max_depth=2, n_features=None):
        self.label = label
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def find_split_mse(self, df):
        if len(df) == 0:
            return None, None, None
        X = df.drop(columns=[self.label])
        cols = list(X.columns)
        if self.n_features is not None:
            cols = np.random.choice(
                cols, size=min(self.n_features, len(cols)), replace=False
            ).tolist()
        best = None
        for col in cols:
            sorted_vals = np.sort(df[col].unique())
            midpoints = [(a + b) / 2 for a, b in zip(sorted_vals, sorted_vals[1:])]
            if not midpoints:
                continue
            for val in midpoints:
                left_mask = df[col] < val
                right_mask = ~left_mask
                y_left = df.loc[left_mask, self.label]
                y_right = df.loc[right_mask, self.label]
                y_left_mean = y_left.mean()
                y_right_mean = y_right.mean()
                total = ((y_left - y_left_mean) ** 2).sum() + (
                    (y_right - y_right_mean) ** 2
                ).sum()
                if best is None or total < best[1]:
                    best = (val, total, col)
        if best is None:
            return None, None, None
        return best

    def build_tree(self, df, curr):
        if curr is None or curr.condition is None or curr.condition_feature is None:
            return
        curr.prediction = df[self.label].mean()
        if curr.level == self.max_depth:
            return
        left_subset = df.loc[df[curr.condition_feature] < curr.condition]
        left_cond, _, left_feature = self.find_split_mse(left_subset)
        if left_cond is not None and left_feature is not None:
            curr.left = TreeNode(left_cond, left_feature, curr.level + 1)
            self.build_tree(left_subset, curr.left)

        right_subset = df.loc[df[curr.condition_feature] >= curr.condition]
        right_cond, _, right_feature = self.find_split_mse(right_subset)
        if right_cond is not None and right_feature is not None:
            curr.right = TreeNode(right_cond, right_feature, curr.level + 1)
            self.build_tree(right_subset, curr.right)

    def fit(self, df):
        cond, _, feature = self.find_split_mse(df)
        if cond is not None and feature is not None:
            self.root = TreeNode(cond, feature, 0)
            self.build_tree(df, self.root)

    def predict_one(self, x_i):
        node = self.root
        while node and (node.left or node.right):
            next_node = node.next(x_i)
            if next_node is None:
                break
            node = next_node
        return None if node is None else node.prediction

    def predict(self, df):
        return df.apply(lambda row: self.predict_one(row), axis=1)

    def print_conditions(self, node=None, indent=""):
        if node is None:
            node = self.root
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
            self.print_conditions(node.left, indent + "  ")
            self.print_conditions(node.right, indent + "  ")


class RandomForest:
    def __init__(self, label, n_trees=50, max_depth=4, n_features=None, verbose=True):
        self.label = label
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.verbose = verbose
        self.trees = []

    def fit(self, df):
        n_features = self.n_features
        for _ in tqdm(range(self.n_trees), disable=not self.verbose):
            bootstrap = df.sample(len(df), replace=True)
            tree = DecisionTree(
                self.label, max_depth=self.max_depth, n_features=n_features
            )
            tree.fit(bootstrap)
            self.trees.append(tree)

    def predict(self, df):
        preds = np.stack([tree.predict(df).to_numpy() for tree in self.trees])
        return preds.mean(axis=0)


class GradientBoost:
    def __init__(
        self, label, iterations=50, learning_rate=0.1, n_features=None, verbose=True
    ):
        self.label = label
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.verbose = verbose
        self.trees = []
        self.init_pred = None

    def fit(self, df):
        self.init_pred = df[self.label].mean()
        curr_pred = np.full(len(df), self.init_pred)
        for _ in tqdm(range(self.iterations), disable=not self.verbose):
            residuals = df[self.label] - curr_pred
            residual_df = df.drop(columns=[self.label]).assign(residual=residuals)
            stump = DecisionTree("residual", max_depth=1, n_features=self.n_features)
            stump.fit(residual_df)
            self.trees.append(stump)
            curr_pred += self.learning_rate * stump.predict(residual_df).to_numpy()

    def predict(self, df):
        pred = np.full(len(df), self.init_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(df).to_numpy()
        return pred


def make_preds(m, df):
    p = m.predict(df)
    return p.to_numpy() if hasattr(p, "to_numpy") else p


def main():
    import matplotlib

    matplotlib.use("Agg")

    n_datasets = 100
    sigma = 0.3
    plot_idx = 42
    rng = np.random.default_rng(0)

    x_test = np.linspace(0, 1, 1000)
    df_test = pd.DataFrame({"x": x_test})
    y_true = np.sin(2 * np.pi * x_test)

    model_fns = {
        "tree (depth 4)": lambda: DecisionTree(label="y", max_depth=4),
        "forest (10)": lambda: RandomForest(
            label="y", n_trees=10, max_depth=4, verbose=False
        ),
        "boost (10, lr=0.3)": lambda: GradientBoost(
            label="y", iterations=10, learning_rate=0.3, verbose=False
        ),
    }

    all_preds = {k: np.zeros((n_datasets, len(x_test))) for k in model_fns}
    plot_preds = {}
    plot_df = None

    for i in tqdm(range(n_datasets), desc="datasets"):
        x = rng.uniform(0, 1, 100)
        y = np.sin(2 * np.pi * x) + rng.normal(0, sigma, 100)
        df = pd.DataFrame({"x": x, "y": y})
        if i == plot_idx:
            plot_df = df
        for name, make_model in model_fns.items():
            m = make_model()
            m.fit(df)
            preds = make_preds(m, df_test)
            all_preds[name][i] = preds
            if i == plot_idx:
                plot_preds[name] = preds

    print(f"\n{'Model':<25} {'Bias²':>8} {'Variance':>10} {'MSE':>8}")
    print("-" * 55)
    for name, preds in all_preds.items():
        mean_pred = preds.mean(axis=0)
        bias2 = float(np.mean((mean_pred - y_true) ** 2))
        variance = float(np.mean(preds.var(axis=0)))
        mse = float(np.mean((preds - y_true) ** 2))
        print(f"{name:<25} {bias2:>8.4f} {variance:>10.4f} {mse:>8.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, name in zip(axes, model_fns):
        mean_pred = all_preds[name].mean(axis=0)
        ax.scatter(
            plot_df["x"], plot_df["y"], s=3, alpha=0.3, color="gray", label="train"
        )
        ax.plot(x_test, y_true, "k-", linewidth=1.5, label="true f(x)")
        ax.plot(x_test, plot_preds[name], "r-", linewidth=1.5, label="single fit")
        ax.plot(x_test, mean_pred, "b--", linewidth=1.5, label="mean pred")
        ax.set_title(name)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("bias_variance.png", bbox_inches="tight")
    print("saved bias_variance.png")


"""
Model                        Bias²   Variance      MSE
-------------------------------------------------------
tree (depth 4)              0.0015     0.0316   0.0331
forest (10)                 0.0015     0.0191   0.0207
boost (50, lr=0.3)          0.0078     0.0164   0.0242
"""

if __name__ == "__main__":
    main()
