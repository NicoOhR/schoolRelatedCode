from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .criteria import MSELoss, SplitFinder


def _mean_leaf(df, label):
    return df[label].mean()


class TreeNode:
    def __init__(self, cond, feature, lvl):
        self.condition = cond
        self.condition_feature = feature
        self.level = lvl
        self.prediction = None
        self.left = None
        self.right = None

    def next(self, x_i):
        return self.left if x_i[self.condition_feature] < self.condition else self.right


@dataclass
class Tree:
    """Single CART decision tree.

    Parameters
    ----------
    label : str
        Name of the target column in the DataFrame passed to fit/predict.
    max_depth : int
        Maximum number of splits from root to any leaf.
    n_features : int, optional
        Number of features to consider at each split (column subsampling).
        None uses all features.
    split_finder : SplitFinder, optional
        Controls split scoring and the gamma pruning threshold.
        Defaults to SplitFinder(MSELoss()).
    leaf_fn : callable, optional
        f(df, label) -> float that computes the leaf output value.
        Defaults to the column mean.
    """

    label: str
    max_depth: int = 2
    n_features: Optional[int] = None
    split_finder: Optional[SplitFinder] = None
    leaf_fn: Optional[Callable] = None
    root: Optional[TreeNode] = field(default=None, init=False)

    def __post_init__(self):
        if self.split_finder is None:
            self.split_finder = SplitFinder(MSELoss())
        if self.leaf_fn is None:
            self.leaf_fn = _mean_leaf

    def find_split(self, df):
        return self.split_finder.find_split(df, self.label, self.n_features)

    def build_tree(self, df, curr):
        if curr is None or curr.condition is None or curr.condition_feature is None:
            return
        curr.prediction = self.leaf_fn(df, self.label)
        if curr.level == self.max_depth:
            return
        left_subset = df.loc[df[curr.condition_feature] < curr.condition]
        left_cond, _, left_feature = self.find_split(left_subset)
        if left_cond is not None and left_feature is not None:
            curr.left = TreeNode(left_cond, left_feature, curr.level + 1)
            self.build_tree(left_subset, curr.left)

        right_subset = df.loc[df[curr.condition_feature] >= curr.condition]
        right_cond, _, right_feature = self.find_split(right_subset)
        if right_cond is not None and right_feature is not None:
            curr.right = TreeNode(right_cond, right_feature, curr.level + 1)
            self.build_tree(right_subset, curr.right)

    def fit(self, df):
        cond, _, feature = self.find_split(df)
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
        if self.root is None:
            return pd.Series(np.nan, index=df.index)
        col_idx = {col: i for i, col in enumerate(df.columns)}
        arr = df.to_numpy()
        results = np.empty(len(arr))
        for i in range(len(arr)):
            node = self.root
            while node and (node.left or node.right):
                next_node = (
                    node.left
                    if arr[i, col_idx[node.condition_feature]] < node.condition
                    else node.right
                )
                if next_node is None:
                    break
                node = next_node
            results[i] = np.nan if node is None else node.prediction
        return pd.Series(results, index=df.index)

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
