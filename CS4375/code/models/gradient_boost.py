from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tqdm import tqdm

from .criteria import LossFunction, MSELoss, SplitFinder
from .tree import Tree


@dataclass
class GradientBoost:
    """Gradient boosting ensemble of regression trees.

    Parameters
    ----------
    label : str
        Name of the target column in the DataFrame passed to fit/predict.
    iterations : int
        Number of boosting rounds (trees) to fit.
    learning_rate : float
        Shrinkage factor applied to each tree's predictions (eta).
    n_features : int, optional
        Number of features considered at each split. None uses all features.
    loss : LossFunction
        Supplies node impurity, pseudo-residuals, leaf values, and the
        initial prediction. Swap this to change task or objective.
    tree_depth : int
        Maximum depth of each boosting tree.
    verbose : bool
        Show a tqdm progress bar during fitting.
    """

    label: str
    iterations: int = 50
    learning_rate: float = 0.1
    n_features: Optional[int] = None
    loss: LossFunction = field(default_factory=MSELoss)
    tree_depth: int = 1
    verbose: bool = True
    split_finder: SplitFinder = field(default=None, init=False)
    trees: list = field(default_factory=list, init=False)
    init_pred: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        self.split_finder = SplitFinder(self.loss)

    def fit(self, df):
        self.init_pred = self.loss.init_prediction(df[self.label])
        curr_pred = np.full(len(df), self.init_pred)
        for _ in tqdm(range(self.iterations), disable=not self.verbose):
            residuals = self.loss.residuals(df[self.label], curr_pred)
            residual_df = df.drop(columns=[self.label]).assign(residual=residuals)
            stump = Tree(
                "residual",
                max_depth=self.tree_depth,
                n_features=self.n_features,
                split_finder=self.split_finder,
                leaf_fn=self.loss.leaf_value,
            )
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
