from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .criteria import MSEHessLoss, SecondOrderLoss, SplitFinder
from .tree import Tree


@dataclass
class XGBoost:
    """XGBoost ensemble: gradient boosting with second-order loss and regularisation.

    Parameters
    ----------
    label : str
        Name of the target column in the DataFrame passed to fit/predict.
    iterations : int
        Number of boosting rounds (trees) to fit.
    learning_rate : float
        Shrinkage factor applied to each tree's predictions (eta).
    n_features : int, optional
        Number of features considered at each split (column subsampling).
        None uses all features.
    loss : SecondOrderLoss
        Must supply gradients, hessians, leaf values, and the initial
        prediction. lambda_reg and alpha_reg on the loss object control
        L2 and L1 leaf regularisation respectively.
    tree_depth : int
        Maximum depth of each boosting tree.
    subsample : float
        Fraction of training rows sampled (without replacement) before
        fitting each tree. 1.0 disables row subsampling.
    gamma : float
        Minimum gain required to accept a split (structural regularisation).
        Splits whose gain falls below gamma are rejected during tree construction.
    verbose : bool
        Show a tqdm progress bar during fitting.
    """

    label: str
    iterations: int = 50
    learning_rate: float = 0.1
    n_features: Optional[int] = None
    loss: SecondOrderLoss = field(default_factory=MSEHessLoss)
    tree_depth: int = 4
    subsample: float = 1.0
    gamma: float = 0.0
    verbose: bool = True
    split_finder: SplitFinder = field(default=None, init=False)
    trees: list = field(default_factory=list, init=False)
    init_pred: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        self.split_finder = SplitFinder(self.loss, gamma=self.gamma)

    def fit(self, df):
        self.init_pred = self.loss.init_prediction(df[self.label])
        df = df.reset_index(drop=True)
        curr_pred = pd.Series(np.full(len(df), self.init_pred))
        for _ in tqdm(range(self.iterations), disable=not self.verbose):
            train_df = (
                df.sample(frac=self.subsample, replace=False)
                if self.subsample < 1.0
                else df
            )
            train_preds = curr_pred.loc[train_df.index]
            residuals = self.loss.residuals(train_df[self.label], train_preds)
            hessians = self.loss.hessians(train_df[self.label], train_preds)
            node_df = train_df.drop(columns=[self.label]).assign(
                residual=residuals,
                __grad__=residuals,
                __hess__=hessians,
            )
            stump = Tree(
                "residual",
                max_depth=self.tree_depth,
                n_features=self.n_features,
                split_finder=self.split_finder,
                leaf_fn=self.loss.leaf_value,
            )
            stump.fit(node_df)
            self.trees.append(stump)
            curr_pred += (
                self.learning_rate
                * stump.predict(df.drop(columns=[self.label])).to_numpy()
            )

    def predict(self, df):
        pred = np.full(len(df), self.init_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(df).to_numpy()
        return pred


def make_preds(m, df):
    p = m.predict(df)
    return p.to_numpy() if hasattr(p, "to_numpy") else p
