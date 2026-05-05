import numpy as np
from tqdm import tqdm

from criteria import MSELoss, MSESplitFinder
from tree import Tree


class GradientBoost:
    def __init__(
        self, label, iterations=50, learning_rate=0.1, n_features=None,
        loss=None, split_finder=None, verbose=True
    ):
        self.label = label
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.loss = loss or MSELoss()
        self.split_finder = split_finder or MSESplitFinder()
        self.verbose = verbose
        self.trees = []
        self.init_pred = None

    def fit(self, df):
        self.init_pred = self.loss.init_prediction(df[self.label])
        curr_pred = np.full(len(df), self.init_pred)
        for _ in tqdm(range(self.iterations), disable=not self.verbose):
            residuals = self.loss.residuals(df[self.label], curr_pred)
            residual_df = df.drop(columns=[self.label]).assign(residual=residuals)
            stump = Tree(
                "residual", max_depth=1, n_features=self.n_features,
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
