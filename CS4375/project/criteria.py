import numpy as np
from abc import ABC, abstractmethod


class SplitFinder(ABC):
    @abstractmethod
    def find_split(self, df, label, n_features=None):
        """Return (split_value, score, feature_name) or (None, None, None)."""
        pass


class MSESplitFinder(SplitFinder):
    def find_split(self, df, label, n_features=None):
        if len(df) == 0:
            return None, None, None
        X = df.drop(columns=[label])
        cols = list(X.columns)
        if n_features is not None:
            cols = np.random.choice(
                cols, size=min(n_features, len(cols)), replace=False
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
                y_left = df.loc[left_mask, label]
                y_right = df.loc[right_mask, label]
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


class IGSplitFinder(SplitFinder):
    """Information-gain split finder for classification trees."""

    def find_split(self, df, label, n_features=None):
        if len(df) == 0:
            return None, None, None
        X = df.drop(columns=[label])
        cols = list(X.columns)
        if n_features is not None:
            cols = np.random.choice(
                cols, size=min(n_features, len(cols)), replace=False
            ).tolist()
        p_y = df[label].value_counts(normalize=True)
        h_y = -(p_y * np.log2(p_y)).sum()
        candidates = []
        for col in cols:
            sorted_vals = np.sort(df[col].to_numpy())
            midpoints = [(a + b) / 2 for a, b in zip(sorted_vals, sorted_vals[1:])]
            if not midpoints:
                continue
            best_ig, best_mid = None, None
            for point in midpoints:
                left_mask = df[col] < point
                p_left = left_mask.mean()
                p_right = 1.0 - p_left
                p_y_left = df.loc[left_mask, label].value_counts(normalize=True)
                p_y_right = df.loc[~left_mask, label].value_counts(normalize=True)
                h_left = -(p_y_left * np.log2(p_y_left)).sum()
                h_right = -(p_y_right * np.log2(p_y_right)).sum()
                ig = h_y - (p_left * h_left + p_right * h_right)
                if best_ig is None or ig > best_ig:
                    best_ig, best_mid = ig, point
            if best_mid is not None:
                candidates.append((best_mid, best_ig, col))
        if not candidates:
            return None, None, None
        return max(candidates, key=lambda item: item[1])


class LossFunction(ABC):
    @abstractmethod
    def residuals(self, y, pred):
        """Negative gradient of the loss w.r.t. predictions."""
        pass

    @abstractmethod
    def leaf_value(self, y):
        """Optimal constant prediction for a leaf given target values."""
        pass

    @abstractmethod
    def init_prediction(self, y):
        """Initial prediction before any trees are fit."""
        pass


class MSELoss(LossFunction):
    def residuals(self, y, pred):
        return y - pred

    def leaf_value(self, y):
        return y.mean()

    def init_prediction(self, y):
        return y.mean()


class MAELoss(LossFunction):
    def residuals(self, y, pred):
        return np.sign(y - pred)

    def leaf_value(self, y):
        return float(np.median(y))

    def init_prediction(self, y):
        return float(np.median(y))
