import numpy as np
from abc import ABC, abstractmethod

_RESERVED = {"__grad__", "__hess__"}


class LossFunction(ABC):
    @abstractmethod
    def node_loss(self, df, label) -> float:
        """Loss of a node — lower is better. Used by SplitFinder."""
        pass

    @abstractmethod
    def residuals(self, y, pred):
        """Negative gradient of the loss w.r.t. predictions."""
        pass

    @abstractmethod
    def leaf_value(self, df, label):
        """Optimal constant prediction for a leaf."""
        pass

    @abstractmethod
    def init_prediction(self, y):
        """Initial prediction before any trees are fit."""
        pass


class SplitFinder:
    def __init__(self, loss: LossFunction, gamma=0.0, alpha=0.0):
        self.loss = loss
        self.gamma = gamma
        self.alpha = alpha

    def find_split(self, df, label, n_features=None):
        if len(df) == 0:
            return None, None, None

        cols = [c for c in df.columns if c != label and c not in _RESERVED]
        if n_features is not None:
            cols = np.random.choice(
                cols, size=min(n_features, len(cols)), replace=False
            ).tolist()

        is_second_order = isinstance(self.loss, SecondOrderLoss)

        if is_second_order:
            G_all = df["__grad__"].to_numpy(dtype=float)
            H_all = df["__hess__"].to_numpy(dtype=float)
            G_total = G_all.sum()
            H_total = H_all.sum()
            lam = self.loss.lambda_reg
            alpha_reg = getattr(self.loss, "alpha_reg", 0.0)
            parent_loss = -np.maximum(abs(G_total) - alpha_reg, 0) ** 2 / (H_total + lam)
        else:
            y_all = df[label].to_numpy(dtype=float)
            n = len(y_all)
            y_sum = y_all.sum()
            y2_sum = (y_all ** 2).sum()
            parent_loss = y2_sum - y_sum ** 2 / n

        best_val = None
        best_gain = 0.0  # only accept gains strictly below 0 (after gamma)
        best_col = None

        for col in cols:
            vals = df[col].to_numpy(dtype=float)
            sort_idx = np.argsort(vals, kind="stable")
            sorted_vals = vals[sort_idx]

            # positions where adjacent sorted values differ — valid split boundaries
            boundaries = np.where(np.diff(sorted_vals) > 0)[0]
            if len(boundaries) == 0:
                continue

            if is_second_order:
                cum_g = np.cumsum(G_all[sort_idx])
                cum_h = np.cumsum(H_all[sort_idx])
                G_L = cum_g[boundaries]
                H_L = cum_h[boundaries]
                G_R = G_total - G_L
                H_R = H_total - H_L
                loss_L = -np.maximum(np.abs(G_L) - alpha_reg, 0) ** 2 / (H_L + lam)
                loss_R = -np.maximum(np.abs(G_R) - alpha_reg, 0) ** 2 / (H_R + lam)
            else:
                sorted_y = y_all[sort_idx]
                cum_y  = np.cumsum(sorted_y)
                cum_y2 = np.cumsum(sorted_y ** 2)
                n_L = boundaries + 1
                n_R = n - n_L
                sum_L  = cum_y[boundaries]
                sum2_L = cum_y2[boundaries]
                sum_R  = y_sum - sum_L
                sum2_R = y2_sum - sum2_L
                loss_L = sum2_L - sum_L ** 2 / n_L
                loss_R = sum2_R - sum_R ** 2 / n_R

            gains = loss_L + loss_R - parent_loss + self.gamma
            idx = int(np.argmin(gains))
            if gains[idx] < best_gain:
                best_gain = float(gains[idx])
                best_val = float(
                    (sorted_vals[boundaries[idx]] + sorted_vals[boundaries[idx] + 1]) / 2
                )
                best_col = col

        if best_col is None:
            return None, None, None
        return best_val, best_gain, best_col


class SecondOrderLoss(LossFunction, ABC):
    @abstractmethod
    def hessians(self, y, pred):
        pass


class MSELoss(LossFunction):
    def node_loss(self, df, label):
        y = df[label]
        return ((y - y.mean()) ** 2).sum()

    def residuals(self, y, pred):
        return y - pred

    def leaf_value(self, df, label):
        return df[label].mean()

    def init_prediction(self, y):
        return y.mean()


class MAELoss(LossFunction):
    def node_loss(self, df, label):
        y = df[label]
        return (np.abs(y - np.median(y))).sum()

    def residuals(self, y, pred):
        return np.sign(y - pred)

    def leaf_value(self, df, label):
        return float(np.median(df[label]))

    def init_prediction(self, y):
        return float(np.median(y))


class EntropyLoss(LossFunction):
    """Classification loss — entropy-based node loss for IG splits."""

    def node_loss(self, df, label):
        p = df[label].value_counts(normalize=True)
        return -(p * np.log2(p + 1e-12)).sum() * len(df)

    def residuals(self, y, pred):
        raise NotImplementedError("EntropyLoss is for classification trees only")

    def leaf_value(self, df, label):
        return df[label].mode()[0]

    def init_prediction(self, y):
        return y.mode()[0]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LogLoss(LossFunction):
    """Binary cross-entropy for gradient boosting. Works in log-odds space."""

    def node_loss(self, df, label):
        y = df[label]
        return ((y - y.mean()) ** 2).sum()

    def residuals(self, y, pred):
        return y - _sigmoid(pred)

    def leaf_value(self, df, label):
        return df[label].mean()

    def init_prediction(self, y):
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1 - p)))


class LogLossHess(LogLoss, SecondOrderLoss):
    """Second-order binary cross-entropy for XGBoost."""

    def __init__(self, lambda_reg=1.0, alpha_reg=0.0):
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def hessians(self, y, pred):
        p = _sigmoid(pred)
        return p * (1 - p)

    def node_loss(self, df, label):
        G = df["__grad__"].sum()
        H = df["__hess__"].sum()
        return -np.max(abs(G) - self.alpha_reg, 0) ** 2 / (H + self.lambda_reg)

    def leaf_value(self, df, label):
        G = df["__grad__"].sum()
        H = df["__hess__"].sum()
        return G / (H + self.lambda_reg)


class MSEHessLoss(MSELoss, SecondOrderLoss):
    def __init__(self, lambda_reg=0.0, alpha_reg=0.0):
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def hessians(self, y, pred):
        return np.ones_like(y, dtype=float)

    def node_loss(self, df, label):
        G = df["__grad__"].sum()
        H = df["__hess__"].sum()
        return -np.max(abs(G) - self.alpha_reg, 0) ** 2 / (H + self.lambda_reg)

    def leaf_value(self, df, label):
        G = df["__grad__"].sum()
        H = df["__hess__"].sum()
        return G / (H + self.lambda_reg)
