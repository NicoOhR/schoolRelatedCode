import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, df, label, eta=0.01, epochs=1000):
        self.data = df.drop(columns=[label])
        self.label = df[label]
        self.eta = eta
        self.epochs = epochs
        self.b = 0
        self.w = np.zeros(self.data.shape[1], dtype=float)

    def forward(self, x):
        return 1 / (1 + np.exp(-(self.b + np.dot(self.w, x))))

    def log_prob(self, x, y):
        p = self.forward(x)
        grad_b = y - p
        grad_w = x * (y - p)
        return (grad_w, grad_b)

    def mse(self, X=None, y=None):
        if X is None:
            X = self.data.to_numpy()
        if y is None:
            y = self.label.to_numpy()
        preds = np.array([self.forward(x) for x in X])
        return np.mean((preds - y) ** 2)

    def train(self):
        for _ in range(self.epochs):
            (grad_w, grad_b) = map(
                sum,
                zip(
                    *[
                        self.log_prob(x, y)
                        for x, y in zip(self.data.to_numpy(), self.label.to_numpy())
                    ]
                ),
            )
            self.b += self.eta * grad_b
            self.w += self.eta * grad_w


def main():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df = df.loc[df["target"] < 2]
    feature_cols = [iris.feature_names[0], iris.feature_names[2]]
    df = df[feature_cols + ["target"]]

    reg = LogisticRegression(df, "target", eta=0.001, epochs=10000)
    reg.train()
    print(f"MSE: {reg.mse():.4f}")

    w1, w2 = reg.w
    b = reg.b
    print(f"Decision boundary: {w1:.4f} x1 + {w2:.4f} x2 + {b:.4f} = 0")

    x1 = df[feature_cols[0]].to_numpy()
    x2 = df[feature_cols[1]].to_numpy()
    y = df["target"].to_numpy()

    x_min, x_max = x1.min() - 0.5, x1.max() + 0.5
    x_line = np.array([x_min, x_max])
    y_line = -(w1 * x_line + b) / w2 if abs(w2) > 1e-8 else np.full_like(x_line, np.nan)

    plt.figure(figsize=(5, 4))
    plt.scatter(x1, x2, c=y, cmap="viridis", s=18, alpha=0.85)
    plt.plot(x_line, y_line, color="k", lw=1.5)
    plt.title("Logistic regression decision boundary")
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.tight_layout()
    plt.savefig("logistic_boundary.png", dpi=150)


if __name__ == "__main__":
    main()
