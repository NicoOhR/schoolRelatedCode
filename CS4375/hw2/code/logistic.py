import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SkLogisticRegression


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

    def grad_loss(self, x, y):
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
                        self.grad_loss(x, y)
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
    print(df["target"])
    reg = LogisticRegression(df, "target", eta=0.001, epochs=10000)
    reg.train()
    print(reg.mse())


if __name__ == "__main__":
    main()
