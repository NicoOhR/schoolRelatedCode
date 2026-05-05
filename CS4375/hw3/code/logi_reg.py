from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, log_loss
import numpy as np


class LogisticRegression:
    def __init__(self, df, label, eta=0.01, epochs=1000, lam=None):
        if lam == None:
            self.lam = 0
        else:
            self.lam = lam
        self.data = np.array(df)
        self.label = np.array(label)
        self.eta = eta
        self.epochs = epochs
        self.b = 0
        self.w = np.zeros(self.data.shape[1], dtype=float)

    def forward(self, x):
        return 1 / (1 + np.exp(-(self.b + np.dot(self.w, x))))

    def predict(self, x_list):
        return [1 / (1 + np.exp(-(self.b + np.dot(self.w, x)))) for x in x_list]

    def log_prob(self, x, y):
        p = self.forward(x)
        grad_b = y - p
        grad_w = x * (y - p)
        return (grad_w, grad_b)

    def mse(self, X=None, y=None):
        if X is None:
            X = self.data
        if y is None:
            y = self.label
        preds = np.array([self.forward(x) for x in X])
        return np.mean((preds - y) ** 2)

    def log_loss(self, X=None, y=None):
        if X is None:
            X = self.data
        if y is None:
            y = self.label
        preds = np.array([self.forward(x) for x in X])
        return log_loss(y, preds)

    def train(self):
        for _ in range(self.epochs):
            (grad_w, grad_b) = map(
                sum,
                zip(*[self.log_prob(x, y) for x, y in zip(self.data, self.label)]),
            )
            self.b += self.eta * grad_b
            self.w += self.eta * (grad_w - self.lam * 2 * self.w)


def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logi = LogisticRegression(X_train, y_train)
    logi.train()

    train_mse = logi.mse()
    test_mse = logi.mse(X_test, y_test)
    train_loss = logi.log_loss()
    test_loss = logi.log_loss(X_test, y_test)
    # part 1
    print("=== No Regularization ===")
    print(f"train mse: {train_mse}")
    print(f"test mse:  {test_mse}")
    print(f"train log-loss: {train_loss}")
    print(f"test log-loss:  {test_loss}")

    # part 2
    lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
    print("\n=== L2 Regularization Sweep ===")
    print(f"{'lambda':<10} {'train mse':<14} {'test mse':<14} {'train loss':<14} {'test loss':<14}")
    for lam in lambdas:
        m = LogisticRegression(X_train, y_train, lam=lam)
        m.train()
        print(f"{lam:<10} {m.mse():<14.6f} {m.mse(X_test, y_test):<14.6f} {m.log_loss():<14.6f} {m.log_loss(X_test, y_test):<14.6f}")

    # part 3: polynomial features — find overfitting degree
    print("\n=== Polynomial Features, No Regularization ===")
    print(f"{'degree':<10} {'train mse':<14} {'test mse':<14} {'train loss':<14} {'test loss':<14}")
    poly_results = {}
    for degree in [1, 2, 3]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        scaler2 = StandardScaler()
        X_train_poly = scaler2.fit_transform(X_train_poly)
        X_test_poly = scaler2.transform(X_test_poly)
        m = LogisticRegression(X_train_poly, y_train, eta=0.01, epochs=1000)
        m.train()
        tr_mse = m.mse()
        te_mse = m.mse(X_test_poly, y_test)
        tr_loss = m.log_loss()
        te_loss = m.log_loss(X_test_poly, y_test)
        poly_results[degree] = (X_train_poly, X_test_poly)
        print(f"{degree:<10} {tr_mse:<14.6f} {te_mse:<14.6f} {tr_loss:<14.6f} {te_loss:<14.6f}")

    # part 4: regularize the overfit degree-2 model
    X_train_poly2, X_test_poly2 = poly_results[2]
    print("\n=== Degree-2 Polynomial + L2 Regularization ===")
    print(f"{'lambda':<10} {'train mse':<14} {'test mse':<14} {'train loss':<14} {'test loss':<14}")
    for lam in [0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
        m = LogisticRegression(X_train_poly2, y_train, eta=0.01, epochs=1000, lam=lam)
        m.train()
        print(f"{lam:<10} {m.mse():<14.6f} {m.mse(X_test_poly2, y_test):<14.6f} {m.log_loss():<14.6f} {m.log_loss(X_test_poly2, y_test):<14.6f}")

    y_score = logi.predict(X_test)
    display = RocCurveDisplay.from_predictions(
        y_test,
        y_score,
        name=f"ROC",
        curve_kwargs=dict(color="darkorange"),
        plot_chance_level=True,
        despine=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="",
    )
    display.figure_.savefig("roc_curve.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
