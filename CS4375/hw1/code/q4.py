import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class BostonHousingDataset:
    def __init__(self):
        self.url = "http://lib.stat.cmu.edu/datasets/boston"
        self.feature_names = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
        ]

    def load_dataset(
        self, local_path="housing.xls", use_synthetic=False, n_samples=506, seed=42
    ):
        local_file = Path(local_path)
        if local_file.exists():
            raw_df = pd.read_csv(local_file, sep=r"\s+", header=None)
            data = raw_df.iloc[:, :-1].to_numpy()
            target = raw_df.iloc[:, -1].to_numpy()
        elif use_synthetic:
            rng = np.random.default_rng(seed)
            data = rng.normal(size=(n_samples, len(self.feature_names)))
            true_w = rng.normal(size=len(self.feature_names))
            true_b = rng.normal()
            noise = rng.normal(scale=0.1, size=n_samples)
            target = data @ true_w + true_b + noise
        else:
            # Fetch data from URL
            raw_df = pd.read_csv(self.url, sep=r"\s+", skiprows=22, header=None)
            data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            target = raw_df.values[1::2, 2]

        # Create the dictionary in sklearn format
        dataset = {
            "data": [],
            "target": [],
            "feature_names": self.feature_names,
            "DESCR": "Boston House Prices dataset",
        }

        dataset["data"] = data
        dataset["target"] = target

        return dataset


boston_housing = BostonHousingDataset()
boston_dataset = boston_housing.load_dataset(local_path="housing.xls")


# the loss functions are here for book keepings sake, not used in impl
def mse(w, b, x, y):
    err = w @ x + b - y
    return np.square(err)


def mae(w, b, x, y):
    err = w @ x + b - y
    return np.abs(err)


def grad_mse(w, b, x, y):
    err = w @ x + b - y
    return (2 * err * x, 2 * err)


def grad_mae(w, b, x, y):
    err = w @ x + b - y
    return (np.sign(err) * x, np.sign(err))


def grad_descent(grad_loss, train_x, train_y, epochs, eta):
    w = np.zeros(train_x.shape[1], dtype=float)
    b = 0.0
    n = train_x.shape[0]
    for _ in range(epochs):
        grad_w = np.zeros(train_x.shape[1], dtype=float)
        grad_b = 0.0
        for x, y in zip(train_x, train_y):
            g_w, g_b = grad_loss(w, b, x, y)
            grad_w += g_w
            grad_b += g_b
        w -= (grad_w / n) * eta
        b -= (grad_b / n) * eta
    return w, b


def model_eval(w, b, test_x, test_y):
    preds = test_x @ w + b
    residuals = preds - test_y
    mse = np.mean(residuals**2)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0
    return mse, r2


def main():
    dataset = boston_housing.load_dataset()
    data = np.asarray(dataset["data"], dtype=float)
    target = np.asarray(dataset["target"], dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    w, b = grad_descent(grad_mse, x_train, y_train, 10000, 1e-3)
    print("model w:", w)
    print("model b: ", b)
    train_mse, train_r2 = model_eval(w, b, x_train, y_train)
    mse, r2 = model_eval(w, b, x_test, y_test)
    print("model_train_mse: ", train_mse)
    print("model_train_r2: ", train_r2)
    print("model mse: ", mse)
    print("model r2: ", r2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    sklearn_test_mse = np.mean((model.predict(x_test) - y_test) ** 2)
    print("sklearn_coef:", model.coef_)
    print("sklearn_intercept:", model.intercept_)
    print("sklearn_train_r2:", model.score(x_train, y_train))
    print("sklearn_test_r2:", model.score(x_test, y_test))
    print("sklearn_test_mse:", sklearn_test_mse)


if __name__ == "__main__":
    main()
