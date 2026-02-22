import numpy as np
import pandas as pd
from pathlib import Path


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


def mse(w, b, x, y):
    err = np.dot(w, x) + b - y
    return np.square(err)


def grad_mse(w, b, x, y):
    err = np.dot(w, x) + b - y
    return (2 * err * x, 2 * err)


def mae(w, b, x, y):
    err = np.dot(w, x) + b - y
    return np.abs(err)


def grad_mae(w, b, x, y):
    err = np.dot(w, x) + b - y
    return (np.sign(err) * x, np.sign(err))


def reg(w, b, x):
    return np.dot(w, x) + b


def grad_decent(grad_loss, dataset, epochs):
    print("in grad")
    eta = 1e-4
    data = np.asarray(dataset["data"], dtype=float)
    target = np.asarray(dataset["target"], dtype=float)
    # Normalize features to stabilize updates.
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    data = (data - mean) / std
    w = np.zeros(data.shape[1], dtype=float)
    b = 0.0
    n = data.shape[0]
    for _ in range(epochs):
        grad_w = np.zeros(data.shape[1], dtype=float)
        grad_b = 0.0
        for x, y in zip(data, target):
            g_w, g_b = grad_loss(w, b, x, y)
            grad_w += g_w
            grad_b += g_b
        w -= (grad_w / n) * eta
        b -= (grad_b / n) * eta
    return w, b


def main():
    dataset = boston_housing.load_dataset()
    w, b = grad_decent(grad_mse, dataset, epochs=1000)
    # print("final_mse:", history[-1])
    print("w_shape:", w.shape)
    print("b:", b)


if __name__ == "__main__":
    main()
