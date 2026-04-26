from sklearn.datasets import load_digits
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# TODO: need to graph the clustering loss over time
class kmeans:
    def __init__(self, k: int, df, epochs=10000, labels=None):
        self.k = k
        self.data = df
        self.epochs = epochs
        self.assignments = np.zeros(len(self.data))
        self.labels = labels
        self.means = np.random.uniform(
            self.data.min(), self.data.max(), size=(self.k, len(self.data.columns))
        )

    def update_assignments(self):
        def get_closest_centroid(row):
            distances = np.linalg.norm(self.means - row.values, axis=1)
            return np.argmin(distances)

        self.assignments = self.data.apply(get_closest_centroid, axis=1).values

    def update_means(self):
        grouped = (
            self.data.assign(assignments=self.assignments.astype(int))
            .groupby("assignments")
            .mean()
        )
        for cluster in range(self.k):
            if cluster in grouped.index:
                self.means[cluster] = grouped.loc[cluster].values
            else:
                self.means[cluster] = self.data.sample(1).values[0]

    def _map_assignments(self):
        mapped = np.zeros_like(self.assignments, dtype=int)
        for cluster in range(self.k):
            mask = self.assignments == cluster
            if mask.any():
                mapped[mask] = np.bincount(self.labels[mask]).argmax()
        return mapped

    def misclassification_rates(self):
        mapped = self._map_assignments()
        return {
            label: (mapped[self.labels == label] != label).mean()
            for label in np.unique(self.labels)
        }

    def log_loss(self):
        def row_probs(row):
            distances = np.linalg.norm(self.means - row.values, axis=1)
            neg_dist = -distances
            exp = np.exp(neg_dist - neg_dist.max())
            return exp / exp.sum()

        probs = np.stack(self.data.apply(row_probs, axis=1).values)
        return sklearn_log_loss(self.labels, probs)

    def plot_clusters(self):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.data.values)
        reduced_centroids = pca.transform(self.means)
        mapped = self._map_assignments()
        centroid_labels = np.array(
            [
                (
                    np.bincount(self.labels[self.assignments == c]).argmax()
                    if (self.assignments == c).any()
                    else 0
                )
                for c in range(self.k)
            ]
        )

        norm = plt.Normalize(vmin=0, vmax=9)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(
            reduced[:, 0], reduced[:, 1], c=mapped, cmap="tab10", norm=norm, s=5
        )
        ax1.scatter(
            reduced_centroids[:, 0],
            reduced_centroids[:, 1],
            c=centroid_labels,
            cmap="tab10",
            norm=norm,
            s=200,
            marker="X",
            edgecolors="black",
            linewidths=0.5,
        )
        ax1.set_title("cluster assignments")
        ax2.scatter(
            reduced[:, 0], reduced[:, 1], c=self.labels, cmap="tab10", norm=norm, s=5
        )
        ax2.set_title("true labels")
        plt.tight_layout()
        plt.show()

    def train(self):
        self.update_assignments()
        for _ in tqdm(range(self.epochs)):
            self.update_means()
            self.update_assignments()


if __name__ == "__main__":
    digits = load_digits()
    df = pd.DataFrame(digits.data)
    model = kmeans(k=10, df=df, labels=digits.target)
    model.train()
    print(f"log loss: {model.log_loss():.4f}")
    rates = model.misclassification_rates()
    for label, rate in rates.items():
        print(f"  digit {label}: {rate:.2%} misclassified")
    model.plot_clusters()
