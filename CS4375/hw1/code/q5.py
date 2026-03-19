from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def grad_hinge(w, b, x, y):
    if y * (w @ x + b) < 1:
        return (-y * x, -y)
    else:
        return np.zeros_like(w), 0


def grad_precep(w, b, x, y):
    if y * (w @ x + b) <= 0:
        return (-y * x, -y)
    else:
        return np.zeros_like(w), 0


# the gradient decent optimizer is the same between this and q4 since
# the functions are all convex, the update rule remains relatively unchanged
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


def plot_classifier(X_train, y_train, X_test, y_test, w, b, title, out_path):
    fig, ax = plt.subplots()
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="bwr",
        edgecolors="k",
        label="train",
    )
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", marker="x", label="test"
    )
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    xx = np.linspace(x_min, x_max, 200)
    if w[1] != 0:
        yy = -(w[0] * xx + b) / w[1]
        ax.plot(xx, yy, "k-", label="decision boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
    y = np.where(y == 0, -1, 1)
    rng = np.random.default_rng(0)
    indices = rng.permutation(X.shape[0])
    split = int(0.8 * X.shape[0])
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    w_hinge, b_hinge = grad_descent(grad_hinge, X_train, y_train, epochs=200, eta=0.01)
    w_precep, b_precep = grad_descent(
        grad_precep, X_train, y_train, epochs=200, eta=0.01
    )

    preds_hinge = np.sign(X_test @ w_hinge + b_hinge)
    preds_hinge[preds_hinge == 0] = 1
    accuracy_hinge = np.mean(preds_hinge == y_test)
    print(f"hinge test accuracy: {accuracy_hinge:.3f}")

    preds_precep = np.sign(X_test @ w_precep + b_precep)
    preds_precep[preds_precep == 0] = 1
    accuracy_precep = np.mean(preds_precep == y_test)
    print(f"perceptron test accuracy: {accuracy_precep:.3f}")

    plot_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        w_hinge,
        b_hinge,
        "Hinge loss classifier",
        "hinge.png",
    )
    plot_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        w_precep,
        b_precep,
        "Perceptron classifier",
        "perceptron.png",
    )


if __name__ == "__main__":
    main()
