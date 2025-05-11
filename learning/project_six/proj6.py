import numpy as np
import pandas as pd
from ISLP import load_data
from pandas.core.arrays import categorical
from pandas.core.common import random_state
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree

Hitters = pd.get_dummies(
    pd.DataFrame(load_data("Hitters")).dropna(subset=["Salary"]),
    columns=["League", "Division", "NewLeague"],
    drop_first=False,
)

tree = DecisionTreeRegressor(random_state=0)

X = Hitters.drop(columns=["Salary"])
y = np.log(Hitters["Salary"])


tree.fit(X, y)

import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree

# Excessively large tree
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True)
plt.show()

# tree_rules = export_text(tree_model)

from sklearn.model_selection import LeaveOneOut, cross_val_score
from tqdm import tqdm

cv_mse = -np.mean(
    cross_val_score(
        tree, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=-1
    )
)

print(cv_mse)

# b)
tree = DecisionTreeRegressor(random_state=0)
path = tree.cost_complexity_pruning_path(X, y)
alphas, impurities = path.ccp_alphas, path.impurities

trees = []
for alpha in tqdm(alphas):
    regressor = DecisionTreeRegressor(random_state=0, ccp_alpha=alpha)
    regressor.fit(X, y)
    trees.append(regressor)


loo = LeaveOneOut()
mse_scores = []
counts = []
for regressor in tqdm(trees):
    scores = cross_val_score(
        regressor, X, y, cv=loo, scoring="neg_mean_squared_error", n_jobs=-1
    )
    mse_scores.append(-scores.mean())
    counts.append(regressor.tree_.node_count)

optimal_index = np.argmin(mse_scores)
optimal_alpha = alphas[optimal_index]
print(f"Optimal ccp_alpha: {optimal_alpha}")
optimal_tree = DecisionTreeRegressor(random_state=0, ccp_alpha=optimal_alpha)
optimal_tree.fit(X, y)
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(optimal_tree, feature_names=X.columns, filled=True)
plt.title("Pruned Decision Tree")
plt.savefig("pruned.png")
plt.figure(figsize=(8, 6))

plt.plot(alphas, mse_scores, marker="o", drawstyle="steps-post")
plt.xlabel("ccp_alpha")
plt.ylabel("LOOCV Mean Squared Error")
plt.title("LOOCV MSE vs. ccp_alpha")
plt.grid(True)
plt.savefig("prunedMSEvsAlpha.png")

plt.figure(figsize=(8, 6))
plt.plot(counts, mse_scores, marker="o")
plt.xlabel("Number of Nodes in Tree")
plt.ylabel("LOOCV Mean Squared Error")
plt.title("Tree Size vs LOOCV MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("prunedSizevsMSE.png")

# Extract feature importances
importances = optimal_tree.feature_importances_
feature_names = X.columns

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Pruned Decision Tree")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("prunedimportance.png")


# c)
from sklearn.ensemble import BaggingRegressor

B = 1000
bagg = BaggingRegressor(n_estimators=B, random_state=0).fit(X, y)
cv_mse = -np.mean(
    cross_val_score(
        bagg, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=-1
    )
)
print(cv_mse)
# Ensure that the base estimator exposes feature_importances_
importances = np.mean([tree.feature_importances_ for tree in bagg.estimators_], axis=0)

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Bagging Regressor")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("baggingImportance.png")

# d
from sklearn.ensemble import RandomForestRegressor

random = RandomForestRegressor(
    n_estimators=B, max_depth=int(np.floor(len(X.columns) / 3)), random_state=0
)
cv_mse = -np.mean(
    cross_val_score(
        random, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=-1
    )
)
print(cv_mse)
# Extract feature importances
importances = random.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Random Forest Regressor")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("randomImportance.png")

# e

from sklearn.ensemble import GradientBoostingRegressor

boost = GradientBoostingRegressor(n_estimators=B, learning_rate=0.01, max_depth=1)
cv_mse = -np.mean(
    cross_val_score(
        boost, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=-1
    )
)
print(cv_mse)
# Extract feature importances
importances = boost.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Gradient Boosting Regressor")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("boostedImportance.png")


# 2)
diabetes = pd.read_csv("diabetes.csv")
diabetes.columns = diabetes.columns.str.replace(" ", "")
diabetes.columns = diabetes.columns.str.replace("\n", "")

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = diabetes.drop(columns="Outcome")
y = diabetes["Outcome"]

scaler = StandardScaler()
scaler.fit_transform(X)

c_logspace = np.logspace(-2, 2, num=15)  # from 0.01 to 100
gamma_values = np.logspace(-15, 3, num=19, base=2)

svc = SVC(kernel="linear")
cv = KFold(n_splits=10)

grid_search = GridSearchCV(
    svc, c_logspace, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

optim_svm = SVC(kernel="linear", C=best_params).fit(X, y)
coefs = optim_svm.coef_[0]
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefs)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance in Linear SVC")
plt.show()

svm = SVC(kernel="poly", degree=2)
grid_search = GridSearchCV(
    svm, c_logspace, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

optim_svm = SVC(kernel="poly", degree=2, C=best_params).fit(X, y)
coefs = optim_svm.coef_[0]
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefs)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance in Linear SVC")
plt.show()

param_grid = {"C": c_logspace, "gamma": gamma_values}

svm = SVC(kernel="rbf")
grid_search = GridSearchCV(
    svm, param_grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
)

best_params = grid_search.best_params_

optim_svm = SVC(kernel="rbf", gamma=best_params["gamma"], C=best_params["C"]).fit(X, y)
coefs = optim_svm.coef_[0]
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefs)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance in Linear SVC")
plt.show()


# 3)
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

Hitters = pd.get_dummies(
    pd.DataFrame(load_data("Hitters")).dropna(subset=["Salary"]),
    columns=["League", "Division", "NewLeague"],
    drop_first=False,
)

std = StandardScaler()
std.fit_transform(Hitters)
Hitters["Salary"] = np.log(Hitters["Salary"])

model = AgglomerativeClustering(
    n_clusters=None, distance_threshold=0, linkage="complete", metric="euclidean"
).fit(Hitters)


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model, truncate_mode="level", p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

from scipy.cluster.hierarchy import fcluster, linkage

Z = linkage(Hitters, method="complete")
labels = fcluster(Z, t=2, criterion="maxclust")
cluster_means = []
for cluster_id in np.unique(labels):
    cluster_data = Hitters[labels == cluster_id]
    cluster_mean = np.mean(cluster_data, axis=0)
    cluster_means.append(cluster_mean)


cluster_means = pd.DataFrame(np.array(cluster_means), columns=Hitters.columns)

cluster_means

# d)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(Hitters)
cluster_means = pd.DataFrame(kmeans.cluster_centers_, columns=Hitters.columns)
