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
plot_tree(tree, filled=True, max_depth=5)
plt.savefig("unpruned.png")

from sklearn.model_selection import LeaveOneOut, cross_val_score
from tqdm import tqdm

cv_mse = -np.mean(
    cross_val_score(
        tree, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=3
    )
)

print("Unpruned Tree MSE: ", cv_mse)

importances = tree.feature_importances_
feature_names = X.columns  # If X is a DataFrame

feature_importances = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
)

feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances["Feature"], feature_importances["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances in Unpruned Decision Tree")
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.savefig("unprunedImportance.png")
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
        regressor, X, y, cv=loo, scoring="neg_mean_squared_error", n_jobs=3
    )
    mse_scores.append(-scores.mean())
    counts.append(regressor.tree_.node_count)

optimal_index = np.argmin(mse_scores)
optimal_alpha = alphas[optimal_index]
print(f"Optimal ccp_alpha: {optimal_alpha}")
optimal_tree = DecisionTreeRegressor(random_state=0, ccp_alpha=optimal_alpha)
optimal_tree.fit(X, y)

print(
    np.mean(
        cross_val_score(
            optimal_tree, X, y, cv=loo, scoring="neg_mean_squared_error", n_jobs=3
        )
    )
)
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

importances = optimal_tree.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)[::-1]

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
        bagg, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=3
    )
)
print("Bagging MSE", cv_mse)

importances = np.mean([tree.feature_importances_ for tree in bagg.estimators_], axis=0)
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
        random, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=3
    )
)
print("Random Forest MSE: ", cv_mse)
random.fit(X, y)
importances = random.feature_importances_

indices = np.argsort(importances)[::-1]

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
        boost, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=3
    )
)
print("Boosting MSE ", cv_mse)
boost.fit(X, y)
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

from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = diabetes.drop(columns="Outcome")
y = diabetes["Outcome"]

scaler = StandardScaler()
scaler.fit_transform(X)

c_logspace = np.logspace(-2, 2, num=10)  # from 0.01 to 100
gamma_values = np.logspace(-15, 3, num=19, base=2)

param_grid = {"C": c_logspace}

svc = SVC(kernel="linear")
cv = KFold(n_splits=10)

grid_search = GridSearchCV(
    svc, param_grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1, verbose=3
).fit(X, y)

cv_mse = -np.mean(
    cross_val_score(svc, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=3)
)

cv_mse

best_params = grid_search.best_params_
best_score = -grid_search.best_score_
best_params
optim_svm = SVC(kernel="linear", C=best_params["C"]).fit(X, y)
coefs = optim_svm.coef_[0]
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefs)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance in Linear SVC")
plt.savefig("linearImportance.png")

svm = SVC(kernel="poly", degree=2)
grid_search = GridSearchCV(
    svm, param_grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=3, verbose=3
).fit(X, y)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_
print(best_params)
print(best_score)
optim_svm = SVC(kernel="poly", degree=2, C=best_params["C"]).fit(X, y)

cv_mse = -np.mean(
    cross_val_score(
        optim_svm, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=3, verbose=3
    )
)
print(cv_mse)
result = permutation_importance(optim_svm, X, y, n_repeats=30, random_state=0)

importances = result.importances_mean
feature_names = X.columns

plt.barh(feature_names, importances)
plt.xlabel("Mean Decrease in Accuracy")
plt.title("Permutation Feature Importance in SVM with Polynomial Kernel")
plt.savefig("polyImportance.png")

param_grid = {"C": c_logspace, "gamma": gamma_values}

svm = SVC(kernel="rbf")
grid_search = GridSearchCV(
    svm, param_grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=3, verbose=3
).fit(X, y)

best_params = grid_search.best_params_

optim_svm = SVC(kernel="rbf", gamma=best_params["gamma"], C=best_params["C"]).fit(X, y)

cv_mse = -np.mean(
    cross_val_score(optim_svm, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=3)
)
print(cv_mse)

print(best_params)
result = permutation_importance(optim_svm, X, y, n_repeats=30, random_state=0)

importances = result.importances_mean
feature_names = X.columns

plt.barh(feature_names, importances)
plt.xlabel("Mean Decrease in Accuracy")
plt.title("Permutation Feature Importance in SVM with RBF Kernel")
plt.savefig("rbfImportance.png")

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


import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


plt.figure(figsize=(12, 8))
plt.title("Hierarchical Clustering Dendrogram")
# Set orientation to 'left' to display distances on the left
plot_dendrogram(
    model,
    orientation="left",
    truncate_mode="level",
    p=6,
    show_leaf_counts=True,
    leaf_rotation=0,
    leaf_font_size=10,
)
plt.xlabel("Distance")
plt.tight_layout()
plt.savefig("fullDendrogram.png")

from scipy.cluster.hierarchy import fcluster, linkage

Z = linkage(Hitters, method="complete")
labels = fcluster(Z, t=2, criterion="maxclust")
cluster_means = []
for cluster_id in np.unique(labels):
    cluster_data = Hitters[labels == cluster_id]
    cluster_mean = np.mean(cluster_data, axis=0)
    cluster_means.append(cluster_mean)


cluster_means = pd.DataFrame(np.array(cluster_means), columns=Hitters.columns)

cluster_means.T.to_latex("hier.tex", caption="Hierarchical Means", index=True)

# d)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(Hitters)
cluster_means = pd.DataFrame(kmeans.cluster_centers_, columns=Hitters.columns)

cluster_means.T.to_latex("KMeans.tex", caption="K-Means Sample Mean", index=True)
