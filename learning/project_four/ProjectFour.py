# |%%--%%| <ifZhvXAB22|xctttruDvGC>
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.core.indexes.base import format_object_summary
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

df = pd.read_csv("wine.txt", delimiter="\t")
print(df)

# |%%--%%| <xctttruDvGC|pw7JIN8lec>
X = df.drop(columns=["Quality"])
y = df["Quality"]


# |%%--%%| <pw7JIN8lec|JvBCh4Yjcy>
def loocv(formula, df, y):
    errors = []
    loo = LeaveOneOut()
    for train_idx, test_idx in tqdm(loo.split(df), total=df.shape[0]):
        # Fit on n−1 rows
        df_train = df.iloc[train_idx]
        model = smf.ols(formula=formula, data=df_train).fit()

        # Predict on the held‑out row, extracting the scalar correctly
        df_test = df.iloc[test_idx]
        y_pred = model.predict(df_test).iloc[0]
        y_true = y.iloc[test_idx].values[0]

        errors.append((y_true - y_pred) ** 2)

    return np.mean(errors)


# |%%--%%| <JvBCh4Yjcy|QT4m746jVf>
formula = "Quality ~ Clarity + Aroma + Body + Flavor + Oakiness + C(Region)"
print(loocv(formula, df, y))
# |%%--%%| <QT4m746jVf|oH210JLUQU>

predictors = ["Clarity", "Aroma", "Body", "Flavor", "Oakiness", "C(Region)"]
bestOfCombs = []
for k in range(1, len(predictors)):
    adjustedR2List = []
    for subset in itertools.combinations(predictors, k):
        formula = "Quality ~ " + " + ".join(subset)
        model = smf.ols(formula=formula, data=df).fit()
        adjustedR2List.append((formula, model.rsquared_adj))
    bestOfCombs.append(max(adjustedR2List, key=lambda t: t[1]))

bestOverall = max(bestOfCombs, key=lambda t: t[1])[0]
print(bestOverall)
print(loocv(bestOverall, df, y))

# |%%--%%| <oH210JLUQU|PZ6EMqhguX>
selected = []
predictors = {"Clarity", "Aroma", "Body", "Flavor", "Oakiness", "C(Region)"}
remaining = set(predictors)
current_score = -np.inf
while remaining:
    scores = []
    for candidate in remaining:
        formula = "Quality ~ " + " + ".join(selected + [candidate])
        model = smf.ols(formula=formula, data=df).fit()
        scores.append((candidate, model.rsquared_adj))

    best_candidate, best_score = max(scores, key=lambda x: x[1])

    if best_score <= current_score:
        print("done")
        break

    remaining.remove(best_candidate)
    selected.append(best_candidate)
    current_score = best_score

print(selected)
formula = "Quality ~ " + " + ".join(selected)
print(loocv(formula, df, y))


# |%%--%%| <PZ6EMqhguX|KE5hz0t9vd>
predictors = {"Clarity", "Aroma", "Body", "Flavor", "Oakiness", "C(Region)"}
selected = predictors.copy()
M = []
for k in range(len(predictors) - 1):
    scores = []
    for candidates in itertools.combinations(selected, len(selected) - 1):
        formula = "Quality ~ " + " + ".join(candidates)
        model = smf.ols(formula=formula, data=df).fit()
        scores.append((candidates, model.rsquared_adj))
    scores.sort(key=lambda x: x[1], reverse=True)
    best_candidate, best_score = scores[0]
    M.append((best_candidate, best_score))
    dropped = (set(selected) - set(best_candidate)).pop()
    selected.remove(dropped)

M.sort(key=lambda x: x[1])
best_combination, best_rsquared = M[-1]
back_formula = "Quality ~ " + " + ".join(best_combination)
print(back_formula)
print(loocv(back_formula, df, y))

# |%%--%%| <KE5hz0t9vd|BdazYEeoAw>

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

# StatModels was giving back somewhat illogical results
# the rest of the results checked out with SciKitLearn and R
# Not sure what caused the discrepency


df = pd.read_csv("wine.txt", sep="\t")

response_col = "Quality"
df = pd.get_dummies(df, columns=["Region"], drop_first=True)
predictor_cols = [col for col in df.columns if col != response_col]

X = df[predictor_cols]
y = df[response_col]


def loocv_ridge_mse(X, y, alpha):
    loo = LeaveOneOut()
    preds = []
    actuals = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Ridge(alpha=alpha).fit(X_train, y_train)
        preds.append(model.predict(X_test)[0])
        actuals.append(y_test)
    return mean_squared_error(actuals, preds)


errors = []
alpha_grid = np.logspace(-3, 3, 50)
for alpha in alpha_grid:
    mse_val = loocv_ridge_mse(X, y, alpha)
    errors.append((mse_val, alpha))

print(min(errors))


# |%%--%%| <BdazYEeoAw|YAO6yIcJW3>

df = pd.read_csv("wine.txt", sep="\t")

response_col = "Quality"
df = pd.get_dummies(df, columns=["Region"], drop_first=True)
predictor_cols = [col for col in df.columns if col != response_col]

X = df[predictor_cols]
y = df[response_col]


def loocv_ridge_mse(X, y, alpha):
    loo = LeaveOneOut()
    preds = []
    actuals = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Lasso(alpha=alpha).fit(X_train, y_train)
        preds.append(model.predict(X_test)[0])
        actuals.append(y_test)
    return mean_squared_error(actuals, preds)


errors = []
alpha_grid = np.logspace(-3, 3, 50)
for alpha in alpha_grid:
    mse_val = loocv_ridge_mse(X, y, alpha)
    errors.append((mse_val, alpha))

print(min(errors))
