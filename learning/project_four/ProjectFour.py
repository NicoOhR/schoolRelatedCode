# |%%--%%| <ifzhvxab22|xctttrudvgc>
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.core.indexes.base import format_object_summary
from sklearn import model_selection
from sklearn.model_selection import leaveoneout
from tqdm import tqdm

df = pd.read_csv("wine.txt", delimiter="\t")
print(df)

# |%%--%%| <xctttrudvgc|pw7jin8lec>
x = df.drop(columns=["quality"])
y = df["quality"]


# |%%--%%| <pw7jin8lec|jvbch4yjcy>
def loocv(formula, df, y):
    errors = []
    loo = leaveoneout()
    for train_idx, test_idx in tqdm(loo.split(df), total=df.shape[0]):
        # fit on n−1 rows
        df_train = df.iloc[train_idx]
        model = smf.ols(formula=formula, data=df_train).fit()
        # predict on the held‑out row, extracting the scalar correctly
        df_test = df.iloc[test_idx]
        y_pred = model.predict(df_test).iloc[0]
        y_true = y.iloc[test_idx].values[0]
        errors.append((y_true - y_pred) ** 2)
    return np.mean(errors)


# |%%--%%| <jvbch4yjcy|qt4m746jvf>
formula = "quality ~ clarity + aroma + body + flavor + oakiness + c(region)"
print(loocv(formula, df, y))
# |%%--%%| <qt4m746jvf|oh210jluqu>

predictors = ["clarity", "aroma", "body", "flavor", "oakiness", "c(region)"]
bestofcombs = []
for k in range(1, len(predictors)):
    adjustedr2list = []
    for subset in itertools.combinations(predictors, k):
        formula = "quality ~ " + " + ".join(subset)
        model = smf.ols(formula=formula, data=df).fit()
        adjustedr2list.append((formula, model.rsquared_adj))
    bestofcombs.append(max(adjustedr2list, key=lambda t: t[1]))

bestoverall = max(bestofcombs, key=lambda t: t[1])[0]
print(bestoverall)
print(loocv(bestoverall, df, y))

# |%%--%%| <oh210jluqu|pz6emqhgux>
selected = []
predictors = {"clarity", "aroma", "body", "flavor", "oakiness", "c(region)"}
remaining = set(predictors)
current_score = -np.inf
while remaining:
    scores = []
    for candidate in remaining:
        formula = "quality ~ " + " + ".join(selected + [candidate])
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
formula = "quality ~ " + " + ".join(selected)
print(loocv(formula, df, y))


# |%%--%%| <pz6emqhgux|ke5hz0t9vd>
predictors = {"clarity", "aroma", "body", "flavor", "oakiness", "c(region)"}
selected = predictors.copy()
m = []
for k in range(len(predictors) - 1):
    scores = []
    for candidates in itertools.combinations(selected, len(selected) - 1):
        formula = "quality ~ " + " + ".join(candidates)
        model = smf.ols(formula=formula, data=df).fit()
        scores.append((candidates, model.rsquared_adj))
    scores.sort(key=lambda x: x[1], reverse=true)
    best_candidate, best_score = scores[0]
    m.append((best_candidate, best_score))
    dropped = (set(selected) - set(best_candidate)).pop()
    selected.remove(dropped)

m.sort(key=lambda x: x[1])
best_combination, best_rsquared = m[-1]
back_formula = "quality ~ " + " + ".join(best_combination)
print(back_formula)
print(loocv(back_formula, df, y))

# |%%--%%| <ke5hz0t9vd|bdazyeeoaw>

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso, ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import leaveoneout

# statmodels was giving back somewhat illogical results
# the rest of the results checked out with scikitlearn and r
# not sure what caused the discrepency


df = pd.read_csv("wine.txt", sep="\t")

response_col = "quality"
df = pd.get_dummies(df, columns=["region"], drop_first=true)
predictor_cols = [col for col in df.columns if col != response_col]

x = df[predictor_cols]
y = df[response_col]


def loocv_ridge_mse(x, y, alpha):
    loo = leaveoneout()
    preds = []
    actuals = []
    for train_idx, test_idx in loo.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = ridge(alpha=alpha).fit(x_train, y_train)
        preds.append(model.predict(x_test)[0])
        actuals.append(y_test)
    return mean_squared_error(actuals, preds)


errors = []
alpha_grid = np.logspace(-3, 3, 200)
for alpha in alpha_grid:
    mse_val = loocv_ridge_mse(x, y, alpha)
    errors.append((mse_val, alpha))

print(min(errors))


# |%%--%%| <bdazyeeoaw|yao6yicjw3>

df = pd.read_csv("wine.txt", sep="\t")

response_col = "quality"
df = pd.get_dummies(df, columns=["region"], drop_first=true)
predictor_cols = [col for col in df.columns if col != response_col]

x = df[predictor_cols]
y = df[response_col]


def loocv_lasso(x, y, alpha):
    loo = leaveoneout()
    preds = []
    actuals = []
    for train_idx, test_idx in loo.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lasso(alpha=alpha).fit(x_train, y_train)
        preds.append(model.predict(x_test)[0])
        actuals.append(y_test)
    return mean_squared_error(actuals, preds)


errors = []
alpha_grid = np.logspace(-3, 3, 50)
for alpha in alpha_grid:
    mse_val = loocv_lasso(x, y, alpha)
    errors.append((mse_val, alpha))

print(min(errors))
