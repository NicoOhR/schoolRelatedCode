import random
from re import S

import matplotlib as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

diabetes = pd.read_csv("diabetes.csv")
diabetes = diabetes.rename(columns=lambda x: x.strip())

feature_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
X = diabetes[feature_cols]
y = diabetes["Outcome"]

diabetes_mod = LogisticRegression(solver="lbfgs", max_iter=1000)
diabetes_mod.fit(X, y)
y_pred = diabetes_mod.predict(X)
c_matrix = confusion_matrix(y, y_pred)


def metrics(confusion_matrix):
    TP, FP, FN, TN = confusion_matrix.ravel()
    error = (FP + FN) / (TP + FP + FN + TN)
    sensitivity = TP / (TP + FN)
    specifity = TN / (TN + FP)
    return (error, sensitivity, specifity)


print(metrics(c_matrix))


##part b
def oneZeroError(prediction, row):
    actual = row.at["Outcome"]
    pred = 0 if prediction <= 0.5 else 1
    return 1 if (pred == actual) else 0


errors = []
for idx, row in tqdm(diabetes.iterrows()):
    X_test = pd.DataFrame([row[feature_cols]])
    dropped = diabetes.drop(idx)
    X = dropped[feature_cols]
    y = dropped["Outcome"]
    diabetes_mod.fit(X, y)
    prediction = diabetes_mod.predict(X_test)
    errors.append(oneZeroError(prediction, row))

print(np.mean(errors))

# Part c
X = diabetes[feature_cols]
y = diabetes["Outcome"]

loo = LeaveOneOut()

y_true, y_pred = [], []

for train_index, test_index in tqdm(loo.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    diabetes_mod.fit(X_train, y_train)

    prediction = diabetes_mod.predict(X_test)

    y_true.append(y_test.values[0])
    y_pred.append(prediction[0])

accuracy = accuracy_score(y_true, y_pred)
print(f"LOOCV Accuracy: {accuracy:.4f}")

##Question 2

blood = pd.read_csv("oxygen_saturation.txt", sep="\t")
blood["agreement"] = np.abs(blood["pos"] - blood["osm"])
# part b

# non-bootstrap estimation of the 90th quantile
quant = np.percentile(blood["agreement"], 90)
print(quant)

estimates = []
for _ in tqdm(range(100000)):
    bootstrap_sample = np.random.choice(blood["agreement"], size=72, replace=True)
    quant_estimate = np.percentile(bootstrap_sample, 90)
    estimates.append(quant_estimate)

bootstrap_quant_estimate = np.mean(estimates)
upper_ci = np.percentile(estimates, 95)
bias = bootstrap_quant_estimate - quant
se = np.std(estimates, ddof=1)
print(bias)
print(se)
print(upper_ci)
print(bootstrap_quant_estimate)


# part c
def quantile_90(x):
    return np.percentile(x, 90)


data = (np.array(blood["agreement"]).reshape(1, -1),)
res = bootstrap(
    data,
    statistic=quantile_90,
    confidence_level=0.95,
    n_resamples=100000,
    method="percentile",
    vectorized=False,
    axis=1,
)
bias = np.mean(res.bootstrap_distribution) - np.quantile(data, 0.9)
print(np.mean(res.bootstrap_distribution))
print(bias)
print(res.confidence_interval)
print(res.standard_error)
