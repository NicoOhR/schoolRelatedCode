import pathlib
from ucimlrepo import fetch_ucirepo
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

REPORT = pathlib.Path(__file__).parent / "report"
TAB, FIG = REPORT / "tables", REPORT / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

bike_sharing = fetch_ucirepo(id=275)
X = bike_sharing.data.features
y = bike_sharing.data.targets
df = pd.concat([X,y], axis=1)

num = X.select_dtypes("number")          
g = sns.displot(num.melt(var_name="feature"), x="value",
                col="feature", col_wrap=4, height=2.5,
                bins=30, common_bins=False,
                facet_kws=dict(sharex=False, sharey=False))
g.set_titles("{col_name}")
g.set_axis_labels("", "count")
g.savefig(FIG / "distributions.pdf", bbox_inches="tight")
plt.show()

#clearly non linear + trends, can't use day of year
# explicit fig/ax: the displot above owns the current figure, and a bare
# sns.lineplot(...) would draw into its last facet instead of a new plot
fig, ax = plt.subplots(figsize=(9, 3.5))
sns.lineplot(data=df, x=pd.to_datetime(df.dteday), y="cnt", ax=ax, lw=.7)
ax.set_xlabel("")
ax.set_ylabel("hourly count")
fig.savefig(FIG / "timeseries.pdf", bbox_inches="tight")
plt.show()

X = X.drop(columns=["dteday"])

#workingday would be perfectly colinear with weekday + holiday, so at least one can be dropped, more interesting to talk about holiday than workingday
#yr has to be kept since it's a level shift of the ridership, so inorder to discuss the others holding year constant, it must be included
#temp and atemp are very highly correlated, atemp is a function of temp humidity and windspeed, so either we use just that or the other three
#the choice between month and season is somewhat arbitrary but only one can stay.
#weathersit has two partly cloudy entries, 4 and 3, and 4 is very rarely observed, so we merge it into 3, interpreting that as the "bad weather" 
X = X.assign(weathersit=X.weathersit.replace(4, 3))   # .replace() is not in-place
enc = OneHotEncoder(sparse_output=False, drop="first").set_output(transform="pandas")
cats = ['season', 'hr', 'weekday', 'weathersit']
X_cleaned = pd.concat([X.drop(columns=cats), enc.fit_transform(X[cats])], axis=1).drop(columns=['workingday', 'atemp', 'mnth'])
scalar = StandardScaler().set_output(transform="pandas").fit(X_cleaned)
X_scaled = scalar.transform(X_cleaned)
yv = y.to_numpy().ravel()

# center=0 matters here: correlations are signed, and a sequential palette
# would render -0.7 and +0.7 as similar colours
plt.figure(figsize=(9, 8))
sns.heatmap(X_scaled.corr(), center=0, cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, cbar_kws=dict(shrink=.6))
plt.tick_params(labelsize=6)
plt.savefig(FIG / "corr_heatmap.pdf", bbox_inches="tight")
plt.show()

Z = StandardScaler().fit_transform(X[["temp", "atemp", "hum", "windspeed"]])
pca = PCA().fit(Z)

eigenvalues = pca.explained_variance_

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', color='darkred')
plt.axhline(y=1, color='black', linestyle='--', label='Kaiser Criterion Cutoff (y=1)')
plt.title('Scree Plot (Eigenvalues)', fontsize=14)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue Size')
plt.xticks(range(1, len(eigenvalues) + 1))
plt.legend()
plt.grid(True)
plt.savefig(FIG / "scree.pdf", bbox_inches="tight")
plt.show()

print(pca.components_)
print(pca.explained_variance_ratio_)

# ---- prediction: tuned SGD ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, yv, test_size=0.2, random_state=42)

parameters = {'penalty': ('l1', 'l2'), 'alpha': np.logspace(-5, 3, num=100)}
clf = GridSearchCV(SGDRegressor(max_iter=5000, tol=1e-4, random_state=42),
                   parameters, verbose=1)
clf.fit(X_train, y_train)
final_model = clf.best_estimator_        

pred = final_model.predict(X_test)
print(f"""
SGDRegressor
  best params   {clf.best_params_}
  CV R2         {clf.best_score_:.4f}
  test R2       {r2_score(y_test, pred):.4f}
  test RMSE     {mean_squared_error(y_test, pred) ** 0.5:.1f} riders/hour
  test MAE      {mean_absolute_error(y_test, pred):.1f} riders/hour
""")

print(pd.Series(final_model.coef_, index=X_cleaned.columns)
        .sort_values(key=abs, ascending=False).round(2))

X_sm = sm.add_constant(X_scaled, prepend=False)
mod = sm.OLS(yv, X_sm)
res = mod.fit()
print(res.summary())

sd = pd.Series(scalar.scale_, index=X_cleaned.columns)
hour_cols = [c for c in X_cleaned.columns if c.startswith("hr_")]
hour_effect = pd.concat([pd.Series({"hr_0": 0.0}), res.params[hour_cols] / sd[hour_cols]])
hour_effect.index = [int(c.split("_")[1]) for c in hour_effect.index]
hour_effect = hour_effect.sort_index()
mean_by_hour = df.groupby("hr")["cnt"].mean()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(mean_by_hour.index, mean_by_hour.values, color="steelblue", alpha=.65,
       label="observed mean ridership")
ax.set_xlabel("hour of day")
ax.set_ylabel("mean count")
ax.set_xticks(range(24))
ax2 = ax.twinx()
ax2.axhline(0, color="darkred", ls=":", lw=.8)
ax2.plot(hour_effect.index, hour_effect.values, color="darkred", marker="o",
         ms=4, lw=1.5, label="OLS hour coefficient")
ax2.set_ylabel("coefficient (riders/hour, vs. midnight)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left")
fig.savefig(FIG / "hour_effects.pdf", bbox_inches="tight")
plt.show()

X_tr_sm = sm.add_constant(X_train, prepend=False)
X_te_sm = sm.add_constant(X_test, prepend=False)
k = X_train.shape[1]
sst_te = ((y_test - y_test.mean()) ** 2).sum()

rows = []
for alpha in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    for l1_wt in [0.0, 0.15, 0.5, 0.85, 1.0]:
        a_vec = 0.0 if alpha == 0 else np.r_[np.full(k, alpha), 0.0]
        fit_r = sm.OLS(y_train, X_tr_sm).fit_regularized(alpha=a_vec, L1_wt=l1_wt)
        pred_r = np.asarray(fit_r.predict(X_te_sm))
        rows.append(dict(
            alpha=alpha, L1_wt=l1_wt,
            test_R2=1 - ((y_test - pred_r) ** 2).sum() / sst_te,
            test_MSE=((y_test - pred_r) ** 2).mean(),
            test_RMSE=((y_test - pred_r) ** 2).mean() ** 0.5,
            nonzero=int((np.abs(fit_r.params) > 1e-8).sum()),
        ))

reg_results = pd.DataFrame(rows)
print("\nfit_regularized: held-out R2 by (alpha, L1_wt)")
print(reg_results.pivot(index="alpha", columns="L1_wt", values="test_R2").round(4).to_string())

best = reg_results.loc[reg_results.test_R2.idxmax()]
print(f"""
best regularized OLS
  alpha         {best.alpha:g}
  L1_wt         {best.L1_wt:g}
  test R2       {best.test_R2:.4f}
  test MSE      {best.test_MSE:.1f}
  test RMSE     {best.test_RMSE:.1f} riders/hour
  nonzero       {int(best.nonzero)}/{k + 1}
""")

qq = sm.qqplot(res.resid, line="s")
plt.title("Q-Q plot of OLS residuals")
qq.savefig(FIG / "qqplot.pdf", bbox_inches="tight")
plt.show()

def vif_table(frame):
    Z = sm.add_constant(frame.astype(float), prepend=False)
    return pd.Series(
        [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])],
        index=Z.columns).drop("const")

X_candidate = pd.concat([X.drop(columns=cats), enc.transform(X[cats])], axis=1)
plain = [c for c in X_candidate.columns if c in X.columns]

before, after = vif_table(X_candidate), vif_table(X_cleaned)
print("\nVIF of the non-dummy predictors")
print(pd.DataFrame({"before drops": before.reindex(plain),
                    "after drops":  after.reindex(plain)}).round(2).to_string())
print(f"\nmax VIF over dummy levels: before {before.drop(plain, errors='ignore').max():.2f}, "
      f"after {after.drop(plain, errors='ignore').max():.2f}")
print(f"condition number: before {np.linalg.cond(sm.add_constant(X_candidate.astype(float))):.1f}, "
      f"after {np.linalg.cond(sm.add_constant(X_cleaned.astype(float))):.1f}")


# ======================================================================
# LaTeX output for report/report.tex
# ======================================================================

def esc(s):
    """LaTeX-escape a label (column names carry underscores: hr_1, season_2).

    Strings already containing $ or a backslash are passed through untouched:
    they are hand-written LaTeX (e.g. r"$\\alpha$") and must not be re-escaped.
    """
    s = str(s)
    if "$" in s or "\\" in s:
        return s
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def _cell(v):
    return f"{v:g}" if isinstance(v, float) else (v if isinstance(v, str) else esc(v))


def write(name, body):
    (TAB / f"{name}.tex").write_text(body)
    print(f"  tables/{name}.tex")


def to_tex(frame, name, caption, label, longtable=False):
    """Emit a booktabs table. Hand-rolled because pandas' to_latex now routes
    through Styler, which needs jinja2."""
    cols = [esc(c) for c in frame.columns]
    head = " & ".join([esc(frame.index.name or "")] + cols) + r" \\"
    align = "l" + "r" * len(cols)
    rows = [" & ".join([esc(i)] + [_cell(v) for v in r]) + r" \\"
            for i, r in zip(frame.index, frame.to_numpy())]

    if longtable:
        body = [rf"\begin{{longtable}}{{{align}}}",
                rf"\caption{{{caption}}}\label{{{label}}}\\",
                r"\toprule", head, r"\midrule", r"\endfirsthead",
                r"\toprule", head, r"\midrule", r"\endhead",
                rf"\midrule \multicolumn{{{len(cols)+1}}}{{r}}{{\footnotesize continued}} \\",
                r"\endfoot", r"\bottomrule", r"\endlastfoot",
                *rows, r"\end{longtable}"]
    else:
        body = [r"\begin{table}[htbp]", r"\centering",
                rf"\caption{{{caption}}}\label{{{label}}}",
                rf"\begin{{tabular}}{{{align}}}", r"\toprule", head,
                r"\midrule", *rows, r"\bottomrule",
                r"\end{tabular}", r"\end{table}"]
    write(name, "\n".join(body) + "\n")


def to_tex_twocol(frame, name, caption, label, gap="2.5em"):
    """Same table, split into two side-by-side blocks so it fits on one page."""
    half = (len(frame) + 1) // 2
    left, right = frame.iloc[:half], frame.iloc[half:]
    cols = [esc(c) for c in frame.columns]
    hdr = " & ".join([esc(frame.index.name or "")] + cols)
    block = "l" + "r" * len(cols)
    align = block + "@{\\hskip " + gap + "}" + block

    rows, lv, rv = [], left.to_numpy(), right.to_numpy()
    for i in range(half):
        cells = [esc(left.index[i])] + [_cell(v) for v in lv[i]]
        cells += ([esc(right.index[i])] + [_cell(v) for v in rv[i]]
                  if i < len(right) else [""] * (len(cols) + 1))
        rows.append(" & ".join(cells) + r" \\")

    body = [r"\begin{table}[htbp]", r"\centering",
            rf"\caption{{{caption}}}\label{{{label}}}",
            rf"\begin{{tabular}}{{{align}}}", r"\toprule",
            hdr + " & " + hdr + r" \\", r"\midrule", *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(name, "\n".join(body) + "\n")


print("\nwriting LaTeX fragments:")

# --- PCA on the weather block ---
ratio = pca.explained_variance_ratio_
pca_var = pd.DataFrame({
    "eigenvalue":        [f"{v:.4f}" for v in eigenvalues],
    "proportion of var": [f"{v:.4f}" for v in ratio],
    "cumulative":        [f"{v:.4f}" for v in ratio.cumsum()],
}, index=[f"PC{i+1}" for i in range(len(ratio))])
pca_var.index.name = "component"
to_tex(pca_var, "pca_variance",
       "PCA on the standardised weather block (\\texttt{temp}, \\texttt{atemp}, "
       "\\texttt{hum}, \\texttt{windspeed}): eigenvalues and explained variance.",
       "tab:pca-variance")

pca_load = pd.DataFrame(pca.components_.T, index=["temp", "atemp", "hum", "windspeed"],
                        columns=[f"PC{i+1} ({v:.1%})" for i, v in enumerate(ratio)]).round(3)
pca_load.index.name = "variable"
to_tex(pca_load, "pca_loadings",
       "PCA loadings; each column is one component, headed by its share of total variance.",
       "tab:pca-loadings")

# --- SGD ---
sgd_metrics = pd.DataFrame({"value": {
    "penalty":    clf.best_params_["penalty"],
    "alpha":      f"{clf.best_params_['alpha']:.3e}",
    "CV $R^2$":   f"{clf.best_score_:.4f}",
    "test $R^2$": f"{r2_score(y_test, pred):.4f}",
    "test RMSE":  f"{mean_squared_error(y_test, pred) ** 0.5:.1f}",
    "test MAE":   f"{mean_absolute_error(y_test, pred):.1f}",
    "test MSE":   f"{mean_squared_error(y_test, pred):.1f}",
}})
sgd_metrics.index.name = "statistic"
to_tex(sgd_metrics, "sgd_metrics",
       "Tuned SGDRegressor: selected hyper-parameters and held-out performance.",
       "tab:sgd-metrics")

sgd_coefs = (pd.Series(final_model.coef_, index=X_cleaned.columns)
               .sort_values(key=abs, ascending=False).round(3).to_frame("coefficient"))
sgd_coefs.index.name = "predictor"
to_tex_twocol(sgd_coefs, "sgd_coefs",
              "SGDRegressor coefficients on standardised predictors, by magnitude.",
              "tab:sgd-coefs")

# --- OLS fit() ---
jb, jbpv, skew, kurt = sm.stats.jarque_bera(res.resid)
fit_stats = pd.DataFrame({"value": {
    "observations":     f"{res.nobs:.0f}",
    "df model":         f"{res.df_model:.0f}",
    "df residuals":     f"{res.df_resid:.0f}",
    "$R^2$":            f"{res.rsquared:.4f}",
    "adjusted $R^2$":   f"{res.rsquared_adj:.4f}",
    "F-statistic":      f"{res.fvalue:.1f}",
    "Prob (F)":         f"{res.f_pvalue:.3e}",
    "log-likelihood":   f"{res.llf:.1f}",
    "AIC":              f"{res.aic:.1f}",
    "BIC":              f"{res.bic:.1f}",
    "Durbin--Watson":   f"{sm.stats.durbin_watson(res.resid):.4f}",
    "Jarque--Bera":     f"{jb:.1f}",
    "Prob (JB)":        f"{jbpv:.3e}",
    "skew":             f"{skew:.4f}",
    "kurtosis":         f"{kurt:.4f}",
    "condition number": f"{res.condition_number:.1f}",
}})
fit_stats.index.name = "statistic"
to_tex(fit_stats, "ols_fitstats",
       "OLS (\\texttt{fit}) overall fit statistics and residual diagnostics.",
       "tab:ols-fitstats")

ci = res.conf_int()
ols_coefs = pd.DataFrame({
    "coef":      res.params.round(3),
    "std err":   res.bse.round(3),
    "t":         res.tvalues.round(3),
    "$P>|t|$":   res.pvalues.map(lambda v: f"{v:.3e}" if v < 1e-3 else f"{v:.3f}"),
    "[0.025":    ci.iloc[:, 0].round(3),
    "0.975]":    ci.iloc[:, 1].round(3),
})
ols_coefs.index.name = "predictor"
to_tex(ols_coefs, "ols_coefs",
       "OLS (\\texttt{fit}) coefficient table with standard errors, $t$-statistics, "
       "$p$-values and 95\\% confidence intervals.", "tab:ols-coefs", longtable=True)

# --- OLS fit_regularized() ---
grid = reg_results.pivot(index="alpha", columns="L1_wt", values="test_R2").round(4)
grid.index = [f"{a:g}" for a in grid.index]
grid.columns = [f"{w:g}" for w in grid.columns]
grid.index.name = r"$\alpha$"
to_tex(grid, "reg_grid",
       "Regularised OLS (\\texttt{fit\\_regularized}): held-out $R^2$ over the "
       "$(\\alpha,\\ \\mathrm{L1\\_wt})$ grid. Columns are L1\\_wt.", "tab:reg-grid")

reg_best = pd.DataFrame({"value": {
    r"$\alpha$":  f"{best.alpha:g}",
    "L1\\_wt":    f"{best.L1_wt:g}",
    "test $R^2$": f"{best.test_R2:.4f}",
    "test MSE":   f"{best.test_MSE:.1f}",
    "test RMSE":  f"{best.test_RMSE:.1f}",
    "non-zero coefficients": f"{int(best.nonzero)}/{k + 1}",
}})
reg_best.index.name = "statistic"
to_tex(reg_best, "reg_best",
       "Regularised OLS: grid point with the highest held-out $R^2$.", "tab:reg-best")

# --- collinearity ---
vif = pd.DataFrame({"before drops": before.reindex(plain), "after drops": after.reindex(plain)})
vif = vif.map(lambda v: "---" if pd.isna(v) else (f"{v:.3g}" if v >= 1e4 else f"{v:.2f}"))
vif.index.name = "predictor"
to_tex(vif, "vif",
       "Variance inflation factors for the non-dummy predictors, before and after "
       "the collinearity drops. \\texttt{---} marks a predictor removed from the "
       "final matrix.", "tab:vif")

cond = pd.DataFrame({
    "before drops": [f"{before.drop(plain, errors='ignore').max():.3g}",
                     f"{np.linalg.cond(sm.add_constant(X_candidate.astype(float))):.3g}"],
    "after drops":  [f"{after.drop(plain, errors='ignore').max():.2f}",
                     f"{np.linalg.cond(sm.add_constant(X_cleaned.astype(float))):.1f}"],
}, index=["max VIF over dummy levels", "condition number"])
cond.index.name = "diagnostic"
to_tex(cond, "cond",
       "Collinearity of the design matrix as a whole, before and after the drops.",
       "tab:cond")

print(f"\nreport assets written to {REPORT}")
