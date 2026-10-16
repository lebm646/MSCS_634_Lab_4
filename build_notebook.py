"""Build and execute Minh Le's reproducible regression lab."""
from pathlib import Path
import textwrap
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

md('''
# Lab 4: Regression Analysis with Regularization Techniques
**Name:** Minh Le  
**Course:** Advanced Big Data and Data Mining (MSCS-634-M50), Fall 2026  
**Dataset:** scikit-learn Diabetes

This lab compares simple, multiple, and polynomial regression with Ridge and Lasso.
The goal is to measure predictive accuracy and examine the tradeoff between model
complexity and generalization. All reported results below are produced by running this notebook.
''')
md('''
## Step 1: Data preparation and exploration
The data contain 442 observations and 10 baseline measurements. The target is a
quantitative measure of disease progression one year after baseline. Features are
age, sex, BMI, average blood pressure (`bp`), and six serum measurements (`s1`–`s6`).
We load unscaled values so scaling can be learned from training data inside each pipeline.
The sex feature is a coded category, not a continuous physiological measurement.
''')
code('''
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

# Surface convergence problems instead of silently accepting unreliable fits.
warnings.filterwarnings('error', category=ConvergenceWarning)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 12})
OUT = Path('results')
OUT.mkdir(exist_ok=True)
data = load_diabetes(as_frame=True, scaled=False)
X, y = data.data.copy(), data.target.copy()
frame = X.assign(target=y)
print(f'Observations: {len(X)}; predictors: {X.shape[1]}')
display(frame.head())
display(frame.describe().T.round(2))
display(pd.DataFrame({'dtype': frame.dtypes.astype(str), 'missing': frame.isna().sum()}))
print('Duplicate full rows:', frame.duplicated().sum())
assert not frame.isna().any().any(), 'Missing values require a training-fitted imputer.'
assert np.isfinite(frame.to_numpy()).all(), 'Nonfinite values require investigation.'
# No rows are removed: the built-in dataset is complete and numerical.
''')
code('''
# Split once; all models use the same held-out observations.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
print(f'Training rows: {len(X_train)}; test rows: {len(X_test)}')

# Examine training distributions to avoid using test patterns for model choices.
train_frame = X_train.assign(target=y_train)
train_frame.hist(bins=20, figsize=(13, 10), color='#3274A1', edgecolor='white')
plt.suptitle('Training-data feature and target distributions', y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'distributions.png', bbox_inches='tight')
plt.show()
corr = train_frame.corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr)), corr.columns)
fig.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Training-data correlations')
plt.tight_layout()
plt.savefig(OUT / 'correlations.png', bbox_inches='tight')
plt.show()
display(corr['target'].drop('target').sort_values(ascending=False).to_frame('Target correlation').round(3))
''')
md('''
### Evaluation design
The 80/20 split uses seed 42. Hyperparameters are chosen by five-fold training-only
cross-validation (CV), minimizing mean fold RMSE. Scaling is fitted separately inside
each CV fold. Test scores are used only for the final comparison, not for tuning.

MAE is the mean absolute error; MSE is the mean squared error; RMSE is its square root.
Lower values are better. MAE and RMSE use target units, while MSE uses squared units.
R² measures improvement over predicting the evaluation set's mean: 1 is perfect,
0 matches that reference, and negative values are possible. A separate training-mean
dummy model provides a practical baseline. These are predictive associations, not causal effects.
''')
code('''
models, rows, predictions = {}, [], {}
def evaluate(name, model, columns=None, cv_rmse=None):
    # Reuse identical rows, even when the simple model uses only BMI.
    columns = list(X.columns) if columns is None else columns
    model.fit(X_train[columns], y_train)
    train_pred = model.predict(X_train[columns])
    pred = model.predict(X_test[columns])
    models[name] = model
    predictions[name] = pred
    row = {'Model': name, 'MAE': mean_absolute_error(y_test, pred),
           'MSE': mean_squared_error(y_test, pred),
           'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
           'R2': r2_score(y_test, pred),
           'Train RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
           'CV RMSE': cv_rmse}
    rows.append(row)
    return pd.DataFrame([row]).set_index('Model').round(3)

def actual_plot(ax, name):
    pred = predictions[name]
    ax.scatter(y_test, pred, alpha=0.7, s=24)
    limits = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(limits, limits, '--', color='black', label='Perfect prediction')
    ax.set(xlabel='Actual progression', ylabel='Predicted progression', title=name)
    ax.legend(fontsize=8)

display(evaluate('Mean baseline', DummyRegressor(strategy='mean')))
''')
md('''
## Step 2: Simple linear regression
BMI is chosen in advance as an interpretable single predictor. This model fits
`progression = intercept + slope × BMI`. It cannot represent information from the
other health measurements, so it may underfit compared with multiple regression.
''')
code('''
simple = Pipeline([('scale', StandardScaler()), ('reg', LinearRegression())])
simple_cv = -cross_val_score(simple, X_train[['bmi']], y_train, cv=cv,
                             scoring='neg_root_mean_squared_error').mean()
display(evaluate('Simple linear (BMI)', simple, ['bmi'], simple_cv))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_test['bmi'], y_test, alpha=0.65, label='Actual test values')
order = np.argsort(X_test['bmi'].to_numpy())  # Sort x-values to draw a continuous line.
ax.plot(X_test['bmi'].iloc[order], predictions['Simple linear (BMI)'][order],
        color='darkorange', linewidth=2, label='Fitted prediction')
ax.set(xlabel='BMI', ylabel='Disease progression', title='Simple regression on held-out data')
ax.legend()
plt.tight_layout()
plt.savefig(OUT / 'simple_regression.png')
plt.show()
''')
md('''
## Step 3: Multiple regression
Multiple regression uses all 10 predictors and fits their additive linear effects.
Predictor correlations can make individual coefficients unstable even when predictions
are useful. Standardization puts coefficients on comparable feature scales.
''')
code('''
multiple = Pipeline([('scale', StandardScaler()), ('reg', LinearRegression())])
multiple_cv = -cross_val_score(multiple, X_train, y_train, cv=cv,
                               scoring='neg_root_mean_squared_error').mean()
display(evaluate('Multiple linear', multiple, cv_rmse=multiple_cv))
fig, ax = plt.subplots(figsize=(6, 5))
actual_plot(ax, 'Multiple linear')
plt.tight_layout()
plt.savefig(OUT / 'multiple_regression.png')
plt.show()
''')
md('''
## Step 4: Polynomial regression and model complexity
Polynomial expansion includes powers and interactions of all 10 predictors. Degrees
1, 2, and 3 produce 10, 65, and 285 columns, respectively (excluding the intercept).
Degree 1 is the multiple linear model. We scale before expansion for numerical stability
and afterward so regularization will treat the expanded terms comparably.
Powers of the binary-coded sex feature are redundant; ordinary least squares handles
rank deficiency, but this is another reason not to interpret every expanded coefficient.
''')
code('''
def polynomial_pipeline(degree, reg):
    return Pipeline([('input_scale', StandardScaler()),
                     ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                     ('scale', StandardScaler()), ('reg', reg)])

degree_rows = []
for degree in [1, 2, 3]:
    model = polynomial_pipeline(degree, LinearRegression())
    fold_rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='neg_root_mean_squared_error')
    model.fit(X_train, y_train)
    degree_rows.append({'Degree': degree,
                       'Features': model.named_steps['poly'].n_output_features_,
                       'Train RMSE': np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
                       'CV RMSE': fold_rmse.mean(), 'CV SD': fold_rmse.std()})
    if degree > 1:
        evaluate(f'Polynomial degree {degree}', model, cv_rmse=fold_rmse.mean())
degree_results = pd.DataFrame(degree_rows).set_index('Degree')
display(degree_results.round(3))
print('Degree preferred by training CV:', degree_results['CV RMSE'].idxmin())
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(degree_results.index, degree_results['Train RMSE'], 'o-', label='Training')
ax.errorbar(degree_results.index, degree_results['CV RMSE'], yerr=degree_results['CV SD'],
            fmt='o-', capsize=4, label='CV mean ± fold SD')
ax.set(xlabel='Polynomial degree', ylabel='RMSE (log scale)', yscale='log',
       title='Increasing complexity: training fit versus validation error', xticks=[1, 2, 3])
ax.legend()
plt.tight_layout()
plt.savefig(OUT / 'polynomial_complexity.png')
plt.show()
''')
md('''
## Step 5: Ridge and Lasso regularization
Ridge adds an L2 penalty (squared coefficients); Lasso adds an L1 penalty (absolute
coefficients). Increasing alpha encourages smaller coefficients and can reduce variance;
excessive alpha causes underfitting. Lasso can set coefficients to zero, whereas Ridge
typically retains them. The intercept is not penalized. scikit-learn normalizes the loss
differently for Ridge and Lasso, so the same numeric alpha does not mean equal strength.

We tune both methods on the original 10 predictors and on the degree-2 expansion.
The extra polynomial fits test whether regularization improves an overly flexible model.
All alpha grids and degree choices are specified before inspecting test performance.
''')
code('''
searches = {}
for degree in [1, 2]:
    for kind, estimator, alphas in [
        ('Ridge', Ridge(), np.logspace(-3, 5, 25)),
        ('Lasso', Lasso(max_iter=100000, tol=1e-4), np.logspace(-2, 2, 25))
    ]:
        name = kind if degree == 1 else f'{kind} polynomial degree 2'
        pipe = (Pipeline([('scale', StandardScaler()), ('reg', estimator)])
                if degree == 1 else polynomial_pipeline(degree, estimator))
        # GridSearchCV refits the best setting on all training observations.
        search = GridSearchCV(pipe, {'reg__alpha': alphas}, cv=cv,
                              scoring='neg_root_mean_squared_error', n_jobs=1,
                              return_train_score=True, error_score='raise')
        search.fit(X_train, y_train)
        searches[name] = search
        display(evaluate(name, search.best_estimator_, cv_rmse=-search.best_score_))
        print(f'{name}: selected alpha = {search.best_params_["reg__alpha"]:.5g}')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, search) in zip(axes.flat, searches.items()):
    values = search.cv_results_
    alpha = np.asarray(values['param_reg__alpha'], dtype=float)
    ax.semilogx(alpha, -values['mean_train_score'], label='Training')
    ax.semilogx(alpha, -values['mean_test_score'], label='Validation')
    ax.axvline(search.best_params_['reg__alpha'], color='black', ls='--', label='Selected alpha')
    ax.set(title=name, xlabel='Alpha (log scale)', ylabel='Mean fold RMSE')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / 'alpha_effects.png')
plt.show()
''')
code('''
# Compare coefficients on the same standardized original-feature basis.
coefficients = pd.DataFrame({name: models[name].named_steps['reg'].coef_
                             for name in ['Multiple linear', 'Ridge', 'Lasso']}, index=X.columns)
display(coefficients.round(3))
coefficients.plot.bar(figsize=(10, 5))
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel('Coefficient per standardized feature unit')
plt.title('Ridge shrinkage and Lasso feature selection')
plt.tight_layout()
plt.savefig(OUT / 'coefficients.png')
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, name in zip(axes.flat, searches):
    actual_plot(ax, name)
plt.tight_layout()
plt.savefig(OUT / 'regularized_predictions.png')
plt.show()
''')
md('''
## Step 6: Model comparison and analysis
The table reports held-out MAE, MSE, RMSE, and R² alongside training and CV RMSE.
The CV column is the mean fold RMSE, not the RMSE pooled across folds. Fold SD in
the degree experiment describes variation across folds, not a confidence interval.
''')
code('''
results = pd.DataFrame(rows).set_index('Model')
display(results.sort_values('RMSE').round(3))
results.to_csv(OUT / 'model_comparison.csv')
degree_results.to_csv(OUT / 'polynomial_degrees.csv')
coefficients.to_csv(OUT / 'coefficients.csv')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
results[['Train RMSE', 'RMSE']].plot.barh(ax=axes[0], logx=True)
axes[0].set(xlabel='RMSE (log scale)', title='Training versus test error')
axes[0].legend(['Training', 'Test'])
# Choose the model family by training CV, not by the test ranking.
recommended = results['CV RMSE'].idxmin()
residuals = y_test - predictions[recommended]
axes[1].scatter(predictions[recommended], residuals, alpha=0.7)
axes[1].axhline(0, color='black', ls='--')
axes[1].set(xlabel='Predicted progression', ylabel='Actual − predicted',
            title=f'Residuals: {recommended}')
plt.tight_layout()
plt.savefig(OUT / 'model_comparison.png', bbox_inches='tight')
plt.show()

# Save compact numerical evidence for the accompanying written summary.
summary = {'recommended': recommended, 'test_best': results['RMSE'].idxmin(),
           'alphas': {n: float(s.best_params_['reg__alpha']) for n, s in searches.items()},
           'lasso_zero_count': int(np.sum(np.isclose(coefficients['Lasso'], 0))),
           'correlations': corr['target'].drop('target').to_dict()}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
''')
md('ANALYSIS_PLACEHOLDER')
md('''
## Limitations and reproducibility
There are only 442 observations, and results depend on the chosen split. CV guides
tuning, but its winning score can be optimistic after searching many settings. A single
test set is not enough to establish statistical superiority. Correlated predictors and
redundant polynomial columns complicate coefficient interpretation. Predictions do not
establish causation or clinical usefulness. No missing-value imputation or outlier removal
was needed; extreme observations were retained because there was no evidence of data errors.

Run all cells in order with the packages in `requirements.txt`. The dataset ships with
scikit-learn, so no external dataset download or private data is required. Figures and
tables are saved in `results/`, and all notebook outputs are embedded for GitHub viewing.

## References
- [scikit-learn Diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [scikit-learn linear models guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {
    'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
path = ROOT / 'MSCS_634_Lab_4.ipynb'
nbf.write(nb, path)
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
nbf.write(nb, path)
print(f'Executed {path}')
from finalize_lab import finalize
finalize()
