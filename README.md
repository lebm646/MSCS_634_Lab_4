# MSCS 634 — Lab 4: Regression Analysis with Regularization Techniques

**Student:** Minh Le  
**Course:** Advanced Big Data and Data Mining (MSCS-634-M50), Fall 2026

## Purpose
Compare simple linear, multiple linear, polynomial, Ridge, and Lasso regression on
scikit-learn's Diabetes dataset. The executed notebook includes data exploration,
inline code comments, training and test comparisons, MAE, MSE, RMSE, R², and figures.

## Files
- `MSCS_634_Lab_4.ipynb`: complete notebook with embedded outputs and written analysis.
- `results/`: exported figures, model metrics, coefficients, and tuning summary.
- `requirements.txt`: package versions used with Python 3.13.2.
- `build_notebook.py` and `finalize_lab.py`: reproducible notebook generation and validation.

## Method
The dataset has 442 observations and 10 predictors. An 80/20 split with seed 42 gives
353 training rows and 89 test rows. BMI is the simple regression predictor. All other
models use all predictors, with polynomial powers/interactions where specified.
Five-fold shuffled CV (seed 42) chooses alpha from fixed logarithmic grids on training
data only. All scaling occurs within pipelines. No missing values or duplicate full
rows were found. Test data are reserved for final evaluation.

## Held-out results
| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Mean baseline | 64.01 | 5361.53 | 73.22 | -0.012 |
| Simple linear (BMI) | 52.26 | 4061.83 | 63.73 | 0.233 |
| Multiple linear | 42.79 | 2900.19 | 53.85 | 0.453 |
| Polynomial degree 2 | 43.58 | 3096.03 | 55.64 | 0.416 |
| Polynomial degree 3 | 161.21 | 77860.01 | 279.03 | -13.696 |
| Ridge | 42.81 | 2892.01 | 53.78 | 0.454 |
| Lasso | 42.82 | 2882.15 | 53.69 | 0.456 |
| Ridge polynomial degree 2 | 41.47 | 2649.61 | 51.47 | 0.500 |
| Lasso polynomial degree 2 | 41.27 | 2569.71 | 50.69 | 0.515 |

## Key insights and decisions
1. **Simple versus multiple regression:** BMI alone gives test RMSE **63.73** and R²
   **0.233**, improving on the training-mean baseline (RMSE **73.22**). Using all 10
   predictors reduces RMSE to **53.85**, with R² **0.453**. The single-feature model
   misses useful information; its similar training and validation errors suggest limited
   capacity rather than a large overfitting problem.
2. **Polynomial complexity:** Degree 2 reduces training RMSE from **53.56** to **48.92**,
   but raises CV RMSE from **55.39** to **62.50** and test RMSE to **55.64**. Degree 3
   drives training RMSE down to **27.38**, while CV RMSE reaches **1394.88** and test
   RMSE **279.03** (R² **−13.696**). This is severe overfitting and instability in a
   large, correlated polynomial design. Additional flexibility does not guarantee better
   predictions. Among the unregularized degrees, CV prefers degree 1.
3. **Regularization:** On the original predictors, Ridge (alpha **1**) and Lasso
   (alpha **0.14678**) give test RMSE **53.78** and **53.69**. Their gains over ordinary
   multiple regression are small. Lasso sets **1 of 10** coefficients to zero. On the
   degree-2 expansion, Ridge (alpha **100**) reduces test RMSE to **51.47**, and Lasso
   (alpha **3.16228**) reduces it to **50.69**. Both improve on unregularized degree 2;
   Lasso polynomial also has the lowest test MAE (**41.27**) and MSE (**2569.71**).
   The alpha curves illustrate the balance between insufficient shrinkage and excessive
   shrinkage; larger alpha is not automatically better.
4. **Model choice:** Original-feature Lasso has the lowest training CV RMSE (**55.35**),
   narrowly ahead of Ridge (**55.37**) and multiple regression (**55.39**), so it is the
   CV-based recommendation. Polynomial Lasso has the best observed test result
   (R² **0.515**), but its CV RMSE is **55.96**. The held-out ranking is descriptive,
   not a reason to retune on the test set. Small differences among linear models do not
   establish meaningful superiority without further validation.
5. **Dataset insights:** In training data, BMI and `s5` have positive target correlations
   of approximately **0.605** and **0.552**, while `s3` has correlation **−0.384**.
   The correlation heatmap also shows dependence among serum predictors. Combining
   measurements improves prediction, yet substantial variation remains unexplained.
   A weak marginal correlation does not prove that a feature is useless conditionally.

### Challenges and decisions
The main challenges were avoiding preprocessing leakage, comparing models fairly, and
controlling the growth of polynomial terms. A single reproducible split, pipeline-based
scaling, and training-only CV address these issues. Degrees were limited to 1–3 because
degree 3 already creates 285 columns from only 353 training observations. The original
raw dataset values were used so scaling is estimated only on training folds. Lasso's
iteration limit was increased, and convergence warnings were treated as errors during
execution. No missing values or duplicate full rows were found, so no imputation or
row deletion was applied.

## Reproduce
Create and activate a Python environment, then run:

```bash
python -m pip install -r requirements.txt
python build_notebook.py
```

Alternatively, open the notebook in Jupyter or VS Code, select the environment's Python
kernel, and run all cells. Jupyter's user interface is optional and can be installed with
`python -m pip install jupyterlab`; the build script does not require it. The dataset is
bundled with scikit-learn. Re-running writes figures and CSVs to `results/`.

## Limitations
This is a small educational dataset and one holdout split. The best CV score can be
optimistic after tuning; small differences are not proof of superiority. Results describe
associations, not causal or clinically validated effects.

## References
- [Diabetes dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [Linear models guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
