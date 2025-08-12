# Project 1 – Linear Regression with ElasticNet Regularization

**Description**  
This project implements a Linear Regression model augmented with ElasticNet regularization—a weighted combination of L1 (Lasso) and L2 (Ridge) penalties. It is especially useful for regression tasks requiring both shrinkage and variable selection, such as datasets with correlated predictors or high-dimensional features.

---

## 1. What does the model do and when should it be used?

- Performs linear regression with controlled regularization via ElasticNet.
- Ideal when you want to mitigate overfitting, control model complexity, and promote feature sparsity—especially helpful for data with multicollinearity or many features.

---

## 2. How was the model tested for correctness?

- A synthetic regression dataset is generated using `generate_regression_data.py`.
- The model is trained and evaluated using common regression metrics (e.g., MSE or RMSE).
- Optionally compared against baseline implementations (e.g., scikit-learn’s ElasticNet) for validation.

---

## 3. What parameters are exposed to users for tuning?

The following hyperparameters are available:

- `alpha`: (float) Regularization strength (0 = no regularization, higher = stronger regularization).
- `l1_ratio`: (float) Weighting between L1 and L2 terms (0 = pure Ridge, 1 = pure Lasso).

**Usage Example:**
```python
from your_module import ElasticNetRegression
from generate_regression_data import generate_data

X_train, X_test, y_train, y_test = generate_data(n_samples=300, noise=0.1)

model = ElasticNetRegression(alpha=0.5, l1_ratio=0.3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 4. Are there specific inputs the model struggles with?

- **High-dimensional, sparse datasets** may result in slow convergence without efficient solvers.  
- **Highly skewed or unscaled features** can skew results; feature normalization is recommended.  
- **Categorical or non-numeric features** are not supported natively and must be preprocessed (e.g., via one-hot encoding).  

Given more time, adding solver optimizations, feature scaling utilities, and categorical data support could greatly extend usability.

---

## 5. Repository Contents

- `generate_regression_data.py`: Script to generate and split synthetic regression data.  
- *(Presumed)* Model implementation file (e.g., `elasticnet.py`): Contains the `ElasticNetRegression` class with `fit` and `predict` methods.  
- `requirements.txt`: Lists dependencies needed to run the project.

