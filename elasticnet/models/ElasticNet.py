## Importing Libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


## ElecticNet Class

# Elastic Net Regression
class ElasticRegression:
    def __init__(self, learning_rate, iterations, l1_penalty, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = np.zeros(self.n)

        error = (self.Y - Y_pred)
        for j in range(self.n):

            l1_grad = self.l1_penalty
            l2_grad = 2 * self.l2_penalty * self.W[j]

            if self.W[j] > 0:
                dW[j] = (- (2 * (self.X[:, j]).dot(error)) + l1_grad +
                         l2_grad) / self.m
            else:
                dW[j] = (- (2 * (self.X[:, j]).dot(error))
                         - l1_grad + l2_grad) / self.m

        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.W) + self.b

## Importing Data

# Importing dataset
df = pd.read_csv("/content/winequality-red.csv", sep=';')  # Make sure to download this dataset first

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

## Normalizing features

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

## Spliting data in Training : Testing (7:3)

# Splitting dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

## Model fitting

# Model training
model = ElasticRegression(iterations=1000,
                          learning_rate=0.01, l1_penalty=0.7, l2_penalty=0.3)

model.fit(X_train, Y_train)


## Prediction

Y_pred = model.predict(X_test)

## Evaluating model

# Calculate MSE
mse = mean_squared_error(Y_test, Y_pred)

# Calculate R-squared
r_squared = r2_score(Y_test, Y_pred)

# Calculate MAE
mae = mean_absolute_error(Y_test, Y_pred)

# Print the results
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

## Scatter Plot

# Visualization on test set (for one feature)
plt.scatter(X_test[:, 0], Y_test, color='blue')
plt.scatter(X_test[:, 0], Y_pred, color='orange')
plt.title('Wine Quality Prediction')
plt.xlabel('Feature')
plt.ylabel('Quality')
plt.show()

## TESTING

# TESTING

penalties = [0.0, 0.1, 1.0, 10.0]
for l1 in penalties:
    for l2 in penalties:
        model = ElasticRegression(iterations=5000, learning_rate=0.001, l1_penalty=l1, l2_penalty=l2)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        print(f"L1: {l1}, L2: {l2}, MSE: {mse}")

