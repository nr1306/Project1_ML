# Project 1 
- A20602211 - Kannekanti Nikhil
- A20586642 - Nesh Rochwani

Dataset Utilized: https://drive.google.com/file/d/1ow1yR4nSKOjzjIIq2rcNDTQBdVssqxYl/view?usp=sharing

*Q-1 What does the model you have implemented do and when should it be used?

Elastic Net Regression model was implemented that utilizes a combination of L1 (Lasso) and L2 (Ridge) regularization in linear regression.

This model should be used when:
- Selecting relevant features and excluding irrelevant ones automatically.
- Handling feature datasets with high correlations.
- Dealing with high-dimensional datasets, it reduces model complexity and avoiding overfitting
- Achieving a sparse solution where some coefficients become zero while still benefiting from the regularization advantages of Ridge regression.

*Q-2 How did you test your model to determine if it is working reasonably correctly?

- I used the Wine Quality dataset to test the model. I verified its performance by dividing the dataset was divided into training (70%) and testing (30%) subsets.
- Using the test data, the model's performance was assessed using:
  - The average squared difference between the actual and anticipated values is measured using the mean squared error, or MSE.
  - R-squared (R²): To calculate the percentage of the dependent variable's variance that can be predicted based on the characteristics.
  - The average magnitude of the errors in a series of predictions is measured using the mean absolute error (MAE), which does not take direction into account.
- I experimented with different combinations of L1 and L2 penalties understand how different regularization levels affected the model's performance.

*Q-3 What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

- learning_rate: Regulates the size of each gradient descent step when modifying weights.
- The number of iterations determines how many times gradient descent will be used in the training process.
- L1_penalty: Controls the L1 regularization's intensity (Lasso).
- l2_penalty: Controls the L2 regularization's intensity (Ridge).

*Q-4 Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

- For some datasets, the predefined number of iterations might not be optimal. Early stopping with convergence criteria could reduce training time, avoid overfitting, and improve performance.

- Since it is a linear model, it might have trouble interpreting the data's extremely non-linear relationships. This is crucial to the model, however it can be fixed by using kernel techniques or adding polynomial features.

- The current implementation makes use of gradient descent. Adam and RMSprop are examples of more complex optimization methods that may improve convergence and performance.
