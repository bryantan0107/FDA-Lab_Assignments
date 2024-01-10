# Solution for task 1 (Lasso) of lab assignment - FDA SS23 by Tan

# if necessary, write text as answer in comments or use a Jupyter notebook

# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.metrics import mean_squared_error

# load data (change path if necessary)
df = pd.read_csv("lasso_data.csv")


# Task 1.1 Is it possible to solve the lasso optimisation problem analytically? Explain. (3 points)
# In general, no. Because minimization problems involve differentiation and the lasso involves an absolute 
# value. The L1-norm penalty in the optimization objective function is non-differentiable at zero, which 
# makes it difficult to solve analytically.

# https://allmodelsarewrong.github.io/lasso.html

    
# Task 1.2 Split the data into a train and a test set with appropriate test size. (2 points)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# https://towardsdatascience.com/how-to-split-a-dataset-into-training-and-testing-sets-b146b1649830

    
# Task 1.3 Fit a linear regression model for Y using all remaining variables on the training data. (5 points)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# Task 1.4 Make a model prediction on unseen data and assess model performance using a suitable metric. (5 points)
y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression Mean Squared Error:", mse)

# Task 1.5 Perform lasso regression using the same data as in task 1.3 (6 points)
lasso = Lasso(alpha=0.1) # or lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("Lasso Regression Mean Squared Error:", mse_lasso)


# Task 1.6 Compare model performance to the original linear model by using the same metric and test set as in 1.4.
# What do you observe? (2 points)
#In this case, the Linear model has better performance than the Lasso model because it has lower Mean Squared Error.


# Task 1.7 Print out the model coefficients for both, the linear model and the lasso model. (2 points)
print("Linear Regression Coefficients: ", lin_reg.coef_)
print("Lasso Regression Coefficients: ", lasso.coef_)

# https://python-forum.io/thread-11252.html


# Task 1.8 What do you observe comparing the estimated model coefficients? Was this result expected? (5 points)
# Hint: Look at the data generating process and lasso explanation to answer this question
# The Lasso model has fewer non-zero coefficients than the Linear model. The result was expected because 
# Lasso regression can sparse the coefficients and set the coefficients of the variables that have less influence 
# on Y to zero for the purpose of feature selection.


