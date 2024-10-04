import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
data = pd.read_csv('Position_Salaries.csv')

# Independent and dependent variables for regression models
X = data[['Level']].values  # Independent variable (Level)
y = data['Salary'].values   # Dependent variable (Salary)

# 1. Simple Linear Regression
simple_lin_reg = LinearRegression()
simple_lin_reg.fit(X, y)

# Predictions for Simple Linear Regression
y_pred_simple = simple_lin_reg.predict(X)

# Plotting Simple Linear Regression results
plt.scatter(X, y, color='red')  # Actual data points
plt.plot(X, y_pred_simple, color='blue')  # Regression line
plt.title('Simple Linear Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# 2. Multiple Linear Regression
# Creating dummy variables for the 'Position' column
data_with_dummies = pd.get_dummies(data, columns=['Position'], drop_first=True)

# Independent variables (Level + Dummy Variables for Position)
X_multiple = data_with_dummies.drop('Salary', axis=1).values

# Initialize and fit the Multiple Linear Regression model
multiple_lin_reg = LinearRegression()
multiple_lin_reg.fit(X_multiple, y)

# Predictions for Multiple Linear Regression
y_pred_multiple = multiple_lin_reg.predict(X_multiple)

# Output model coefficients and intercept
print("Multiple Linear Regression Coefficients:", multiple_lin_reg.coef_)
print("Multiple Linear Regression Intercept:", multiple_lin_reg.intercept_)

# 3. Polynomial Regression (degree 4)
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)

# Initialize and fit the Polynomial Regression model
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Predictions for Polynomial Regression
y_pred_poly = poly_reg.predict(X_poly)

# Plotting Polynomial Regression results
plt.scatter(X, y, color='red')  # Actual data points
plt.plot(X, y_pred_poly, color='green')  # Polynomial regression curve
plt.title('Polynomial Regression: Level vs Salary (Degree 4)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
