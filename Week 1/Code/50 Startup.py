# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read the CSV file
file_path = '50_Startups.csv'  # Adjust the file path accordingly
data_startups = pd.read_csv(file_path)

# SIMPLE LINEAR REGRESSION
# Use 'R&D Spend' to predict 'Profit'
X_simple = data_startups[['R&D Spend']]
y_simple = data_startups['Profit']

# Split data into training (80%) and testing (20%) sets
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Create Simple Linear Regression model
simple_lr_model_startups = LinearRegression()

# Train the model
simple_lr_model_startups.fit(X_train_simple, y_train_simple)

# Make predictions using the test data
y_pred_simple_startups = simple_lr_model_startups.predict(X_test_simple)

# Get the coefficients and intercept
coef_simple_startups = simple_lr_model_startups.coef_
intercept_simple_startups = simple_lr_model_startups.intercept_

# Print results for Simple Linear Regression
print("Simple Linear Regression Coefficient:", coef_simple_startups)
print("Simple Linear Regression Intercept:", intercept_simple_startups)
print("First 5 Predictions (Simple LR):", y_pred_simple_startups[:5])

# MULTIPLE LINEAR REGRESSION
# Use 'R&D Spend', 'Administration', and 'Marketing Spend' to predict 'Profit'
X_multi_startups = data_startups[['R&D Spend', 'Administration', 'Marketing Spend']]

# Split data into training and testing sets (80% training, 20% testing)
X_train_multi_startups, X_test_multi_startups, y_train_multi_startups, y_test_multi_startups = train_test_split(X_multi_startups, y_simple, test_size=0.2, random_state=42)

# Create Multiple Linear Regression model
multi_lr_model_startups = LinearRegression()

# Train the model
multi_lr_model_startups.fit(X_train_multi_startups, y_train_multi_startups)

# Make predictions using the test data
y_pred_multi_startups = multi_lr_model_startups.predict(X_test_multi_startups)

# Get the coefficients and intercept for Multiple Linear Regression
coef_multi_startups = multi_lr_model_startups.coef_
intercept_multi_startups = multi_lr_model_startups.intercept_

# Print results for Multiple Linear Regression
print("Multiple Linear Regression Coefficients:", coef_multi_startups)
print("Multiple Linear Regression Intercept:", intercept_multi_startups)
print("First 5 Predictions (Multiple LR):", y_pred_multi_startups[:5])

# POLYNOMIAL REGRESSION
# Use 'R&D Spend' as a polynomial feature to predict 'Profit'
poly_startups = PolynomialFeatures(degree=2)

# Transform the 'R&D Spend' data into polynomial form (degree=2)
X_poly_startups = poly_startups.fit_transform(X_simple)

# Split data into training and testing sets (80% training, 20% testing)
X_train_poly_startups, X_test_poly_startups, y_train_poly_startups, y_test_poly_startups = train_test_split(X_poly_startups, y_simple, test_size=0.2, random_state=42)

# Create Polynomial Regression model
poly_model_startups = LinearRegression()

# Train the Polynomial model
poly_model_startups.fit(X_train_poly_startups, y_train_poly_startups)

# Make predictions using the test data
y_pred_poly_startups = poly_model_startups.predict(X_test_poly_startups)

# Get the coefficients and intercept for Polynomial Regression
coef_poly_startups = poly_model_startups.coef_
intercept_poly_startups = poly_model_startups.intercept_

# Print results for Polynomial Regression
print("Polynomial Regression Coefficients:", coef_poly_startups)
print("Polynomial Regression Intercept:", intercept_poly_startups)
print("First 5 Predictions (Polynomial LR):", y_pred_poly_startups[:5])
