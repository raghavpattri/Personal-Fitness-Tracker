# Required Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Ensure plots render inline
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Load Data
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

# Merge datasets
exercise_df = exercise.merge(calories, on='User_ID')
exercise_df.drop(columns=['User_ID'], inplace=True)

# Add BMI Column
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Splitting Data into Train & Test
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Select relevant features
features = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
exercise_train_data = exercise_train_data[features]
exercise_test_data = exercise_test_data[features]

# One-hot encoding for gender
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate Features & Target Variable
x_train = exercise_train_data.drop(columns=["Calories"])
y_train = exercise_train_data["Calories"]

x_test = exercise_test_data.drop(columns=["Calories"])
y_test = exercise_test_data["Calories"]

# Train Model - Random Forest Regressor
random_reg = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=6, random_state=42)
random_reg.fit(x_train, y_train)

# Predictions
random_reg_predictions = random_reg.predict(x_test)

# Evaluation Metrics
print("Random Forest Regressor Performance:")
print("MAE:", round(metrics.mean_absolute_error(y_test, random_reg_predictions), 2))
print("MSE:", round(metrics.mean_squared_error(y_test, random_reg_predictions), 2))
print("RMSE:", round(np.sqrt(metrics.mean_squared_error(y_test, random_reg_predictions)), 2))

# Visualizing Model Performance
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=random_reg_predictions, alpha=0.7)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs. Predicted Calories")
plt.show()

# Learning Curve Function
def plot_learning_curve(model, x_train, y_train, x_test, y_test):
    train_errors, val_errors = [], []
    
    for m in range(10, len(x_train), 50):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_test)
        
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_test, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation")
    plt.xlabel("Training set size")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

# Plot Learning Curve for Linear Regression
linreg = LinearRegression()
plot_learning_curve(linreg, x_train, y_train, x_test, y_test)

# Train Linear Regression Model
linreg.fit(x_train, y_train)
linreg_predictions = linreg.predict(x_test)

# Linear Regression Evaluation
print("\nLinear Regression Performance:")
print("MAE:", round(metrics.mean_absolute_error(y_test, linreg_predictions), 2))
print("MSE:", round(metrics.mean_squared_error(y_test, linreg_predictions), 2))
print("RMSE:", round(np.sqrt(metrics.mean_squared_error(y_test, linreg_predictions)), 2))

# Visualizing Linear Regression Predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=linreg_predictions, alpha=0.7, color="red")
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned (Linear Regression)")
plt.title("Actual vs. Predicted Calories (Linear Regression)")
plt.show()

# Sample Prediction
sample_input = np.array([[1, 24, 25, 28, 100, 40]]).reshape(1, -1)
sample_prediction = random_reg.predict(sample_input)
print("\nPredicted Calories Burned:", round(sample_prediction[0], 2), "kcal")
