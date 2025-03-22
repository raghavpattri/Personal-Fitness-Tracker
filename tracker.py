import time
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# Title of the app
st.write("## Personal Fitness Tracker")
st.write("In this WebApp you can predict the calories burned based on your input parameters.")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    Age = st.sidebar.slider('Age', 10, 100, 30)
    bmi = st.sidebar.slider('BMI', 15, 40, 20)
    duration = st.sidebar.slider('Duration (min)', 0, 35, 15)
    heart_rate = st.sidebar.slider('Heart Rate', 60, 130, 80)
    body_temp = st.sidebar.slider('Body Temperature (°C)', 36, 42, 38)
    gender_button = st.sidebar.radio('Gender', ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0

    data_model = {
        'Age': Age,
        'BMI': bmi,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp,
        'Gender_male': gender
    }
    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("User Input Parameters")
st.write(df)

# Load datasets
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

# Merge and preprocess data
exercise_df = exercise.merge(calories, on='User_ID')
exercise_df.drop(columns=['User_ID'], inplace=True)

# Add BMI column
for data in [exercise_df]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare train-test data
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and target variable
x_train = exercise_train_data.drop(columns=["Calories"])
y_train = exercise_train_data["Calories"]

x_test = exercise_test_data.drop(columns=["Calories"])
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(x_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=x_train.columns, fill_value=0)

# Make predictions
prediction = random_reg.predict(df)

st.write("---")
st.header("Predicted Calories Burned")
st.write(f"{round(prediction[0], 2)} Kilocalories")

# Find similar results
calories_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calories_range[0]) & (exercise_df["Calories"] <= calories_range[1])]
st.write("---")
st.header("Similar Results")
st.write(similar_data.sample(5))

# General information
st.write("---")
st.header("General Information")

# Ensure no errors due to missing columns
st.write("df columns:", df.columns)

if not df.empty:
    boolean_age = (exercise_df["Age"] < df["Age"].iloc[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].iloc[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].iloc[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].iloc[0]).tolist()

    st.write("You are older than", round((sum(boolean_age) / len(boolean_age)) * 100, 2), "% of the people in the dataset")
    st.write("Your exercise duration is higher than", round((sum(boolean_duration) / len(boolean_duration)) * 100, 2), "% of the people in the dataset")
    st.write("Your body temperature is higher than", round((sum(boolean_body_temp) / len(boolean_body_temp)) * 100, 2), "% of the people in the dataset")
    st.write("Your heart rate is higher than", round((sum(boolean_heart_rate) / len(boolean_heart_rate)) * 100, 2), "% of the people in the dataset")
else:
    st.write("No data available for comparison.")