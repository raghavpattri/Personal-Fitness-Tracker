# Personal-Fitness-Tracker


## Overview
The Personal Fitness Tracker is a Streamlit web application that predicts the number of calories burned based on user input parameters such as age, BMI, exercise duration, heart rate, body temperature, and gender.

## Features
- **User Input Parameters:**
  - Age
  - BMI
  - Exercise Duration (minutes)
  - Heart Rate
  - Body Temperature (°C)
  - Gender (Male/Female)
- **Data Processing:**
  - Loads and merges `calories.csv` and `exercise.csv` datasets
  - Computes BMI from height and weight
  - Splits data into training and testing sets
  - Encodes categorical variables
- **Machine Learning Model:**
  - Uses a Random Forest Regressor to predict calorie expenditure
  - Trained with 80% of the dataset
  - Configured with `n_estimators=1000`, `max_features=3`, and `max_depth=6`
- **Prediction Output:**
  - Displays predicted calories burned
  - Shows five similar data samples
  - Provides general statistics comparing the user to dataset distributions

## Installation
1. Clone the repository or download the project files.
2. Install the required dependencies:
   ```sh
   pip install streamlit pandas numpy scikit-learn
   ```
3. Place `calories.csv` and `exercise.csv` files in the project directory.

## Running the Application
Run the following command in the terminal:
```sh
streamlit run app.py
```

## Dataset Requirements
Ensure `calories.csv` and `exercise.csv` contain relevant fitness data with the following columns:
- `User_ID`
- `Gender`
- `Age`
- `Height`
- `Weight`
- `Duration`
- `Heart_Rate`
- `Body_Temp`
- `Calories`

## Notes
- The app assumes missing values are handled before dataset usage.
- The model's accuracy depends on dataset quality and distribution.

## Future Enhancements
- Add more input features for better prediction accuracy
- Incorporate additional machine learning models for comparison
- Enhance UI with data visualizations

