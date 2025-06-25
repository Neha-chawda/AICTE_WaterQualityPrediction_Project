import pandas as pd
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("D:/Git Demo/Week_1_AICTE-Project/PB_All_2000_2021.csv", sep=';')

# Dataset structure
print("\nInitial Data Info:")
print(df.info())

# Dimensions and summary statistics
print("\nShape of dataset:", df.shape)
print("\nData Summary:")
print(df.describe().T)

# Null values per column
print("\nMissing Values:")
print(df.isnull().sum())

# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
print("\nData Info after date conversion:")
print(df.info())

# Sort by id and date
df = df.sort_values(by=['id', 'date'])

# Extract year and month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Preview updated data
print("\nPreview of updated DataFrame:")
# print(df[['id', 'date', 'year', 'month']].head())
print(df.head())
print(df.columns)
# List of pollutant variables
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
print("\nPollutant columns:", pollutants)

# drop the missing values - dropna()
df = df.dropna(subset=pollutants)
print(df.head())

print(df.isnull().sum())

# Feature and target selection - Feature - independent variable and Target dependent variable
X = df[['id', 'year']]
y = df[pollutants]

# Encoding - onehotencoder - 22 stations - 1 - 1
X_encoded = pd.get_dummies(X, columns=['id'], drop_first=True)

# Train, Test and Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train the model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
print(model.fit(X_train, y_train))

# Evaluate model
y_pred = model.predict(X_test)

print("Model Performance on the Test Data:")
for i, pollutant in enumerate(pollutants):
    print(f'{pollutant}:')
    print('   MSE:', mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print('   R2:', r2_score(y_test.iloc[:, i], y_pred[:, i]))
    print()

station_id = '22'
year_input = 2024
input_data = pd.DataFrame({'year': [year_input], 'id': [station_id]})
input_encoded = pd.get_dummies(input_data, columns=['id'])
# Align with training feature columns
missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]  # reorder columns
# Predict pollutants
predicted_pollutants = model.predict(input_encoded)[0]
print(f"\nPredicted pollutant levels for station '{station_id}' in {year_input}:")
for p, val in zip(pollutants, predicted_pollutants):
    print(f"  {p}: {val:.2f}")    


import joblib

joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")
print('Model and cols structure are saved!')