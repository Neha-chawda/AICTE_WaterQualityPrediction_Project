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
print(df[['id', 'date', 'year', 'month']].head())

# List of pollutant variables
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
print("\nPollutant columns:", pollutants)
