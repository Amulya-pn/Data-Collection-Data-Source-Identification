import pandas as pd
import kagglehub
import os

# Download dataset
df = pd.read_csv("student_dataset_10000_rows.csv")

# Show first rows
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Column types
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("\nNumerical Columns:")
print(list(numerical_cols))

print("\nCategorical Columns:")
print(list(categorical_cols))

# Unique values
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())

# Data types
print("\nData Types:")
print(df.dtypes)
