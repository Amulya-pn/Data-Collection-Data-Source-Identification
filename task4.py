# ==========================================
# Task 4: Feature Encoding & Scaling
# Adult Census Income Dataset
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("adult.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ==========================================
# Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Identify Categorical Columns
# ==========================================

categorical_cols = df.select_dtypes(include=['object']).columns

print("\nCategorical Columns:")
print(categorical_cols)

# ==========================================
# Label Encoding (Target Variable)
# ==========================================

if 'income' in df.columns:

    le = LabelEncoder()

    df['income_label'] = le.fit_transform(df['income'])

    print("\nIncome Label Encoding:")
    print(df[['income', 'income_label']].head())

# ==========================================
# Remove Target Variable from One-Hot Encoding
# ==========================================

categorical_cols = [col for col in categorical_cols if col != 'income']

# ==========================================
# One-Hot Encoding
# ==========================================

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("\nShape After One-Hot Encoding:")
print(df_encoded.shape)

# ==========================================
# Continuous Numerical Features
# ==========================================

continuous_features = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]

# Keep only features that exist
continuous_features = [
    col for col in continuous_features
    if col in df_encoded.columns
]

print("\nContinuous Features:")
print(continuous_features)

# ==========================================
# Standardization
# ==========================================

standard_scaler = StandardScaler()

df_standardized = df_encoded.copy()

df_standardized[continuous_features] = standard_scaler.fit_transform(
    df_standardized[continuous_features]
)

print("\nStandardization Completed Successfully")

# ==========================================
# Normalization
# ==========================================

minmax_scaler = MinMaxScaler()

df_normalized = df_encoded.copy()

df_normalized[continuous_features] = minmax_scaler.fit_transform(
    df_normalized[continuous_features]
)

print("Normalization Completed Successfully")

# ==========================================
# Distribution Comparison
# ==========================================

feature = 'age'

plt.figure(figsize=(15,5))

# Original
plt.subplot(1,3,1)
plt.hist(df_encoded[feature], bins=30)
plt.title("Original Data")
plt.xlabel(feature)
plt.ylabel("Frequency")

# Standardized
plt.subplot(1,3,2)
plt.hist(df_standardized[feature], bins=30)
plt.title("Standardized Data")
plt.xlabel(feature)

# Normalized
plt.subplot(1,3,3)
plt.hist(df_normalized[feature], bins=30)
plt.title("Normalized Data")
plt.xlabel(feature)

plt.tight_layout()
plt.show()

# ==========================================
# Summary Statistics
# ==========================================

print("\n==============================")
print("ORIGINAL DATA STATISTICS")
print("==============================")
print(df_encoded[continuous_features].describe())

print("\n==============================")
print("STANDARDIZED DATA STATISTICS")
print("==============================")
print(df_standardized[continuous_features].describe())

print("\n==============================")
print("NORMALIZED DATA STATISTICS")
print("==============================")
print(df_normalized[continuous_features].describe())

# ==========================================
# Correlation Matrix (Bonus)
# ==========================================

plt.figure(figsize=(10,8))

corr_matrix = df_standardized[continuous_features].corr()

plt.imshow(corr_matrix, cmap='coolwarm')
plt.colorbar()

plt.xticks(
    range(len(continuous_features)),
    continuous_features,
    rotation=45
)

plt.yticks(
    range(len(continuous_features)),
    continuous_features
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# Final Summary
# ==========================================

print("\n===================================")
print("FEATURE ENCODING & SCALING COMPLETED")
print("===================================")

print("\nTechniques Applied:")
print("1. Label Encoding (income)")
print("2. One-Hot Encoding (categorical features)")
print("3. Standardization (continuous features)")
print("4. Min-Max Normalization")
print("5. Distribution Comparison")
print("6. Correlation Analysis")