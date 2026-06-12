# ==========================================
# Task 4: Feature Encoding & Scaling
# Adult Census Income Dataset
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# Load Dataset
# ==========================================

column_names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
]

df = pd.read_csv(
    "adult_data.csv",
    header=None,
    names=column_names,
    skipinitialspace=True
)

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
print(df.info())

print("\nDataset Shape:", df.shape)

# ==========================================
# Missing Values Check
# ==========================================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# ==========================================
# Identify Categorical Columns
# ==========================================

categorical_cols = df.select_dtypes(include=['object']).columns

print("\n" + "=" * 50)
print("CATEGORICAL COLUMNS")
print("=" * 50)
print(list(categorical_cols))

# ==========================================
# Label Encoding Target Variable
# ==========================================

label_encoder = LabelEncoder()

df['income_label'] = label_encoder.fit_transform(df['income'])

print("\n" + "=" * 50)
print("LABEL ENCODING")
print("=" * 50)
print(df[['income', 'income_label']].head())

# ==========================================
# Remove Target from One-Hot Encoding
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

print("\n" + "=" * 50)
print("AFTER ONE-HOT ENCODING")
print("=" * 50)
print("Shape:", df_encoded.shape)

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
# Visualization
# ==========================================

feature = 'age'

plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 3, 1)
plt.hist(df_encoded[feature], bins=30)
plt.title("Original Data")
plt.xlabel(feature)
plt.ylabel("Frequency")

# Standardized
plt.subplot(1, 3, 2)
plt.hist(df_standardized[feature], bins=30)
plt.title("Standardized Data")
plt.xlabel(feature)

# Normalized
plt.subplot(1, 3, 3)
plt.hist(df_normalized[feature], bins=30)
plt.title("Normalized Data")
plt.xlabel(feature)

plt.tight_layout()
plt.show()

# ==========================================
# Statistics Comparison
# ==========================================

print("\n" + "=" * 50)
print("ORIGINAL DATA STATISTICS")
print("=" * 50)
print(df_encoded[continuous_features].describe())

print("\n" + "=" * 50)
print("STANDARDIZED DATA STATISTICS")
print("=" * 50)
print(df_standardized[continuous_features].describe())

print("\n" + "=" * 50)
print("NORMALIZED DATA STATISTICS")
print("=" * 50)
print(df_normalized[continuous_features].describe())

# ==========================================
# Correlation Matrix
# ==========================================

corr_matrix = df_standardized[continuous_features].corr()

plt.figure(figsize=(8, 6))
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

print("\n" + "=" * 50)
print("FEATURE ENCODING & SCALING COMPLETED")
print("=" * 50)

print("""
Techniques Applied:
1. Label Encoding (income)
2. One-Hot Encoding (categorical features)
3. Standardization
4. Min-Max Normalization
5. Distribution Comparison
6. Correlation Analysis
""")
