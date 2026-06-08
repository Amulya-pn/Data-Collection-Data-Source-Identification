import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('iris')

# Basic Information
print(df.head())
print(df.info())
print(df.describe())

# Missing Values
print(df.isnull().sum())

# -------------------------
# Univariate Analysis
# -------------------------

df.hist(figsize=(10, 8))
plt.suptitle("Histograms")
plt.show()

for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f"Box Plot - {col}")
    plt.show()

# -------------------------
# Bivariate Analysis
# -------------------------

sns.scatterplot(
    data=df,
    x='sepal_length',
    y='petal_length',
    hue='species'
)
plt.show()

# -------------------------
# Correlation Heatmap
# -------------------------

plt.figure(figsize=(8,6))
corr = df.select_dtypes(include='number').corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# Pairplot
# -------------------------

sns.pairplot(df, hue='species')
plt.show()