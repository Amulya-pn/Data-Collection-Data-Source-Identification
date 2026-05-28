import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.impute import SimpleImputer

# =========================
# SETTINGS
# =========================

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("titanic dataset.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print("Rows and Columns:", df.shape)

# =========================
# CHECK DUPLICATES
# =========================

print("\n" + "=" * 50)
print("DUPLICATE VALUES")
print("=" * 50)

duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed successfully.")

# =========================
# CHECK MISSING VALUES
# =========================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("MISSING VALUE PERCENTAGE")
print("=" * 50)

missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent)

# =========================
# SUMMARY STATISTICS
# =========================

print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)
print(df.describe())

# =========================
# VISUALIZE MISSING DATA
# =========================

# Remove unnecessary columns for visualization
drop_cols = ['PassengerId', 'Name', 'Ticket']

existing_cols = [col for col in drop_cols if col in df.columns]

df_visual = df.drop(columns=existing_cols)

# Missing value counts
missing_counts = df_visual.isnull().sum().sort_values(ascending=False)

# Create plots
fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6),
    gridspec_kw={'width_ratios': [1, 2]}
)

# Barplot
sns.barplot(
    x=missing_counts.values,
    y=missing_counts.index,
    ax=axes[0],
    palette='viridis'
)

axes[0].set_title('Missing Values Count')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Columns')

# Heatmap
sns.heatmap(
    df_visual.isnull(),
    yticklabels=False,
    cbar=False,
    cmap='viridis',
    ax=axes[1]
)

axes[1].set_title('Missing Values Heatmap')
axes[1].set_xlabel('Columns')

plt.tight_layout()
plt.savefig('missing_values_overview.png', dpi=300)
plt.show()

# =========================
# HANDLE MISSING VALUES
# =========================

# Age → Mean Imputation
if 'Age' in df.columns:
    age_imputer = SimpleImputer(strategy='mean')
    df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()

# Fare → Median Imputation
if 'Fare' in df.columns:
    fare_imputer = SimpleImputer(strategy='median')
    df['Fare'] = fare_imputer.fit_transform(df[['Fare']]).ravel()

# Embarked → Mode Imputation
if 'Embarked' in df.columns:
    embarked_imputer = SimpleImputer(strategy='most_frequent')
    df['Embarked'] = embarked_imputer.fit_transform(
        df[['Embarked']]
    ).ravel()

# =========================
# DROP CABIN COLUMN
# =========================

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)
    print("\nCabin column dropped successfully.")

# =========================
# OUTLIER DETECTION
# =========================

if 'Fare' in df.columns:

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        x=df['Fare'],
        color='skyblue'
    )

    plt.title("Fare Outliers Before Treatment")

    plt.tight_layout()

    plt.savefig('fare_boxplot_before.png', dpi=300)

    plt.show()

# =========================
# HANDLE OUTLIERS USING IQR
# =========================

if 'Fare' in df.columns:

    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("\n" + "=" * 50)
    print("OUTLIER LIMITS")
    print("=" * 50)

    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)

    # Capping Outliers Instead of Removing Rows
    df['Fare'] = np.where(
        df['Fare'] > upper_bound,
        upper_bound,
        df['Fare']
    )

    df['Fare'] = np.where(
        df['Fare'] < lower_bound,
        lower_bound,
        df['Fare']
    )

# =========================
# VERIFY CLEANED DATA
# =========================

print("\n" + "=" * 50)
print("MISSING VALUES AFTER CLEANING")
print("=" * 50)

print(df.isnull().sum())

print("\n" + "=" * 50)
print("CLEANED DATASET SHAPE")
print("=" * 50)

print(df.shape)

# =========================
# DISTRIBUTION PLOTS
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age Distribution
sns.histplot(
    df['Age'].dropna(),
    kde=True,
    bins=30,
    ax=axes[0],
    color='salmon'
)

axes[0].set_title('Age Distribution')

# Fare Distribution
sns.histplot(
    df['Fare'].dropna(),
    kde=True,
    bins=30,
    ax=axes[1],
    color='dodgerblue'
)

axes[1].set_title('Fare Distribution')

plt.tight_layout()

plt.savefig('age_fare_distributions.png', dpi=300)

plt.show()

# =========================
# SURVIVAL COUNTS
# =========================

if 'Survived' in df.columns:

    plt.figure(figsize=(6, 4))

    sns.countplot(
        x='Survived',
        data=df,
        palette='Set2'
    )

    plt.title('Survival Counts')

    plt.tight_layout()

    plt.savefig('survival_counts.png', dpi=300)

    plt.show()

# =========================
# SURVIVAL BY SEX & PCLASS
# =========================

required_cols = ['Sex', 'Survived', 'Pclass']

if all(col in df.columns for col in required_cols):

    g = sns.catplot(
        x='Sex',
        hue='Survived',
        col='Pclass',
        data=df,
        kind='count',
        palette='Set1',
        height=4,
        aspect=0.9
    )

    g.fig.suptitle(
        'Survival by Sex and Passenger Class',
        y=1.03
    )

    g.savefig('survival_by_sex_pclass.png', dpi=300)

    plt.show()

# =========================
# CORRELATION HEATMAP
# =========================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(10, 8))

corr_matrix = df[numeric_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True
)

plt.title('Correlation Matrix')

plt.tight_layout()

plt.savefig('correlation_matrix.png', dpi=300)

plt.show()

# =========================
# OPTIONAL ENCODING
# =========================

# Convert categorical columns into numeric form
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({
        'male': 0,
        'female': 1
    })

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].map({
        'S': 0,
        'C': 1,
        'Q': 2
    })

# =========================
# SAVE CLEANED DATASET
# =========================

df.to_csv("cleaned_test.csv", index=False)

print("\n" + "=" * 50)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Cleaned dataset saved as:")
print("cleaned_test.csv")
