# 📘 Task 4: Feature Encoding & Scaling


### 📌 Project Title
Feature Encoding & Scaling using Adult Census Income Dataset


# 📖 Introduction

Feature Encoding and Feature Scaling are essential preprocessing techniques in Machine Learning.

Real-world datasets often contain:

- 🔤 Categorical Features
- 🔢 Numerical Features with different scales
- 📊 Non-uniform feature distributions
- ⚠ Features that cannot be directly used by Machine Learning algorithms

Most Machine Learning algorithms require numerical input data. Therefore, categorical variables must be transformed into numerical values, and numerical variables should be scaled appropriately.

This project demonstrates:

✅ Label Encoding

✅ One-Hot Encoding

✅ Standardization

✅ Normalization

✅ Data Visualization

✅ Correlation Analysis

✅ Statistical Comparison

✅ Dataset Preparation for Machine Learning

---

# 🎯 Objective

The primary objectives of this project are:

- Understand the Adult Census Income Dataset
- Identify categorical and numerical features
- Convert categorical data into numerical format
- Apply Label Encoding
- Apply One-Hot Encoding
- Perform Standardization
- Perform Min-Max Normalization
- Compare feature distributions
- Prepare the dataset for Machine Learning applications

---

# 📂 Dataset Description

## Dataset Used

Adult Census Income Dataset

The dataset contains demographic and employment-related information used to predict whether a person's annual income exceeds $50,000.

---

# 📊 Dataset Features

| Feature | Description |
|----------|------------|
| age | Age of individual |
| workclass | Employment type |
| fnlwgt | Final weight |
| education | Education level |
| education-num | Years of education |
| marital-status | Marital status |
| occupation | Occupation type |
| relationship | Relationship status |
| race | Race category |
| sex | Gender |
| capital-gain | Capital gain |
| capital-loss | Capital loss |
| hours-per-week | Working hours per week |
| native-country | Country of origin |
| income | Income category |

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Scikit-Learn | Encoding & Scaling |

---

# ⚙ Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
```

---

# 📥 Step 1: Loading the Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv(
    "adult_data.csv",
    header=None,
    names=column_names,
    skipinitialspace=True
)
```

After loading:

- Dataset structure is examined
- First records are displayed
- Data types are checked
- Shape of dataset is verified

Example:

```python
print(df.head())
print(df.info())
print(df.shape)
```

---

# 🔍 Step 2: Identifying Categorical Features

Categorical columns are identified using:

```python
categorical_cols = df.select_dtypes(
    include=['object']
).columns
```

### Identified Features

- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country
- income

---

# 🔢 Step 3: Label Encoding

Machine Learning models require numerical labels.

The target variable **income** is encoded.

### Original Values

| Income |
|----------|
| <=50K |
| >50K |

### Encoded Values

| Income | Encoded |
|----------|----------|
| <=50K | 0 |
| >50K | 1 |

Implementation:

```python
label_encoder = LabelEncoder()

df['income_label'] = label_encoder.fit_transform(
    df['income']
)
```

---

# 🏷 Step 4: One-Hot Encoding

Several categorical variables do not possess any natural ordering.

Examples:

- workclass
- occupation
- education
- race
- native-country

One-Hot Encoding converts categories into binary columns.

### Example

Original:

| Sex |
|------|
| Male |
| Female |

Encoded:

| sex_Male |
|-----------|
| 1 |
| 0 |

Implementation:

```python
df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)
```

---

# ⚠ Dummy Variable Trap

When all dummy variables are retained, one variable can be predicted from others.

This creates:

- Multicollinearity
- Redundant information
- Poor model performance

To avoid this issue:

```python
drop_first=True
```

is used during One-Hot Encoding.

---

# 📏 Step 5: Feature Scaling

Numerical features have different ranges.

Example:

| Feature | Range |
|----------|----------|
| age | 17 – 90 |
| capital-gain | 0 – 99999 |
| hours-per-week | 1 – 99 |

Without scaling:

- Larger values dominate smaller values
- Distance calculations become biased
- Model performance decreases

---

# 📊 Step 6: Standardization

Standardization transforms features so that:

- Mean = 0
- Standard Deviation = 1

### Formula

```
z = (x - μ) / σ
```

Implementation:

```python
standard_scaler = StandardScaler()

df_standardized[continuous_features] = (
    standard_scaler.fit_transform(
        df_standardized[continuous_features]
    )
)
```

### Advantages

- Removes scale differences
- Improves convergence
- Useful for:
  - Logistic Regression
  - KNN
  - SVM
  - PCA

---

# 📉 Step 7: Normalization

Normalization rescales values between 0 and 1.

### Formula

```
x' = (x - xmin) / (xmax - xmin)
```

Implementation:

```python
minmax_scaler = MinMaxScaler()

df_normalized[continuous_features] = (
    minmax_scaler.fit_transform(
        df_normalized[continuous_features]
    )
)
```

### Advantages

- Uniform range
- Faster Neural Network training
- Useful for distance-based algorithms

---

# 📈 Step 8: Distribution Analysis

Histograms are generated for:

- Original Data
- Standardized Data
- Normalized Data

Feature used:

```python
feature = "age"
```

Generated File:

```text
age_distribution_comparison.png
```

Purpose:

- Compare distributions
- Observe scaling effects
- Verify transformation

---

# 📊 Step 9: Statistical Comparison

Summary statistics are generated using:

```python
describe()
```

Statistics include:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles

Comparison is performed for:

- Original Data
- Standardized Data
- Normalized Data

---

# 🔥 Step 10: Correlation Analysis

Correlation analysis helps identify relationships between numerical features.

Implementation:

```python
corr_matrix = (
    df_standardized[continuous_features]
    .corr()
)
```

Generated File:

```text
correlation_matrix.png
```

Purpose:

- Detect relationships
- Identify highly correlated variables
- Support feature selection

---

# 📁 Output Files Generated

| File Name | Description |
|------------|------------|
| age_distribution_comparison.png | Distribution Comparison |
| correlation_matrix.png | Correlation Heatmap |
| encoded_dataset.csv | Encoded Dataset |
| standardized_dataset.csv | Standardized Dataset |
| normalized_dataset.csv | Normalized Dataset |

---

# 🎯 Learning Outcomes

This project helped in understanding:

✅ Feature Encoding

✅ Label Encoding

✅ One-Hot Encoding

✅ Dummy Variable Trap

✅ Feature Scaling

✅ Standardization

✅ Normalization

✅ Correlation Analysis

✅ Data Visualization

✅ Machine Learning Preprocessing

---

# 🚀 Future Improvements

Possible future enhancements include:

- Machine Learning Model Training
- Feature Selection
- Principal Component Analysis (PCA)
- Hyperparameter Tuning
- Model Evaluation
- Deployment using Streamlit
- Interactive Dashboard Creation

---

# ✅ Conclusion

This project successfully demonstrates the complete workflow of Feature Encoding and Feature Scaling using the Adult Census Income Dataset.

Categorical variables were converted into machine-readable numerical formats using Label Encoding and One-Hot Encoding. Numerical features were scaled using Standardization and Min-Max Normalization.

The processed dataset is now suitable for Machine Learning algorithms such as:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Trees
- Random Forest
- Neural Networks

