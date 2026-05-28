# 🚢 Titanic Dataset Data Cleaning & Exploratory Data Analysis

A complete **Data Cleaning** and **Exploratory Data Analysis (EDA)** project using the famous Titanic dataset with Python 🐍

This project demonstrates professional data preprocessing techniques including:

✨ Missing Value Handling  
✨ Duplicate Removal  
✨ Outlier Detection & Treatment  
✨ Data Visualization  
✨ Correlation Analysis  
✨ Feature Encoding  
✨ Exporting Cleaned Data  

---

# 📌 Project Overview

The Titanic dataset is one of the most popular beginner-friendly datasets in **Data Science** and **Machine Learning**.

This project focuses on cleaning raw passenger data and performing detailed Exploratory Data Analysis (EDA) to discover meaningful patterns inside the dataset 📊

The cleaned dataset can later be used for Machine Learning models such as:

- 🤖 Logistic Regression
- 🌲 Decision Trees
- 🌳 Random Forest
- 📈 Support Vector Machines

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming Language |
| 🐼 Pandas | Data Manipulation |
| 🔢 NumPy | Numerical Operations |
| 📊 Matplotlib | Data Visualization |
| 🎨 Seaborn | Statistical Visualization |
| 🤖 Scikit-learn | Missing Value Imputation |

---

# 📂 Project Structure

```text
Titanic-EDA-Project/
│
├── titanic dataset.csv
├── cleaned_test.csv
├── main.py
│
├── missing_values_overview.png
├── fare_boxplot_before.png
├── age_fare_distributions.png
├── survival_counts.png
├── survival_by_sex_pclass.png
├── correlation_matrix.png
│
└── README.md
```

---

# 📊 Dataset Information

The dataset contains passenger details from the Titanic disaster 🚢

| Column | Description |
|---|---|
| PassengerId | Unique Passenger ID |
| Survived | Survival Status (0 = No, 1 = Yes) |
| Pclass | Passenger Class |
| Name | Passenger Name |
| Sex | Gender |
| Age | Passenger Age |
| SibSp | Number of Siblings/Spouses |
| Parch | Number of Parents/Children |
| Ticket | Ticket Number |
| Fare | Ticket Fare |
| Cabin | Cabin Number |
| Embarked | Port of Embarkation |

---

# ✨ Features Implemented

# 📥 1. Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("titanic dataset.csv")
```

The project displays:

✅ First 5 Rows  
✅ Dataset Information  
✅ Dataset Shape  

---

# 🔍 2. Duplicate Detection

Duplicate rows are detected and removed.

```python
duplicates = df.duplicated().sum()

if duplicates > 0:
    df.drop_duplicates(inplace=True)
```

---

# 🧩 3. Missing Value Analysis

The project checks missing values using:

```python
df.isnull().sum()
```

Missing value percentages are also calculated 📉

---

# 📈 4. Missing Value Visualization

Two visualizations are generated:

📊 Missing Value Bar Plot  
🔥 Missing Value Heatmap  

Generated file:

```text
missing_values_overview.png
```

---

# 🛠️ 5. Missing Value Handling

Different imputation techniques are used for different columns.

| Column | Method |
|---|---|
| Age | Mean Imputation |
| Fare | Median Imputation |
| Embarked | Mode Imputation |

Example:

```python
age_imputer = SimpleImputer(strategy='mean')
df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()
```

---

# 🗑️ 6. Dropping Unnecessary Columns

The Cabin column contains too many missing values and is removed.

```python
df.drop(columns=['Cabin'], inplace=True)
```

---

# 📦 7. Outlier Detection

Outliers in the Fare column are visualized using a boxplot 📦

Generated file:

```text
fare_boxplot_before.png
```

---

# 📏 8. Outlier Treatment Using IQR

The Interquartile Range (IQR) method is used.

Steps:

✅ Calculate Q1  
✅ Calculate Q3  
✅ Compute IQR  
✅ Determine lower & upper bounds  
✅ Cap extreme values  

Example:

```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1
```

---

# 📊 9. Distribution Analysis

Histograms are generated for:

🎂 Age Distribution  
💰 Fare Distribution  

Generated file:

```text
age_fare_distributions.png
```

---

# 🛟 10. Survival Analysis

The project analyzes passenger survival patterns using:

📊 Countplots  
👨‍👩‍👧 Survival by Gender  
🎟️ Survival by Passenger Class  

Generated files:

```text
survival_counts.png
survival_by_sex_pclass.png
```

---

# 🔥 11. Correlation Analysis

A correlation heatmap is generated for numeric features 📈

Generated file:

```text
correlation_matrix.png
```

Example:

```python
corr_matrix = df[numeric_cols].corr()
```

---

# 🔢 12. Feature Encoding

Categorical variables are converted into numerical format.

| Column | Encoding |
|---|---|
| Sex | male → 0, female → 1 |
| Embarked | S → 0, C → 1, Q → 2 |

Example:

```python
df['Sex'] = df['Sex'].map({
    'male': 0,
    'female': 1
})
```

---

# 💾 13. Export Cleaned Dataset

The cleaned dataset is saved as:

```text
cleaned_test.csv
```

Example:

```python
df.to_csv("cleaned_test.csv", index=False)
```

---

# ⚙️ Installation

# 📥 Clone Repository

```bash
git clone https://github.com/your-username/Titanic-EDA-Project.git
```

---

# 📂 Navigate to Project Folder

```bash
cd Titanic-EDA-Project
```

---

# 📦 Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# ▶️ How to Run

Run the Python script:

```bash
python main.py
```

---

# 📁 Generated Output Files

| File Name | Description |
|---|---|
| missing_values_overview.png | Missing value visualization |
| fare_boxplot_before.png | Fare outlier boxplot |
| age_fare_distributions.png | Distribution plots |
| survival_counts.png | Survival count plot |
| survival_by_sex_pclass.png | Survival analysis |
| correlation_matrix.png | Correlation heatmap |
| cleaned_test.csv | Final cleaned dataset |

---

# 🖼️ Sample Screenshots

Add generated images here.

Example:

```markdown
![Correlation Matrix](correlation_matrix.png)
```

```markdown
![Survival Counts](survival_counts.png)
```

---

# 🎯 Learning Outcomes

This project demonstrates:

✅ Data Cleaning  
✅ Exploratory Data Analysis  
✅ Missing Value Treatment  
✅ Outlier Handling  
✅ Data Visualization  
✅ Feature Engineering  
✅ Dataset Preparation for Machine Learning  

---

# 🚀 Future Improvements

Possible future enhancements include:

🤖 Machine Learning model training  
📏 Feature Scaling  
⚡ Hyperparameter Tuning  
🌐 Streamlit Deployment  
📊 Interactive Dashboards  
📈 Model Performance Evaluation  

---

# 💻 Example Commands

# ▶️ Run Jupyter Notebook

```bash
jupyter notebook
```

# 📦 Check Installed Packages

```bash
pip list
```


---

# 🙌 Acknowledgements

Dataset Sources:

- 🚢 Titanic Dataset
- 📊 Kaggle
- 🎨 Seaborn Sample Datasets

