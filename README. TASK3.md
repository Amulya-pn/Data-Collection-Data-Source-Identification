# 📊 Exploratory Data Analysis (EDA) on Iris Dataset

A complete Exploratory Data Analysis (EDA) project performed using Python, Pandas, Matplotlib, and Seaborn.

This project focuses on understanding the dataset through statistical summaries and visualizations to identify patterns, relationships, and trends.

---

# 📌 Project Overview

Exploratory Data Analysis (EDA) is an important step in Data Science that helps understand the structure and characteristics of data before applying Machine Learning algorithms.

In this project, various visualization techniques were used to analyze the Iris Dataset and extract meaningful insights.

---

# 🛠️ Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Programming Language      |
| Pandas           | Data Analysis             |
| NumPy            | Numerical Operations      |
| Matplotlib       | Data Visualization        |
| Seaborn          | Statistical Visualization |
| Jupyter Notebook | Development Environment   |

---

# 📂 Project Structure

```text
EDA-Task3/
│
├── EDA_Task3.ipynb
├── iris.csv
├── documentation.docx
├── README.md
│
├── histogram.png
├── boxplot.png
├── scatterplot.png
├── correlation_heatmap.png
└── pairplot.png
```

---

# 📊 Dataset Information

Dataset: Iris Dataset

Features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width
* Species

Total Records: 150

---

# ✨ Analysis Performed

## 📈 1. Univariate Analysis

Analyzed individual features using:

* Histograms
* Box Plots
* Summary Statistics

Purpose:

* Understand data distribution
* Detect outliers
* Measure spread of data

---

## 🔍 2. Bivariate Analysis

Analyzed relationships between variables using:

* Scatter Plots
* Bar Charts

Purpose:

* Identify relationships between features
* Compare different species

---

## 🔥 3. Correlation Analysis

Generated a correlation matrix and heatmap to measure relationships among numerical variables.

Visualization:

```python
sns.heatmap(df.corr(), annot=True)
```

---

## 📊 4. Pairplot Analysis

Created pairplots to visualize relationships between all numerical features.

Visualization:

```python
sns.pairplot(df, hue="species")
```

---

# 📁 Generated Visualizations

| Visualization       | Purpose                 |
| ------------------- | ----------------------- |
| Histogram           | Feature Distribution    |
| Box Plot            | Outlier Detection       |
| Scatter Plot        | Feature Relationships   |
| Correlation Heatmap | Correlation Analysis    |
| Pairplot            | Multi-variable Analysis |

---

# 🎯 Key Insights

* Petal Length and Petal Width show strong positive correlation.
* Setosa species is clearly separable from the other species.
* Most features follow a near-normal distribution.
* Only a few outliers were observed in the dataset.
* Pairplots reveal clear clustering among species.

---

# ▶️ How to Run

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
EDA_Task3.ipynb
```

---

# 🎓 Learning Outcomes

* Learned Exploratory Data Analysis techniques
* Understood data distributions and correlations
* Gained experience with data visualization
* Learned how to identify patterns and outliers
* Improved data interpretation skills

---

# 🙌 Acknowledgements

* Iris Dataset
* Pandas Documentation
* Matplotlib Documentation
* Seaborn Documentation
