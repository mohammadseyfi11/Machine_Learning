# 🍷 Wine Quality Analysis & Prediction

This code snippet contains an end-to-end exploratory data analysis (EDA) and predictive modeling project on the **Wine Quality Dataset**.
The dataset provides physicochemical properties of red wine samples, along with a quality score assigned by wine tasters.
The goal of this project is to **understand the relationships between chemical features and wine quality** and to build predictive models that can classify wine quality based on these features.

---

## 📖 Project Overview
Wine quality is influenced by a variety of chemical properties such as acidity, alcohol content, sulphates, and chlorides.
By analyzing these features, we can uncover patterns that distinguish high-quality wines from lower-quality ones.

This project covers:
- **Data loading and cleaning**
- **Exploratory Data Analysis (EDA)** with visualizations
- **Correlation analysis** to identify the most influential features
- **Feature-target relationships** (alcohol, volatile acidity, sulphates, etc.)
- **Predictive modeling** using machine learning algorithms

---

## 📊 Dataset Information
- **Source:** [Wine Quality Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **Number of samples:** 1143
- **Number of features:** 12 physicochemical features + 1 target (`quality`)
- **Target variable:** `quality` (integer score between 3 and 8)

### Features:
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality` (target label)

---

## 🔎 Exploratory Data Analysis (EDA)
The EDA phase focused on understanding the distribution of features and their relationships with wine quality.

### Key Visualizations:
- **Histograms & KDE plots** → Showed skewness in features like residual sugar and chlorides.
- **Pair Plots** → Highlighted relationships such as:
  - Higher alcohol content is strongly associated with higher quality.
  - Higher volatile acidity is linked to lower quality.
- **Box Plots** → Helped identify outliers in alcohol and sulphates across different quality levels.
- **Correlation Heatmap** → Revealed the strongest correlations:
  - Positive: `alcohol`, `sulphates`, `citric acid`
  - Negative: `volatile acidity`, `density`, `chlorides`

---

## 📌 Insights from Analysis
- **Alcohol**: The strongest positive predictor of wine quality.
- **Volatile Acidity**: Strong negative correlation with quality; higher values reduce wine quality.
- **Sulphates**: Mild positive effect on quality.
- **Citric Acid**: Slight positive correlation, but less consistent.
- **Quality Distribution**: Imbalanced, with most wines rated **5 or 6**, and fewer samples at the extremes (3 or 8).

---

## 🤖 Modeling Approach
Several machine learning models were tested to predict wine quality:

- **Logistic Regression** (binary classification: good vs. bad wine)
- **Random Forest Classifier** (multi-class classification for quality scores)
- **Polynomial Regression** (to capture non-linear relationships between features like alcohol and chlorides)

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

---

## 📈 Results
- **Alcohol** and **volatile acidity** emerged as the most important features.
- Logistic Regression achieved ~74% accuracy on test data.
- Random Forest (with hyperparameter tuning) achieved ~79% accuracy, with balanced precision and recall.
- Polynomial regression captured non-linear trends (e.g., alcohol vs. chlorides).
