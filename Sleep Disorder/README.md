 
# Sleep Disorder Dataset Analysis

This project explores a health dataset containing demographic, lifestyle, and physiological variables to understand patterns and predictors of sleep disorders. The analysis includes exploratory data analysis (EDA), visualization, and statistical hypothesis testing.

## 📁 Dataset Overview

The dataset (`Sleep_Health.csv`) includes 374 records with the following columns:

- `Person ID`: Unique identifier
- `Gender`: Male / Female
- `Age`: In years
- `Occupation`: Job title (e.g., Doctor, Software Engineer, etc.)
- `Sleep Duration`: Hours per night (float)
- `Quality of Sleep`: Self-reported score (1–9)
- `Physical Activity Level`: Minutes per day
- `Stress Level`: Self-reported score (1–8)
- `BMI Category`: `Normal`, `Normal Weight`, `Overweight`, `Obese`
- `Blood Pressure`: Systolic/Diastolic (e.g., "126/83")
- `Heart Rate`: Beats per minute
- `Daily Steps`: Number of steps per day
- `Sleep Disorder`: `NaN` (no disorder), `"Sleep Apnea"`, or `"Insomnia"`

> **Note**: The `Sleep Disorder` column has 219 missing values (i.e., 155 labeled cases).

---

## 🧪 Key Transformations

- Parsed `Blood Pressure` into two new integer columns:
  - `Systolic Blood Pressure`
  - `Diastolic Blood Pressure`
- Created a binary target variable:
  - `Has Sleep Disorder` = 1 if `Sleep Disorder` is not NaN, else 0

---

## 📊 Exploratory Data Analysis (EDA)

### Distributions
- Histograms and box plots for all numerical variables (`Age`, `Sleep Duration`, `Quality of Sleep`, etc.)
- Identified right/left skewness and potential outliers (e.g., low `Daily Steps` in some occupations)

### Categorical Breakdowns
- Gender: ~50/50 split
- BMI Category: Most common categories are `Normal` and `Overweight`
- Occupation: 11 distinct roles; some (e.g., `Sales Representative`) show extreme health metrics

### Group Comparisons
- **By Gender**: Females report higher sleep quality and lower stress; males have higher heart rate and blood pressure.
- **By Occupation**:
  - `Sales Representative` shows the lowest sleep duration (5.9 hrs), lowest sleep quality (4/9), highest stress (8/8), and highest blood pressure.
  - `Engineer` and `Accountant` report better sleep quality and lower stress.
- **By BMI Category**:
  - `Obese` and `Overweight` groups show significantly higher systolic/diastolic blood pressure and heart rate.
- **By Sleep Disorder Status**:
  - Individuals with sleep disorders have:
    - Shorter sleep duration
    - Lower sleep quality
    - Higher stress
    - Higher blood pressure and heart rate

### Correlations
- Moderate positive correlation between `Sleep Duration` and `Quality of Sleep` (≈ +0.5)
- Weak or no correlation between `Daily Steps` and `Physical Activity Level` (suggesting non-linear or subjective reporting)

---

## 📉 Visualization Highlights

- Violin plots by occupation for sleep and stress metrics
- Bar plots showing average blood pressure by BMI category
- Scatter plots for key pairs (e.g., Age vs. Heart Rate)
- Pair plots for multivariate relationships
- Count and distribution plots for sleep disorder prevalence

---

## 📐 Statistical Testing

Performed hypothesis tests at α = 0.05:

### Two-Group Comparisons (t-test / Mann-Whitney U)
- **Gender** and **Has Sleep Disorder** (binary groups)
- Significant differences found in:
  - Age, Sleep Duration, Quality of Sleep, Stress Level, Heart Rate, Blood Pressure
- No significant difference in `Physical Activity Level` or `Daily Steps` by gender or disorder status

### Multi-Group Comparisons (ANOVA / Kruskal-Wallis)
- **Occupation** and **BMI Category** (≥3 groups)
- All numerical variables showed significant differences across occupations
- Across BMI categories, only `Physical Activity Level` and (marginally) `Stress Level` showed non-significant differences

> Results were consistent between parametric and non-parametric tests, supporting robustness.

---

## 🎯 Insights & Implications

- Sleep disorders are strongly associated with **poorer sleep metrics**, **higher stress**, and **elevated cardiovascular indicators**.
- **Occupation** is a major determinant of sleep health — high-stress roles (e.g., sales) correlate with worse outcomes.
- **BMI** is a key predictor of blood pressure and heart rate, reinforcing links between metabolic health and sleep.
- The dataset supports using **non-invasive metrics** (e.g., sleep duration, stress level) to screen for sleep disorders.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- SciPy (for statistical tests)

Thank You!
