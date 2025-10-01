# Stock Price Prediction - Apple (AAPL) 📈🍏

This project focuses on predicting the stock price of **Apple Inc. (AAPL)** using historical data from Yahoo Finance and machine learning models.
The workflow includes data collection, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## 📥 Dataset

- **Source:** [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/history)
- **Ticker:** AAPL
- **Period:** 2020-01-01 to 2023-01-01 (3 years, daily data)
- **Columns:**
  - `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

---

## 🛠️ Project Workflow

1. **Data Collection & Exploration**
   - Downloaded AAPL daily stock data using `yfinance`
   - Visualized price trends (COVID-19 crash, recovery, 2022 downturn)
   - Compared `Close` vs `Adj Close` to highlight stock split effects

2. **Data Preprocessing**
   - Checked for missing values
   - Selected features: `Open`, `High`, `Low`, `Volume`
   - Target variable: `Adj Close`

3. **Feature Engineering**
   - Time-based features: year, month, day, dayofweek, etc.
   - Technical indicators:
     - Rolling Mean (10-day)
     - Lag features (previous day price)
     - EMA (50 & 200 days)
     - RSI (Relative Strength Index)
     - MACD & Signal Line
   - Dropped NaN rows from rolling calculations

4. **Modeling**
   - Chronological split: 80% train / 20% test
   - Models trained:
     - Linear Regression
     - Random Forest Regressor
     - Gradient Boosting Regressor

5. **Hyperparameter Tuning**
   - `GridSearchCV` for Random Forest
   - `RandomizedSearchCV` for Gradient Boosting
   - Best model selected based on RMSE & R²

6. **Evaluation & Visualization**
   - Actual vs Predicted plots
   - Feature importance visualization
   - Achieved high accuracy (R² > 0.96 on test set)

---

## 📊 Results

| Model                       | MAE   | RMSE  | R² (test) |
|-----------------------------|-------|-------|-----------|
| Linear Regression           | 1.04  | 1.26  | 0.98      |
| Random Forest Regressor     | 1.33  | 1.74  | 0.97      |
| Gradient Boosting Regressor | 1.35  | 1.72  | 0.97      |
| Tuned Random Forest         | 1.32  | 1.72  | 0.97      |
| Tuned Gradient Boosting     | 1.23  | 1.60  | 0.97      |

- **Best Model:** Tuned Gradient Boosting Regressor (RMSE ≈ 1.60, R² ≈ 0.97)
- Most important features: `Low`, `High`, `Open`

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>

