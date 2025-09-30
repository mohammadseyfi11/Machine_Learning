# Auto MPG Regression Analysis 🚗⛽

This project focuses on exploratory data analysis (EDA), data preprocessing, and regression modeling to predict automobile fuel efficiency (MPG) using the well-known [Auto MPG dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg).

## ✨ Project Features
- Load and clean the dataset
- Perform statistical and visual exploratory data analysis
- Handle missing values and encode categorical features (One-hot encoding)
- Split the dataset into training and testing sets
- Normalize numerical features
- Train and evaluate multiple regression models (Linear, Ridge, Lasso, Random Forest, ...)
- Select the best model based on RMSE
- Visualize prediction results

## 📊 Workflow
1. **Exploratory Data Analysis (EDA):**
   - Statistical summary of features
   - Scatter plots and histograms to explore relationships

2. **Data Preprocessing:**
   - Replace missing values in `Horsepower` with the median
   - Drop the `Car Name` column
   - Apply one-hot encoding to `Origin` and `Model Year`

3. **Data Splitting & Normalization:**
   - Train-test split (80/20)
   - Normalize numerical features using `StandardScaler`

4. **Modeling & Evaluation:**
   - Train multiple regression models
   - Compare models using MAE, MSE, RMSE, and R²
   - Ridge regression selected as the best model (lowest RMSE)

5. **Visualization:**
   - Plot Actual vs Predicted values for the best model

## ✅ Results
- **Ridge Regression** achieved the best performance with RMSE ≈ `2.88` and R² ≈ `0.84` on the test set.
- Linear Regression and Random Forest also performed reasonably well.

## 📦 Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>

