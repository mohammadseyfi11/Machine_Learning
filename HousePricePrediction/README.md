# House Price Prediction - README

## Project Overview

This project focuses on predicting house prices using advanced regression techniques. The dataset used is from the Ames Housing Dataset, which contains a comprehensive set of features describing residential homes in Ames, Iowa.

The goal is to build a robust model that accurately predicts the final sale price of a home based on its attributes.

## Files Included

- House_Price_Prediction_Colab.ipynb – Main Jupyter notebook containing data preprocessing, exploratory data analysis (EDA), feature engineering, and modeling.
- train.csv – Training dataset provided by Kaggle.
- test.csv – Test dataset for which predictions are to be made.
- AmesHousing.csv – Alternative version of the dataset used for training.
- README.txt – This file.

## Features Used

The dataset includes over 80 features covering various aspects of residential houses, including:

- Location: Neighborhood, Zoning
- Structure: Number of rooms, basement type, roof type, foundation
- Condition & Quality: Overall quality, overall condition, exterior condition
- Size: Lot area, living area, basement square footage
- Age: Year built, year remodeled
- Utilities & Amenities: Type of utilities, presence of fireplace, garage, pool, etc.

## Technologies & Libraries Used

- Python
- Pandas – For data manipulation
- NumPy – For numerical operations
- Seaborn / Matplotlib – For data visualization
- Scikit-learn – For machine learning models and evaluation metrics
- XGBoost, LightGBM, CatBoost, Gradient Boosting – Advanced gradient boosting algorithms
- Statsmodels – For statistical testing
- KNeighborsRegressor – KNN-based imputation for missing values

## Exploratory Data Analysis (EDA)

- Distribution of SalePrice: Visualized against normal distribution; log transformation applied due to skewness.
- Correlation Matrix: Identified top correlated features like OverallQual, GrLivArea, and TotalBsmtSF.
- Categorical vs SalePrice Plots: Box plots, violin plots, and strip plots used to understand relationships between categorical variables and target.

## Data Preprocessing

- Handling Missing Values:
  - Removed features with high percentage of missing values (e.g., PoolQC, MiscFeature)
  - Imputed missing values for categorical features with 'None' or mode
  - Used KNN Regressor for numerical missing value imputation
- Feature Engineering:
  - Created new features like Total_Bathrooms, Total_Home_Quality, HighQualSF
- Encoding:
  - One-hot encoding applied to categorical features
- Normalization:
  - Log transformation applied to skewed numeric features
  - Target variable (SalePrice) also log-transformed

## Modeling

Multiple regression models were evaluated using 5-fold cross-validation:

| Model                  | RMSE (CV Mean) | RMSE (CV Std) |
|------------------------|----------------|---------------|
| Linear Regression      | 0.1399         | 0.0088        |
| Bayesian Ridge         | 0.1228         | 0.0129        |
| LGBM Regressor         | 0.1252         | 0.0113        |
| Support Vector Regressor | 0.2712       | 0.0148        |
| Decision Tree          | 0.2021         | 0.0098        |
| Random Forest          | 0.1374         | 0.0121        |
| XGBoost                | 0.1320         | 0.0118        |
| Gradient Boosting      | 0.1257         | 0.0116        |
| CatBoost               | 0.1148         | 0.0150        |
| Stacked Regressor      | 0.1172         | 0.0151        |

### Best Performing Model: CatBoost Regressor

- Achieved lowest RMSE score of 0.0961 on validation set
- Feature importance extracted showing key drivers of house prices

## Results Visualization

- QQ-plots and histograms to assess normality
- Heatmap of correlation matrix
- Bar plots comparing RMSE across different models
- Feature importance plot from CatBoost

## Future Improvements

- Explore more sophisticated stacking methods
- Hyperparameter tuning via GridSearch or Bayesian Optimization
- Incorporate external datasets (e.g., economic indicators, school ratings)
- Try deep learning approaches with TensorFlow/Keras


Thank you for checking this project! 🏡📈
