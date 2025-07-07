# House Price Prediction Project

## Overview

This project demonstrates a machine learning model for predicting house prices based on various features such as area, number of bedrooms, location, and other structural and qualitative attributes. The dataset used contains 1460 samples with over 80 features describing different aspects of residential homes in Ames, Iowa.

## Features

The dataset includes a wide variety of features such as:
- Lot size
- Number of bedrooms and bathrooms
- Square footage of living areas
- Year built and last renovated
- Garage size and quality
- Basement condition and finish type
- Roof style and material
- Exterior quality
- Sale type and condition
- And many more (see full list in data description)

## Target Variable

- **SalePrice**: The final sale price of the house (in USD)

## Librairess Used

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn (for visualization)
- Scikit-learn (for preprocessing and evaluation)
- LightGBM (Light Gradient Boosting Machine)
- TensorFlow / Keras (optional deep learning components)
- Streamlit (for web application interface)

## Key Steps in the Workflow

1. **Data Loading & Inspection**
   - Loaded the dataset using Pandas.
   - Checked for missing values and explored basic statistics.
   - Displayed first few rows to understand structure.

2. **Exploratory Data Analysis (EDA)**
   - Visualized feature distributions.
   - Identified key relationships between house features and sale prices.
   - Analyzed categorical and numerical variables separately.

3. **Data Preprocessing**
   - Handled missing values by imputation or removal.
   - Encoded categorical features using one-hot encoding or label encoding.
   - Scaled or normalized numerical features if necessary.

4. **Feature Engineering**
   - Created new features from existing ones (e.g., total square footage).
   - Transformed skewed features using log transformation.
   - Selected important features based on correlation or domain knowledge.

5. **Model Training**
   - Trained a **LightGBM Regressor**, which is effective for tabular data.
   - Tuned hyperparameters for better performance.
   - Evaluated the model using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

6. **Optional Deep Learning Model**
   - Built a simple neural network using TensorFlow/Keras.
   - Explored training dynamics and loss curves.

7. **Web Application (Streamlit)**
   - Developed an interactive web app using Streamlit to deploy the model.
   - Allowed users to input house features and get predicted prices.

## Results

- The LightGBM model performed well, capturing trends in the housing market.
- Feature importance plots helped identify the most influential predictors (e.g., overall quality, living area).
- Optional deep learning models showed potential but were not fully optimized.

## How to Run the Application

### Prerequisites

Ensure Python 3.x is installed. Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm tensorflow streamlit
