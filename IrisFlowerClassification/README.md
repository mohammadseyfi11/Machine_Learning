# Iris Flower Classification Project

## Overview

This project demonstrates the implementation of various machine learning models to classify iris flower species using sepal and petal measurements. The Iris dataset, one of the most well-known datasets in machine learning, contains 150 samples across three species: setosa, versicolor, and virginica, with four features each.

## Features

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Target Classes

- Setosa
- Versicolor
- Virginica

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib & Seaborn (for visualization)
- Streamlit (for creating a web app)

## Machine Learning Models Implemented

- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier

All models achieved **100% accuracy** on the test set, indicating excellent performance due to the high separability of classes in the Iris dataset.

## Key Steps in the Workflow

1. **Data Loading**: Dataset was loaded using `load_iris()` from sklearn.
2. **Exploratory Data Analysis (EDA)**:
   - Converted data into a Pandas DataFrame for better understanding.
   - Visualized feature relationships using pairplots.
3. **Train-Test Split**: Dataset was split into training and testing sets with a 70-30 ratio.
4. **Feature Scaling**: Standardized features using `StandardScaler`.
5. **Model Training & Evaluation**:
   - Trained multiple classifiers including KNN, Random Forest, SVM, etc.
   - Evaluated performance using confusion matrix, classification report, and accuracy score.
6. **Hyperparameter Tuning**:
   - GridSearchCV used to find the optimal number of neighbors (`n_neighbors`) for KNN.
7. **Cross-Validation**:
   - Performed 5-fold cross-validation to assess generalization capability.
8. **Streamlit Web App**:
   - Created an interactive web application that allows users to input flower dimensions and predict the species.

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 100%     |
| SVM                 | 100%     |
| Decision Tree       | 100%     |
| Random Forest       | 100%     |
| KNN                 | 100%     |

Mean Cross-Validation Accuracy: ~95%

## How to Run the Application

### Prerequisites

Make sure you have Python 3.x installed and install required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit
