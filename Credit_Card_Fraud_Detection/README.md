# Credit Card Fraud Detection using Machine Learning

## Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, with a very small percentage of fraudulent transactions, which presents a significant challenge. The primary goal is to build models that can effectively identify these rare fraudulent cases while minimizing false positives.

The analysis involves thorough Exploratory Data Analysis (EDA), feature engineering, data preprocessing, and the training and evaluation of several classification models. Special attention is paid to evaluation metrics suitable for imbalanced datasets.

## Dataset

* **Source:** The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle, provided by MLG-ULB.
    * It can be downloaded using `kagglehub.dataset_download("mlg-ulb/creditcardfraud")`.
* **Shape:** The dataset contains 284,807 transactions and 31 features.
* **Features:**
    * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
    * `Amount`: Transaction amount.
    * `V1` - `V28`: Anonymized features, which are the result of a PCA transformation.
    * `Class`: The target variable, where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.
* **Class Imbalance:** The dataset is highly imbalanced:
    * Non-Fraudulent (Class 0): Approximately 99.827% of transactions (284,315 instances).
    * Fraudulent (Class 1): Approximately 0.173% of transactions (492 instances).
    This imbalance means that accuracy alone is a misleading metric for model evaluation.

## Exploratory Data Analysis (EDA)

### Missing Values
* A check for missing values was performed, and the dataset was found to be clean regarding nulls.

### Transaction Amount
* The distribution of transaction amounts is highly skewed to the right.
* A log transformation (`np.log1p`) was applied to the `Amount` feature to handle this skewness, making the distribution more symmetrical and improving visibility of patterns.
* The `Log_Amount` distribution by class shows that fraudulent transactions generally involve smaller amounts.

### Transaction Time
* The `Time` feature was analyzed. Both legitimate and fraudulent transactions appear throughout the time spectrum.
* Fraudulent transactions show a slightly different pattern but no clear cut-off time perfectly separates fraud from non-fraud.
* Feature Engineering:
    * `Time_Sec_of_Day`: Time in seconds within a 24-hour cycle.
    * `Time_Hour_of_Day`: Hour of the day (0-23).
    * Analysis of fraudulent transactions by `Time_Hour_of_Day` revealed potential patterns, with certain hours showing higher numbers or rates of fraud.

### V-Features (Anonymized PCA Features)
* Distributions of selected V-features (e.g., V4, V10, V12, V14, V17) by class showed clear separation between non-fraudulent and fraudulent transactions.
* Features like V14 and V17 exhibit highly distinct distributions, suggesting they are extremely important for identifying fraud.
* Statistical tests (T-test, Mann-Whitney U, Kolmogorov-Smirnov) were conducted on 'Time', 'Amount', and V1-V28 features.
    * Most features showed statistically significant differences in their means and distributions between fraudulent and non-fraudulent transactions.
* Skewness and Kurtosis were calculated for 'Amount' and V-features, grouped by class.

### Correlations
* Correlation matrices for V-features were computed separately for non-fraudulent and fraudulent transactions.
* The correlation patterns differ notably between the two classes, indicating that relationships between features change in fraudulent scenarios.

## Preprocessing

1.  **Feature and Target Separation:**
    * `X`: Features (all columns except 'Class').
    * `y`: Target variable ('Class').
2.  **Feature Scaling:**
    * `StandardScaler` was applied to the `Time` and `Amount` features.
    * The V-features (V1-V28) were not scaled as they are already PCA-transformed.
    * A `ColumnTransformer` was used to apply scaling selectively.
3.  **Train-Test Split:**
    * The data was split into training (80%) and testing (20%) sets.
    * Stratification (`stratify=y`) was used during the split to ensure that the class proportions were maintained in both sets due to the imbalanced nature of the data.
    * Training set shape: (227845, 32), Test set shape: (56962, 32).
    * Fraudulent cases in training set: 394, Fraudulent cases in test set: 98.

## Model Building and Evaluation

Four different classification models were trained and evaluated. Hyperparameter tuning was performed using `GridSearchCV` with 5-fold `StratifiedKFold` cross-validation.  The primary scoring metric for tuning was 'f1' (F1-score for the minority class).

**Evaluation Metrics:** Due to class imbalance, the focus was on:
* Precision (for the fraud class)
* Recall (for the fraud class)
* F1-Score (for the fraud class)
* ROC AUC Score
* Confusion Matrix

### Models and Results

1.  **Logistic Regression**
    * Hyperparameters included `solver='liblinear'`.
    * Tuned for `classifier__C`: .
    * **Best Parameters:** `{'classifier__C': 0.01}`.
    * **Test Set Performance:**
        * Accuracy: 0.9758
        * Precision (Fraud): 0.0616
        * Recall (Fraud): 0.9184
        * F1-Score (Fraud): 0.1155
        * ROC AUC: 0.9725
    * **Top Coefficients (Positive):** `remainder__V4` (0.921), `num_scaler__Amount` (0.826), `remainder__V22` (0.603).
    * **Top Coefficients (Negative):** `remainder__V14` (-1.132), `remainder__V10` (-0.895), `remainder__V12` (-0.768).

2.  **Decision Tree Classifier**
    * Hyperparameters included `class_weight='balanced'`.
    * Tuned for `classifier__max_depth`: , `classifier__min_samples_leaf`: .
    * **Best Parameters:** `{'classifier__max_depth': None, 'classifier__min_samples_leaf': 1}`.
    * **Test Set Performance:**
        * Accuracy: 0.9990
        * Precision (Fraud): 0.7158
        * Recall (Fraud): 0.6939
        * F1-Score (Fraud): 0.7047
        * ROC AUC: 0.8467
    * **Top Feature Importances:** `remainder__V14` (0.733), `remainder__V4` (0.067), `remainder__V12` (0.026).

3.  **Random Forest Classifier**
    * Hyperparameters included `class_weight='balanced'`.
    * Tuned for `classifier__n_estimators`: [100, 200], `classifier__max_depth`: [5, 10], `classifier__min_samples_leaf`: [5, 10].
    * **Best Parameters:** `{'classifier__max_depth': 10, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 100}`.
    * **Test Set Performance:**
        * Accuracy: 0.9993
        * Precision (Fraud): 0.7593
        * Recall (Fraud): 0.8367
        * F1-Score (Fraud): 0.7961
        * ROC AUC: 0.9746
    * **Top Feature Importances:** `remainder__V14` (0.227), `remainder__V12` (0.138), `remainder__V4` (0.112).

4.  **XGBoost Classifier**
    * Hyperparameters included `use_label_encoder=False`, `eval_metric='logloss'`, and `scale_pos_weight` (calculated based on class distribution in the training set).
    * Tuned for `classifier__n_estimators`: [100, 200], `classifier__max_depth`: [3, 5], `classifier__learning_rate`: [0.01, 0.1].
    * **Best Parameters:** `{'classifier__learning_rate': 0.1, 'classifier__max_depth': 5, 'classifier__n_estimators': 200}`.
    * **Test Set Performance:**
        * Accuracy: 0.9994
        * Precision (Fraud): 0.8367
        * Recall (Fraud): 0.8367
        * F1-Score (Fraud): 0.8367
        * ROC AUC: 0.9778
    * **Top Feature Importances:** `remainder__V14` (0.515), `remainder__V4` (0.068), `remainder__V12` (0.042).

## Feature Importance

Feature importance was analyzed for tree-based models (Decision Tree, Random Forest, XGBoost) and coefficients were examined for Logistic Regression. Across the tree-based models, features like `V14`, `V4`, and `V12` consistently appeared as highly important.

## Conclusion

The project successfully demonstrated the process of building and evaluating machine learning models for credit card fraud detection on an imbalanced dataset.
* XGBoost and Random Forest generally performed the best in terms of F1-score, Precision, Recall for the fraud class, and ROC AUC score.
* Logistic Regression, while achieving high recall, suffered from very low precision for the fraud class.
* The Decision Tree showed reasonable performance but was generally outperformed by the ensemble methods.

The use of techniques to handle class imbalance (e.g., `class_weight`, `scale_pos_weight`) and the focus on appropriate evaluation metrics were crucial for this task. Feature engineering (like `Time_Hour_of_Day`) and careful EDA also provided valuable insights.

## Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn (for `train_test_split`, `GridSearchCV`, `StandardScaler`, metrics, etc.)
* xgboost
* kagglehub (for dataset download)
