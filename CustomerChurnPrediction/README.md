TELECOM CUSTOMER CHURN PREDICTION
-----------------------------------

1. INTRODUCTION AND OBJECTIVES
------------------------------
The project begins by defining customer churn and highlighting its financial impact on businesses, noting that attracting a new customer costs five times more than retaining an existing one. The telecommunications industry faces a significant annual churn rate of 15-25 percent. The primary goal of the notebook is to analyze customer data to identify the key drivers of churn and build a machine learning model to predict which customers are likely to leave. The analysis aims to answer specific questions, such as the overall churn percentage and whether factors like gender or service type influence a customer's decision to churn.

2. SETUP AND DATA LOADING
--------------------------
The first step in the code is to import all the necessary Python libraries. This includes `pandas` for data manipulation, `matplotlib` and `seaborn` for static visualizations, and `plotly` for interactive charts. For the machine learning portion, libraries from `scikit-learn` are imported for data preprocessing, splitting the data, and implementing various classification models. After setting up the environment, the code loads the `WA_Fn-UseC_-Telco-Customer-Churn` dataset from a CSV file into a pandas DataFrame, which is the main data structure used for the analysis.

3. DATA UNDERSTANDING AND CLEANING
----------------------------------
To begin understanding the data, the first few rows of the DataFrame are displayed. An initial inspection of the data types and for missing values is performed.

The data cleaning process involves several steps:
- The `customerID` column is dropped because it is a unique identifier and does not provide predictive value.
- The `TotalCharges` column is converted from a text-based object to a numerical type. This conversion reveals 11 missing values.
- Further investigation shows that these missing `TotalCharges` correspond to customers who have a `tenure` of 0 months.
- These 11 rows are removed from the dataset.
- To handle any other potential missing values, the code is set to fill them with the average `TotalCharges` value.
- For better readability in visualizations, the `SeniorCitizen` column, which contains 0s and 1s, is mapped to "No" and "Yes" strings.

4. EXPLORATORY DATA ANALYSIS (EDA)
----------------------------------
This section uses visualizations to uncover patterns in the data related to customer churn.
- Interactive pie charts are created to show the distribution of gender and churn. The findings show that the customer base is nearly evenly split between males and females, and the overall churn rate is 26.6%.
- A histogram is generated to analyze the relationship between the customer's contract type and churn. It clearly shows that customers with month-to-month contracts are far more likely to churn than those on one or two-year contracts.
- Another histogram reveals that customers using "Electronic check" as their payment method have a higher tendency to churn.
- A box plot comparing customer tenure with churn status indicates that new customers with a shorter tenure are more likely to leave.
- Other visualizations show that customers without add-on services like `OnlineSecurity` and `TechSupport`, or those with Fiber optic internet service, have higher churn rates.

5. DATA PREPROCESSING FOR MODELING
----------------------------------
Before training machine learning models, the data must be prepared.
- A function is defined and applied to the DataFrame to convert all text-based categorical columns (like `gender`, `Contract`, etc.) into numerical representations using a `LabelEncoder`. This is necessary because machine learning algorithms require numerical input.
- The dataset is then separated into features (X) and the target variable (y, the `Churn` column).
- The data is split into a training set (70% of the data) and a testing set (30%). The model will learn from the training set, and its performance will be evaluated on the unseen test set.
- The numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardized using `StandardScaler`. This process scales the features to have a mean of 0 and a standard deviation of 1, preventing features with larger scales from dominating the model.

6. MACHINE LEARNING MODEL EVALUATION
------------------------------------
Several different classification models are trained and evaluated to find the best one for predicting churn. For each model, the process is the same:
1. The model is initialized.
2. It is trained on the preprocessed training data (`X_train`, `y_train`).
3. It makes predictions on the test data (`X_test`).
4. Its performance is measured using accuracy and a detailed classification report, which includes precision, recall, and F1-score.

The models evaluated this way are:
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Random Forest
- Logistic Regression
- Decision Tree Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier

7. FINAL MODEL: VOTING CLASSIFIER
----------------------------------
To potentially improve the predictive power, a `VotingClassifier` is implemented as the final model. This is an ensemble method that combines multiple different models. In this notebook, it combines the Gradient Boosting, Logistic Regression, and AdaBoost models. The final prediction for each customer is determined by a majority vote from these three constituent models. The performance of this final ensemble model is then evaluated, and a confusion matrix is plotted to visualize its predictions, showing that it correctly identified 1400 non-churners and 324 churners from the test set.
