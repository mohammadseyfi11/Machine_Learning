# Titanic Survival Prediction

## Overview
This project aims to predict the survival of passengers aboard the RMS Titanic. It involves data exploration, preprocessing, feature engineering, and training various classification models to determine whether a passenger survived or not based on features like age, class, sex, etc. The analysis is based on the well-known Titanic dataset from Kaggle.

## Dataset
The project uses the Titanic dataset from Kaggle. The primary file used for training and local evaluation is `train.csv`.  A separate `test.csv` is used for generating predictions for submission.

## Project Workflow

The project follows these main steps:

1.  **Data Loading and Initial Inspection:**
    * The `train.csv` dataset is loaded into a pandas DataFrame.
    * Initial checks include viewing the first few rows (`df.head()`), last few rows (`df.tail()`), general information (`df.info()`), and descriptive statistics (`df.describe()`) to understand the dataset's structure, data types, and basic statistical properties.

2.  **Exploratory Data Analysis (EDA) & Visualization:**
    * **Survival Distribution:**
        * Visualized the overall percentage of survivors and non-survivors.  Approximately 61.62% of passengers in the dataset did not survive, while 38.38% survived.
    * **Survival Rate by Features:**
        * **Sex:** Analyzed survival rate by sex, showing females had a significantly higher survival rate (74.20%) compared to males (18.89%).
        * **Pclass (Passenger Class):** Investigated survival rates by passenger class, indicating higher survival for 1st class (62.96%), followed by 2nd class (47.28%), and then 3rd class (24.24%).
        * **SibSp (Number of Siblings/Spouses Aboard):** Visualized survival rates based on `SibSp`. Passengers with 1 sibling/spouse had the highest survival rate (53.59%), while those with 0 had a 34.54% survival rate. Survival rates generally decreased for higher numbers of SibSp, dropping to 0% for 5 or 8 SibSp.
        * **Embarked (Port of Embarkation):** Analyzed survival rates by port, with Cherbourg ('C') showing the highest survival rate (55.36%), followed by Queenstown ('Q') (38.96%), and Southampton ('S') (33.70%).
    * **Age and Fare Distributions by Survival:**
        * Stacked histograms and density plots were used to visualize how 'Age' and 'Fare' distributions differed between survivors and non-survivors.
    * **Missing Values Analysis:**
        * Identified columns with missing values ('Age', 'Cabin', 'Embarked') and their counts/percentages. 'Cabin' has the most missing data (~77.10%), followed by 'Age' (~19.87%), and 'Embarked' (~0.22%).
        * Visualized missing data patterns using a heatmap and a bar chart of missing percentages.
    * **Bivariate Analysis:**
        * Analyzed survival rates broken down by both 'Pclass' and 'Sex' using a grouped bar chart, highlighting interactions (e.g., females in Pclass 1 had a ~96.81% survival rate, while males in Pclass 3 had ~13.54%).
    * **Correlation Analysis:**
        * A correlation heatmap was generated for numerical features ('Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare') to show linear relationships.Notable correlations include 'Fare' and 'Survived' (0.26), and 'Pclass' and 'Survived' (-0.34).
    * **Outlier Detection:**
        * Box plots were used to visualize potential outliers in 'Age' and 'Fare'.
        * Age distribution by 'Pclass' was also visualized using box plots.

3.  **Data Preprocessing (on `train.csv`):**
    * **Handling Missing Values:**
        * 'Age': Missing values were imputed using the median age of the entire dataset.
        * 'Cabin': The 'Cabin' column was dropped due to a high number of missing values.
        * 'Embarked': Missing values were imputed using the mode ('S').
        * A heatmap confirmed that no missing values remained in the processed training data.
    * **Feature Engineering (Encoding Categoricals):**
        * 'Sex': Mapped to a numerical representation ('Sex\_encoded': male to 0, female to 1).
        * 'Embarked': Mapped to numerical representation ('Embarked\_encoded') based on the order of unique values encountered.
    * **Dropping Columns:**
        * Original 'Name', 'Ticket', 'PassengerId', 'Embarked', and 'Sex' columns were dropped after processing/encoding.
    * The final features used for training were 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex\_encoded', and 'Embarked\_encoded'. The target variable is 'Survived'.
4.  **Model Building & Evaluation (on the split from `train.csv`):**
    * A function `classify(model)` was defined to:
        * Split the data (`X`, `y` derived from `train.csv`) into training (75%) and testing (25%) sets.
        * Fit the provided model on this training split.
        * Print the accuracy score on this local test split.
        * Perform 5-fold cross-validation on the entire `X`, `y` and print the mean CV score.
    * The following classification models were evaluated using this function:
        * DecisionTreeClassifier: Accuracy ~0.767, CV Score ~0.777.
        * LGBMClassifier: Accuracy ~0.830, CV Score ~0.827. (Many "No further splits with positive gain" warnings were observed during its CV).
        * XGBClassifier: Accuracy ~0.812, CV Score ~0.824.
        * RandomForestClassifier: Accuracy ~0.812, CV Score ~0.807.
        * ExtraTreesClassifier: Accuracy ~0.803, CV Score ~0.794.
        * LogisticRegression: Accuracy ~0.807, CV Score ~0.793.

5.  **Prediction on Kaggle Test Set (`test.csv`):**
    * The `test.csv` file was loaded.
    * Columns 'PassengerId', 'Name', 'Cabin', 'Ticket' were dropped.
    * **Data Preprocessing on Test Set:**
        * 'Age': Missing values imputed using the mean of the 'Age' column *from the test set itself*.
        * 'Fare': Missing values imputed using the mean of the 'Fare' column *from the test set itself*.
        * 'Sex' and 'Embarked': Encoded using `LabelEncoder` by calling `fit_transform` *on the test set columns*.
        * Encoded columns were renamed to match training set column names (e.g., 'Sex\_encoded', 'Embarked\_encoded').
        * The order of columns in the test set was aligned with the training set.
    * An `XGBClassifier` model (which was fit on the full `train.csv` data after the `classify` function evaluations) was used to make predictions on the preprocessed `test.csv` data.

## Libraries Used
* pandas
* matplotlib
* seaborn
* numpy
* scikit-learn (for `train_test_split`, `cross_val_score`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `LogisticRegression`, `LabelEncoder`)
* LightGBM (`LGBMClassifier`)
* XGBoost (`XGBClassifier`)

## Observations from the Notebook Analysis

* **Data Leakage in Test Set Preprocessing:** A critical observation is the preprocessing of the Kaggle `test.csv` data.
    * Missing 'Age' and 'Fare' values were imputed using the mean calculated *from the test set itself* (`X_test["Age"].mean()`, `X_test["Fare"].mean()`).
    * `LabelEncoder` was applied using `fit_transform` directly on the test set columns for 'Sex' and 'Embarked'.
    * **This approach leads to data leakage.** For correct methodology, imputation statistics (mean, median, mode) and encoder mappings should be learned *only* from the training data (`train.csv`) and then applied to transform the test set.
* **LightGBM Warnings:** The LightGBM model produced numerous "No further splits with positive gain" warnings during cross-validation, suggesting that its default parameters might not be optimal for the dataset size or that some splits did not improve the objective function.
* **`classify` Function Data Splitting:** The `classify` function re-splits the data into train/test for every model evaluation. While fine for quick comparisons, for a more robust final model selection pipeline, it's often better to have a fixed training/validation split or rely solely on cross-validation scores for hyperparameter tuning on the training set.
* **Final Model for Prediction:** An XGBoost model was trained on the full `train.csv` dataset (after initial preprocessing of `df`) before making predictions on the processed `test.csv`.

## Potential Improvements

* **Correct Test Set Preprocessing:** Modify the preprocessing of the Kaggle `test.csv` to use statistics and encoder mappings learned *only* from the `train.csv` data to prevent data leakage.
* **Hyperparameter Tuning:** For models like LightGBM, XGBoost, and RandomForestClassifier, perform systematic hyperparameter tuning (e.g., using `GridSearchCV` or `RandomizedSearchCV` with appropriate cross-validation strategies) to potentially improve their CV scores and generalization. This could also help address the LightGBM warnings by finding better parameters.
* **Advanced Feature Engineering:** Explore creating more sophisticated features (e.g., from 'Name' like 'Title', 'FamilySize' from 'SibSp' and 'Parch', or binning 'Age' and 'Fare').
* **Pipelines:** Implement scikit-learn Pipelines to streamline preprocessing and modeling, ensuring consistency between training and testing phases and preventing data leakage.
* **Stratified Cross-Validation:** Ensure cross-validation uses stratification, especially if there's a class imbalance in the target variable (which there is, with fewer survivors). The `classify` function uses `cv=5` without explicit stratification, though scikit-learn's `cross_val_score` might do it by default for classifiers.
* **Ensemble Methods:** Further explore stacking or blending of different models that showed good performance.
