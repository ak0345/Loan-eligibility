---

# Loan Eligibility Prediction

This project involves building a machine learning model to predict loan eligibility using a dataset containing various features about applicants. The dataset is processed and analyzed to develop predictive models and assess their performance.

## Project Overview

The goal of this project is to predict loan eligibility based on applicant data. The project utilizes different machine learning algorithms and preprocessing techniques to achieve accurate predictions.

## Dataset

The dataset used for this project is `loan-train.csv`, which includes the following columns:

- `Loan_ID` (dropped)
- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- `Loan_Status` (target variable)

## Data Preprocessing

1. **Categorical Encoding:** 
   - Categorical features with <=5 unique values are encoded using `OrdinalEncoder`.

2. **Handling Missing Values:**
   - Categorical NaNs are filled forward.
   - Numeric NaNs are interpolated.

3. **Normalization:**
   - Numeric features are normalized.

4. **Feature Selection:**
   - Features with potential bias or low variability are removed.

## Models Used

1. **Logistic Regression (`LogisticRegression`):**
   - Accuracy: ~79.67%

2. **Decision Tree Classifier (`DecisionTreeClassifier`):**
   - Accuracy: ~71.54%

3. **Random Forest Classifier (`RandomForestClassifier`):**
   - Accuracy: ~79.67%

## Code

The main script for this project is `loan_eligibility_prediction.py`. The code includes:

- Data loading and preprocessing.
- Encoding categorical features and handling missing values.
- Normalizing numerical features.
- Training and evaluating machine learning models.
- Reporting model accuracies.
