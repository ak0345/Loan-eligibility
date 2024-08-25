import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import normalize

df = pd.read_csv("loan_eligibility_data/loan-train.csv").drop("Loan_ID", axis = 1)
print(df.info())

#Encoding categorical features
encoder = OrdinalEncoder()
for i in df.columns:
    print(i)
    if df[i].nunique() <= 5:
        df[i] = encoder.fit_transform(df[i].to_numpy().reshape(-1, 1))
        #Getting rid of categorical NaNs
        df[i] = df[i].fillna(method="ffill")
    else:
        df[i] = df[i].interpolate(method='polynomial', order=2)
df = df.dropna()
print(df.info())

print(df.corr())


#counting Unique values
"""
print(df.nunique())

Gender                 2
Married                2
Dependents             4
Education              2
Self_Employed          2
ApplicantIncome      504
CoapplicantIncome    287
LoanAmount           224
Loan_Amount_Term      24
Credit_History         2
Property_Area          3
Loan_Status            2
"""

categorical = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]
non_categorical = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

#Normalising non_categorical features
for i in non_categorical:
    df[i] = normalize(df[i].to_numpy().reshape(-1, 1), axis = 0)

"""for i in categorical:
    df[i].value_counts().plot(kind="bar", rot=0)
    plt.title(i)
    plt.show()"""

#Gender, Married, Dependants, Education, Self_Employed, Credit History,   categories not being used due to potential bias and lack of variability
#Keeping Credit History as even tho potential bias, it has decent correlation with Loan Status

remove_category_2 = ["Gender", "Married", "Dependents", "Education", "Self_Employed"]

X = df.drop(["Loan_Status"]+remove_category_2, axis = 1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model_lr = LR()
model_lr.fit(X_train, y_train)
Y_hat_lr = model_lr.predict(X_test)

print(accuracy_score(y_test, Y_hat_lr))

model_dtc = DTC(criterion="entropy", random_state=42)
model_dtc.fit(X_train, y_train)
Y_hat_dtc = model_dtc.predict(X_test)

print(accuracy_score(y_test, Y_hat_dtc))

model_rfc = RFC(random_state=42)
model_rfc.fit(X_train, y_train)

Y_hat_rfc = model_rfc.predict(X_test)

print(accuracy_score(y_test, Y_hat_rfc))