import streamlit as st
import pandas as pd
import joblib

# Load models
logistic_model = joblib.load('../Models/logistic_model.pkl')
knn_model = joblib.load('../Models/knn_model.pkl')

# Input Form
st.title("Loan Application Predictor")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input("Dependents", min_value=0, max_value=10)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
employment = st.selectbox("Self-Employed", ["Yes", "No"])
income = st.number_input("Applicant Income", min_value=0)
co_income = st.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
term = st.number_input("Loan Term (in months)", min_value=0)
credit = st.selectbox("Credit History", [1, 0])
area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert Property_Area to One-Hot Encoding
property_area_dict = {
    "Urban": [0, 0, 1],
    "Semiurban": [0, 1, 0],
    "Rural": [1, 0, 0]
}

property_area_encoded = property_area_dict[area]

# Convert Dependents to One-Hot Encoding
dependents_dict = {
   0: [1, 0, 0, 0],
   1: [0, 1, 0, 0],
   2: [0, 0, 1, 0],
   3: [0, 0, 0, 1]
}

if dependents >= 3:
    dependents = 3

dependents_encoded = dependents_dict[dependents]




# Submit and predict
if st.button("Predict"):
    user_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Married': [1 if married == "Yes" else 0],
        'Dependents_0': [dependents_encoded[0]], 
        'Dependents_1': [dependents_encoded[1]], 
        'Dependents_2': [dependents_encoded[2]], 
        'Dependents_3+': [dependents_encoded[3]], 
        'Education': [1 if education == "Graduate" else 0],
        'Self_Employed': [1 if employment == "Yes" else 0],
        'ApplicantIncome': [income],
        'CoapplicantIncome': [co_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [term],
        'Credit_History': [credit],
        'Property_Area_Rural': [property_area_encoded[0]],
        'Property_Area_Semiurban': [property_area_encoded[1]],
        'Property_Area_Urban': [property_area_encoded[2]]
    })

    logistic_pred = logistic_model.predict(user_data)
    knn_pred = knn_model.predict(user_data)

    st.write("Logistic Regression Prediction:", "Approved" if logistic_pred[0] == 1 else "Rejected")
    st.write("KNN Prediction:", "Approved" if knn_pred[0] == 1 else "Rejected")

import matplotlib.pyplot as plt

accuracies = {'Logistic Regression': 85, 'KNN': 82, 'SVM': 88}
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.show()

