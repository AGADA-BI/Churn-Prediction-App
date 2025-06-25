# Gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# Scalar is exported as a pickle file
# Model is exported as model.pkl
# Order is Age	Gender	Tenure	MonthlyCharges

import streamlit as st
import joblib
import numpy as np


scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


st.title("Customer Churn Prediction App")

st.divider()

st.write("Please enter the following details:")

st.divider()

age = st.number_input("Age", min_value=18, max_value=100, value=30)

tenure = st.number_input("Enter tenure", min_value=0, max_value=140, value=10)

monthly_charges = st.number_input("Enter monthly charge", min_value=0.0, max_value=200.0, value=50.0)    

gender = st.selectbox("Enter the Gender", ["Male", "Female"])


st.divider()

predict_button = st.button("Predict")

if predict_button:

    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthly_charges]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"Predicted: {predicted}")


else: 
    st.write("Please fill all the fields to get a prediction then click the predict button.")



