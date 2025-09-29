import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("churn_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered",page_icon="🔮")
st.title("📊 Customer Churn Prediction App")
st.write("This demo predicts whether a customer will churn based on their profile.")

# --- Input form ---
st.sidebar.header("Enter Customer Details")

# Numeric inputs
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, value=600.0)

# Categorical inputs
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method", 
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
)

# --- One-hot encoding ---
input_dict = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "gender_Male": 1 if gender == "Male" else 0,
    "Partner_Yes": 1 if Partner == "Yes" else 0,
    "Dependents_Yes": 1 if Dependents == "Yes" else 0,
    "PhoneService_Yes": 1 if PhoneService == "Yes" else 0,
    "MultipleLines_No phone service": 1 if MultipleLines == "No phone service" else 0,
    "MultipleLines_Yes": 1 if MultipleLines == "Yes" else 0,
    "InternetService_Fiber optic": 1 if InternetService == "Fiber optic" else 0,
    "InternetService_No": 1 if InternetService == "No" else 0,
    "OnlineSecurity_No internet service": 1 if OnlineSecurity == "No internet service" else 0,
    "OnlineSecurity_Yes": 1 if OnlineSecurity == "Yes" else 0,
    "OnlineBackup_No internet service": 1 if OnlineBackup == "No internet service" else 0,
    "OnlineBackup_Yes": 1 if OnlineBackup == "Yes" else 0,
    "DeviceProtection_No internet service": 1 if DeviceProtection == "No internet service" else 0,
    "DeviceProtection_Yes": 1 if DeviceProtection == "Yes" else 0,
    "TechSupport_No internet service": 1 if TechSupport == "No internet service" else 0,
    "TechSupport_Yes": 1 if TechSupport == "Yes" else 0,
    "StreamingTV_No internet service": 1 if StreamingTV == "No internet service" else 0,
    "StreamingTV_Yes": 1 if StreamingTV == "Yes" else 0,
    "StreamingMovies_No internet service": 1 if StreamingMovies == "No internet service" else 0,
    "StreamingMovies_Yes": 1 if StreamingMovies == "Yes" else 0,
    "Contract_One year": 1 if Contract == "One year" else 0,
    "Contract_Two year": 1 if Contract == "Two year" else 0,
    "PaperlessBilling_Yes": 1 if PaperlessBilling == "Yes" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if PaymentMethod == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if PaymentMethod == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if PaymentMethod == "Mailed check" else 0,
}

# Ensure correct feature order
X = pd.DataFrame([input_dict], columns=feature_names)


if st.button("Predict Churn"):
    # Prediction 
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    prob_churn = probs[list(model.classes_).index("Yes")]

    pred_label = "Yes" if pred == "Yes" or pred == 1 else "No"

    if pred_label == "Yes":
        st.error("⚠️ Customer will churn")
        st.error(f"Churn probability: {prob_churn:.2%}")
    else:
        st.success("✅ Customer will not churn")
        st.success(f"Churn probability: {prob_churn:.2%}")
