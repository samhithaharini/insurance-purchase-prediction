import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("insurance_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Insurance Prediction", layout="centered")

st.title("🛡️ Insurance Purchase Prediction")
st.write("Enter user details to predict whether they will buy insurance.")

age = st.number_input("Enter Age", min_value=18, max_value=100, step=1)

affordibility = st.selectbox(
    "Can afford insurance?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

if st.button("Predict"):

    input_data = np.array([[age, affordibility]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Will BUY Insurance (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Will NOT Buy Insurance (Confidence: {1 - probability:.2f})")