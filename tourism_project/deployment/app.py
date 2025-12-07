
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import os

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="airplane",
    layout="wide"
)

st.title("Wellness Tourism Package Prediction")
st.markdown("### Predict if a customer will purchase the Wellness Tourism Package")

with st.sidebar:
    st.header("About")
    st.write("""
    This app predicts whether a customer will purchase the Wellness Tourism Package
    based on their profile and interaction data.
    """)

    st.header("Instructions")
    st.write("""
    1. Fill in the customer details
    2. Click Predict to see the prediction
    3. View purchase probability
    """)

@st.cache_resource
def load_model_and_preprocessors():
    HF_USERNAME = "kkkotha"
    MODEL_REPO = f"{HF_USERNAME}/tourism-model"

    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl")
    encoders_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoders.pkl")
    scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename="scaler.pkl")

    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)

    return model, label_encoders, scaler

model, label_encoders, scaler = load_model_and_preprocessors()

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])

with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=20000)

with col3:
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
    passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.header("Sales Interaction Details")

col4, col5 = st.columns(2)

with col4:
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
    num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)

with col5:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

if st.button("Predict Purchase Probability", type="primary"):
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [num_persons],
        'NumberOfFollowups': [num_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [preferred_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [num_trips],
        'Passport': [passport],
        'PitchSatisfactionScore': [pitch_satisfaction],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [num_children],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income]
    })

    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
    numerical_cols = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']

    for col in categorical_cols:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]

    st.header("Prediction Results")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        if prediction == 1:
            st.success("Customer is LIKELY to purchase the package!")
        else:
            st.warning("Customer is UNLIKELY to purchase the package")

    with col_r2:
        st.metric("Purchase Probability", f"{prediction_proba[1]:.1%}")

    st.subheader("Probability Breakdown")
    prob_df = pd.DataFrame({
        'Outcome': ['Will NOT Purchase', 'Will Purchase'],
        'Probability': [f"{prediction_proba[0]:.1%}", f"{prediction_proba[1]:.1%}"]
    })
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

    st.subheader("Recommendation")
    if prediction == 1 and prediction_proba[1] > 0.7:
        st.info("High Priority Lead: Contact this customer immediately")
    elif prediction == 1:
        st.info("Potential Lead: Schedule a follow-up call")
    else:
        st.info("Low Priority: Consider alternative packages")

st.markdown("---")
st.markdown("Powered by Machine Learning")
