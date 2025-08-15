import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained Keras model
model = tf.keras.models.load_model("churn_model.h5")

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    geo_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("Customer Churn Prediction")

# Inputs
geography = st.selectbox("Geography", geo_encoder.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
estimated_salary = st.number_input("Estimated Salary", min_value=1000.0, max_value=200000.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Encode gender
gender_encoded = gender_encoder.transform([gender])[0]

# One-hot encode geography
geo_encoded = geo_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=geo_encoder.get_feature_names_out(["Geography"])
)

# Create input DataFrame (EXCLUDE 'Geography' as a column)
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender_encoded": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# Concatenate one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# Ensure the order of features matches the model's training
input_data = input_data[scaler.feature_names_in_]

# Scale
scaled_input = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_input)[0][0]

# Output
if prediction > 0.5:
    st.error(f"The customer is likely to churn. (Probability: {prediction:.2f})")
else:
    st.success(f"The customer is not likely to churn. (Probability: {prediction:.2f})")
