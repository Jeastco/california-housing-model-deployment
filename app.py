import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the trained model
model = joblib.load("california_housing_knn_model.pkl")

# App title
st.title("🏠 California Housing Price Predictor")

st.write(
    "Enter the housing information below to estimate the "
    "median house value using the trained KNN regression model."
)

# User inputs
med_inc = st.number_input(
    "Median Income",
    min_value=0.0,
    value=3.5
)

house_age = st.number_input(
    "Median House Age",
    min_value=0.0,
    value=30.0
)

ave_rooms = st.number_input(
    "Average Rooms per Household",
    min_value=0.0,
    value=5.0
)

ave_bedrms = st.number_input(
    "Average Bedrooms per Household",
    min_value=0.0,
    value=1.0
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000.0
)

ave_occup = st.number_input(
    "Average Household Members",
    min_value=0.0,
    value=3.0
)

latitude = st.number_input(
    "Latitude",
    value=34.0
)

longitude = st.number_input(
    "Longitude",
    value=-118.0
)

# Prediction button
if st.button("Predict House Value"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Target is measured in units of $100,000
    estimated_price = prediction * 100000

    st.success(
        f"Estimated Median House Value: ${estimated_price:,.2f}"
    )

    st.write(f"Raw model prediction: {prediction:.4f}")
