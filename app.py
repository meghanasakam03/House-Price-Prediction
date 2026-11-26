
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

# Load trained model
model = joblib.load("house_price_pipeline.pkl")

# Application title
st.title("🏠 House Price Prediction")
st.write("Enter the property details to predict the house price.")

# Numerical inputs
area = st.number_input("Area", 501, 4999, 2500)

bedrooms = st.slider("Bedrooms", 1, 5, 3)

bathrooms = st.slider("Bathrooms", 1, 4, 2)

floors = st.slider("Floors", 1, 3, 1)

year_built = st.number_input("Year Built", 1900, 2023, 2000)

# Categorical inputs
location = st.selectbox(
    "Location", ["Downtown", "Suburban", "Urban", "Rural"]
)

condition = st.selectbox(
    "Condition", ["Excellent", "Good", "Fair", "Poor"]
)

garage = st.radio("Garage", ["No", "Yes"])

# Convert user inputs into a dataframe
input_data = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms],
    "Floors": [floors],
    "YearBuilt": [year_built],
    "Location": [location],
    "Condition": [condition],
    "Garage": [garage]
})

# Prediction
if st.button("Predict House Price"):
    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: {prediction[0]:,.2f}")
