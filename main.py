import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helper import make_prediction

st.set_page_config(page_title="ML Prediction App")

st.title("Logistic Regression Prediction App")
st.write("This app uses a full sklearn Pipeline (preprocessing + model)")

# Example inputs (adjust based on your dataset)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
category = st.selectbox("Category", ["A", "B", "C"])

if st.button("Predict"):

    input_data = {
        "feature1": feature1,
        "feature2": feature2,
        "category": category
    }

    result = make_prediction(input_data)

    st.success(f"Prediction: {result}")