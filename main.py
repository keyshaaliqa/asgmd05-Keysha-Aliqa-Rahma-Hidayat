import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds No. 2 root

from utils.helper import load_pipeline, make_prediction

st.title("ML Prediction App")

st.write("Logistic Regression Prediction using Pipeline")

# load pipeline
model = load_pipeline()

# example inputs
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.selectbox("Category", ["A", "B", "C"])

if st.button("Predict"):

    input_data = {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3
    }

    prediction = make_prediction(input_data, model)

    st.success(f"Prediction: {prediction}")