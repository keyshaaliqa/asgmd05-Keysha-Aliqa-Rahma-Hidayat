import joblib
import pandas as pd


def load_pipeline():
    return joblib.load("models/pipeline.pkl")


def make_prediction(input_data: dict):

    df = pd.DataFrame([input_data])

    model = load_pipeline()

    prediction = model.predict(df)

    return prediction[0]