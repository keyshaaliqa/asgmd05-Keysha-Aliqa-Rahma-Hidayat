import joblib
import pandas as pd
from pathlib import Path


def load_pipeline(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "artifacts" / "pipeline.pkl"
    path = Path(path)

    if not path.exists():
        # Auto-train and save a default pipeline if missing.
        train_pipeline(path)

    model = joblib.load(path)
    return model


def train_pipeline(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "artifacts" / "pipeline.pkl"
    path = Path(path)

    # Ensure artifacts directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Generate synthetic training data (features match app inputs)
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [2.1, 3.4, 1.5, 2.7, 3.0, 0.5, 1.0, 4.0, 1.2, 2.8],
        "feature3": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
        "target": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    })

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression

    numeric_features = ["feature1", "feature2"]
    categorical_features = ["feature3"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
        ]
    )

    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline.fit(X, y)

    joblib.dump(pipeline, path)
    return pipeline


def make_prediction(input_dict, model):

    df = pd.DataFrame([input_dict])

    prediction = model.predict(df)

    return prediction[0]