import os
import joblib
import pandas as pd


def test_model_prediction():
    model_path = 'src/models/model.joblib'
    assert os.path.exists(model_path), "Model not found. Run train.py first."

    model = joblib.load(model_path)

    # Создаем пример с правильными фичами
    sample = pd.DataFrame({
        'fixed_acidity': [7.4],
        'volatile_acidity': [0.7],
        'citric_acid': [0.0],
        'residual_sugar': [1.9],
        'chlorides': [0.076],
        'free_sulfur_dioxide': [11.0],
        'total_sulfur_dioxide': [34.0],
        'density': [0.9978],
        'pH': [3.51],
        'sulphates': [0.56],
        'alcohol': [9.4]
    })

    pred = model.predict(sample)[0]
    assert isinstance(pred, float)
