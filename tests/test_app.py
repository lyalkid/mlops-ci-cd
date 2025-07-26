# import pytest
# import os
from fastapi.testclient import TestClient
from src.app import app
# from pathlib import Path


client = TestClient(app)


# def test_predict_endpoint():
#     # Убедимся, что модель существует
#     model_path = Path("src/models/model.joblib")
#     if not model_path.exists():
#         os.system("make train")

#     test_data = {
#         "KinEng": 1.0,
#         "PotEng": 2.0,
#         "Volume": 3.0,
#         "Step": 1
#     }
#     response = client.post("/predict", json=test_data)
#     assert response.status_code == 500
#     assert "Кинетическая энергия" in response.json()

def test_invalid_data():
    invalid_data = {
        "fixed_acidity": "not_a_number",
        "volatile_acidity": 0.7
    }

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
