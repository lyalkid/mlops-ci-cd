from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import logging
import subprocess

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicalFeatures(BaseModel):
    KinEng: float
    PotEng: float
    Volume: float
    Step: int


app = FastAPI(title="Physical Predictor")

MODEL_PATH = Path("src/models/model.joblib")


def train_model_if_needed():
    """Обучает модель, если она отсутствует"""
    if not MODEL_PATH.exists():
        logger.info("Модель не найдена, запускаю обучение...")
        try:
            subprocess.run(["make", "train"], check=True)
            logger.info("Модель успешно обучена")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise


# Загружаем или обучаем модель при старте
try:
    train_model_if_needed()
    model = joblib.load(MODEL_PATH)
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Не удалось загрузить модель: {e}")
    raise


@app.post("/predict")
def predict(data: PhysicalFeatures):
    try:
        df = pd.DataFrame([data.model_dump()])
        X = df[['PotEng', 'Volume']]  # Выбираем только нужные фичи
        prediction = model.predict(X)[0]
        return {"Кинетическая энергия": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report")
def get_report():
    return FileResponse("predictions/report.html", media_type="text/html")
