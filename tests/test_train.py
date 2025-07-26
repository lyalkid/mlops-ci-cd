import os
from pathlib import Path


def test_model_training():
    """Проверяет, что модель обучается и сохраняется."""
    # Используем тот же путь, что и в train.py
    MODEL_DIR = Path(__file__).parent.parent / "src" / "models"
    MODEL_PATH = MODEL_DIR / "model.joblib"

    print(f"\n🔍 Проверяю путь к модели: {MODEL_PATH}")
    print(f"   Абсолютный путь: {MODEL_PATH.absolute()}")

    # Удаляем старую модель, если есть
    if MODEL_PATH.exists():
        print("🗑 Удаляю старую модель")
        os.remove(MODEL_PATH)

    # Запускаем тренировку
    from src.train import model, X_test, y_test, MODEL_PATH as TRAIN_MODEL_PATH

    print(f"🔄 Путь в train.py: {TRAIN_MODEL_PATH}")

    # Проверяем, что модель обучена
    assert model is not None, "Модель не обучена"

    # Проверяем существование файла
    print("🔎 Проверяю существование модели...")
    print(f"   Путь: {TRAIN_MODEL_PATH}")
    print(f"   Существует: {os.path.exists(TRAIN_MODEL_PATH)}")

    assert os.path.exists(
        TRAIN_MODEL_PATH), f"Модель не найдена по пути: {TRAIN_MODEL_PATH}"

    # Проверяем предсказания
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test), "Количество предсказаний не совпадает"
