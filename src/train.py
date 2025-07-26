import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
# Изменено с относительного на абсолютный импорт
from src.data_loader import load_sample_data


# Настройка путей
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
os.makedirs(MODEL_DIR, exist_ok=True)

# Загрузка данных
X_train, X_test, y_train, y_test = load_sample_data()

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'✅ MSE: {mse:.4f}')
print(f'✅ R²: {r2:.4f}')

# Сохранение модели
joblib.dump(model, MODEL_PATH)
print(f'✅ Model saved to {MODEL_PATH}')
