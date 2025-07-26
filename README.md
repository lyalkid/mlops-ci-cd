# mlops-ci-cd

## Установка зависимостей
```
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Обучение модели
```
python3 src/train.py
```
Это:
- загрузит данные по ссылке
- обучит модель
- сохранит её в src/models/model.joblib

## Инференс (предсказание + HTML-отчёт)
```
python3 src/inference.py
```
Это:
- сгенерирует предсказания для первых 5 записей
- сохранит CSV и report.html в src/predictions/