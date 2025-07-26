import joblib
from datetime import datetime
from src.data_loader import get_sample_features

model = joblib.load("models/model.joblib")
X_sample = get_sample_features(n=10)
preds = model.predict(X_sample)

report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Wine Quality Report</title>
</head>
<body>
    <h1>Предсказание качества вина</h1>
    <p>Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    {X_sample.assign(Predicted_Quality=preds).to_html()}
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(report)
print("✅ Отчёт сохранён в report.html")
