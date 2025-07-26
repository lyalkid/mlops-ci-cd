# ==== ПЕРЕМЕННЫЕ ====
PYTHON := python3
APP := src.app:app
PYTHONPATH := PYTHONPATH=.

# ==== УСТАНОВКА ЗАВИСИМОСТЕЙ ====
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ==== ОБУЧЕНИЕ МОДЕЛИ ====
train:
	$(PYTHONPATH) $(PYTHON) src/train.py

# ==== ИНФЕРЕНС И ОТЧЁТ ====
infer:
	$(PYTHONPATH) $(PYTHON) src/inference.py

# ==== ЗАПУСК FASTAPI ====
serve:
	uvicorn $(APP) --reload

# ==== ЗАПУСК ТЕСТОВ ====
test:
	$(PYTHONPATH) pytest -s tests/

# ==== ПРОВЕРКА СТИЛЯ ====
lint:
	flake8 src tests

# ==== ОЧИСТКА РЕЗУЛЬТАТОВ ====
clean:
	rm -f src/models/*.joblib
	rm -f src/predictions/*.csv
	rm -f src/predictions/*.html
