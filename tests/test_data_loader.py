# import pytest
import pandas as pd
from src.data_loader import load_data, preprocess_data, load_sample_data


def test_load_data():
    """Проверяет, что данные загружаются без ошибок."""
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert not X.empty, "Данные не загружены (X пустой)"
    assert not y.empty, "Целевая переменная не загружена (y пустой)"


def test_preprocess_data():
    """Проверяет, что предобработка данных работает."""
    X, y = load_data()
    X_processed, y_processed = preprocess_data(X, y)

    # Проверка, что нет пропущенных значений
    assert X_processed.isna().sum().sum() == 0, "Есть пропущенные значения в X"
    assert y_processed.isna().sum().sum() == 0, "Есть пропущенные значения в y"


def test_load_sample_data():
    """Проверяет, что данные делятся на train/test."""
    X_train, X_test, y_train, y_test = load_sample_data(test_size=0.2)

    # Проверка размеров выборок
    assert len(X_train) > 0, "Train выборка пустая"
    assert len(X_test) > 0, "Test выборка пустая"
    assert len(X_train) + len(X_test) == len(load_data()
                                             [0]), "Неверное разделение данных"
