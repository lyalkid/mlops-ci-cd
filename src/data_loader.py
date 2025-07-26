import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def load_data(path: str = None):
    """Загружает данные из указанного пути или использует встроенный датасет"""
    if path is None:
        wine_quality = fetch_ucirepo(id=186)
        return wine_quality.data.features, wine_quality.data.targets
    return pd.read_csv(path, sep=r'\s+')


def preprocess_data(df: pd.DataFrame, target: pd.DataFrame):
    """Предобработка данных"""
    df = df.copy()
    df.dropna(inplace=True)
    target = target.loc[df.index]
    return df, target


def load_sample_data(test_size=0.2, random_state=42):
    """Загружает и разделяет данные на train/test"""
    X, y = load_data()
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


def get_sample_features(n: int = 5):
    """Возвращает n примеров признаков"""
    X, _ = load_data()
    return X.head(n)
