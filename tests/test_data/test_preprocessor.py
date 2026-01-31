"""
Testes para o pré-processador de dados.
"""
import numpy as np
import pandas as pd

from src.data.preprocessor import DataPreprocessor


def test_fit_transform_normalizes():
    """Dados devem ser normalizados entre 0 e 1."""
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({"Close": [100, 150, 200, 250, 300]})

    scaled = preprocessor.fit_transform(df)

    assert scaled.min() >= 0
    assert scaled.max() <= 1


def test_inverse_transform_restores():
    """inverse_transform deve restaurar valores originais."""
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({"Close": [100, 150, 200, 250, 300]})

    scaled = preprocessor.fit_transform(df)
    restored = preprocessor.inverse_transform(scaled)

    np.testing.assert_array_almost_equal(restored, df["Close"].values)


def test_create_sequences_shape():
    """Sequências devem ter shape correto."""
    preprocessor = DataPreprocessor()
    data = np.random.rand(100, 1)
    sequence_length = 10

    X, y = preprocessor.create_sequences(data, sequence_length)

    assert X.shape == (90, 10, 1)  # 100 - 10 = 90 sequências
    assert y.shape == (90, 1)


def test_split_data_ratios():
    """Split deve respeitar proporções."""
    preprocessor = DataPreprocessor()
    X = np.random.rand(100, 10, 1)
    y = np.random.rand(100, 1)

    (X_train, _), (X_val, _), (X_test, _) = preprocessor.split_data(X, y)

    assert len(X_train) == 80
    assert len(X_val) == 10
    assert len(X_test) == 10
