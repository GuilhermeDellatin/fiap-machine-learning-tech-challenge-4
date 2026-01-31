"""
Pré-processamento de dados para LSTM.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Prepara dados para treinamento/inferência do LSTM."""

    def __init__(self) -> None:
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted = False

    def fit_transform(
        self, data: pd.DataFrame, feature_col: str = "Close"
    ) -> np.ndarray:
        """
        Fit do scaler e transforma dados.

        Args:
            data: DataFrame com dados
            feature_col: Coluna a usar (default: 'Close')

        Returns:
            Array normalizado (n_samples, 1)
        """
        values = data[feature_col].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)
        self._is_fitted = True
        logger.info(f"Scaler fitted. Range: [{values.min():.2f}, {values.max():.2f}]")
        return scaled

    def transform(
        self, data: pd.DataFrame, feature_col: str = "Close"
    ) -> np.ndarray:
        """Transforma dados usando scaler já fitted."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        values = data[feature_col].values.reshape(-1, 1)
        return self.scaler.transform(values)

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Reverte normalização."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted.")
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()

    def create_sequences(
        self, data: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências para LSTM usando sliding window.

        Args:
            data: Array normalizado (n_samples, 1)
            sequence_length: Tamanho da janela

        Returns:
            X: (n_sequences, sequence_length, 1)
            y: (n_sequences, 1)
        """
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i : (i + sequence_length)])
            y.append(data[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Sequences created: X={X.shape}, y={y.shape}")
        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple:
        """
        Split em train/val/test.

        Args:
            X: Features
            y: Targets
            train_ratio: Proporção treino (default: 0.8)
            val_ratio: Proporção validação (default: 0.1)

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def prepare_for_inference(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        feature_col: str = "Close",
    ) -> np.ndarray:
        """
        Prepara dados para inferência.

        Args:
            data: DataFrame com últimos N dias
            sequence_length: Tamanho esperado da sequência

        Returns:
            Array pronto para modelo (1, sequence_length, 1)
        """
        if len(data) < sequence_length:
            raise ValueError(f"Need {sequence_length} rows, got {len(data)}")

        recent = data.tail(sequence_length)
        scaled = self.transform(recent, feature_col)

        return scaled.reshape(1, sequence_length, 1)

    def save_scaler(self, path: str) -> None:
        """Salva scaler em arquivo."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted.")
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str) -> None:
        """Carrega scaler de arquivo."""
        self.scaler = joblib.load(path)
        self._is_fitted = True
        logger.info(f"Scaler loaded from {path}")
