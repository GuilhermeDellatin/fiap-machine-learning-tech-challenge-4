"""
Inferência em produção.
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from src.models.lstm_model import LSTMPredictor
from src.data.preprocessor import DataPreprocessor
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(Exception):
    """Modelo não carregado."""
    pass


class StockPredictor:
    """
    Inferência de preços de ações.
    Thread-safe para uso em API.
    """

    def __init__(
        self, model_path: str = None, scaler_path: str = None
    ) -> None:
        """
        Args:
            model_path: Caminho do arquivo .pt
            scaler_path: Caminho do arquivo .joblib
        """
        self.model: Optional[LSTMPredictor] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_ticker: Optional[str] = None
        self.model_version: Optional[str] = None

        if model_path and scaler_path:
            self.reload_model(model_path, scaler_path)

    def load_from_registry(self, db, ticker: str) -> bool:
        """
        Carrega modelo ativo do ModelRegistry.

        Args:
            db: Sessão SQLAlchemy
            ticker: Código da ação

        Returns:
            True se carregou, False se não há modelo
        """
        from src.database.repository import ModelRegistryRepository

        repo = ModelRegistryRepository()
        model_info = repo.get_active_model(db, ticker)

        if model_info is None:
            logger.warning(f"No active model for {ticker}")
            return False

        self.reload_model(model_info.model_path, model_info.scaler_path)
        self.current_ticker = ticker
        self.model_version = model_info.version_id

        return True

    def reload_model(self, model_path: str, scaler_path: str) -> None:
        """
        Carrega ou recarrega modelo (hot reload).

        Args:
            model_path: Caminho do .pt
            scaler_path: Caminho do .joblib
        """
        # Carregar scaler
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_scaler(scaler_path)

        # Carregar modelo
        self.model = LSTMPredictor(
            input_size=settings.INPUT_SIZE,
            hidden_size=settings.HIDDEN_SIZE,
            num_layers=settings.NUM_LAYERS,
            dropout=settings.DROPOUT,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded: {model_path}")

    def predict(
        self, historical_data: pd.DataFrame, days_ahead: int = 1
    ) -> List[float]:
        """
        Prediz preços futuros.

        Args:
            historical_data: DataFrame com coluna 'Close'
            days_ahead: Dias para prever

        Returns:
            Lista de preços previstos
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded. Call reload_model first.")

        predictions = []

        # Preparar dados
        sequence_length = settings.SEQUENCE_LENGTH
        data = historical_data.copy()

        for _ in range(days_ahead):
            # Preparar input
            input_tensor = self.preprocessor.prepare_for_inference(
                data, sequence_length
            )
            input_tensor = torch.FloatTensor(input_tensor).to(self.device)

            # Inferência
            with torch.no_grad():
                scaled_pred = self.model(input_tensor)

            # Inverter normalização
            pred_value = self.preprocessor.inverse_transform(
                scaled_pred.cpu().numpy()
            )[0]
            predictions.append(float(pred_value))

            # Adicionar predição aos dados para próxima iteração
            new_row = data.iloc[-1:].copy()
            new_row["Close"] = pred_value
            data = pd.concat([data, new_row], ignore_index=True)

        return predictions

    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inferência direta (dados já preprocessados).

        Args:
            input_tensor: Tensor (batch, seq_len, features)

        Returns:
            Predictions tensor
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded.")

        with torch.no_grad():
            return self.model(input_tensor.to(self.device))

    def is_loaded(self) -> bool:
        """Verifica se modelo está carregado."""
        return self.model is not None and self.preprocessor is not None

    def get_model_info(self) -> Dict:
        """Retorna informações do modelo carregado."""
        if not self.is_loaded():
            return {"loaded": False}

        return {
            "loaded": True,
            "ticker": self.current_ticker,
            "version": self.model_version,
            "device": str(self.device),
            "hyperparameters": self.model.get_hyperparameters(),
        }
