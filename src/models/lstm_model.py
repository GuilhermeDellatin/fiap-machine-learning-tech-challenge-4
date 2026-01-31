"""
Arquitetura LSTM em PyTorch.
"""
import torch
import torch.nn as nn
from typing import Dict


class LSTMPredictor(nn.Module):
    """
    Modelo LSTM para previsão de séries temporais.

    Arquitetura:
    - LSTM layers com dropout
    - Fully connected layer
    - Saída linear (regressão)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """
        Args:
            input_size: Número de features (default: 1 para Close)
            hidden_size: Neurônios por camada LSTM
            num_layers: Camadas LSTM empilhadas
            dropout: Dropout entre camadas (0 se num_layers=1)
            output_size: Dimensão da saída
        """
        super(LSTMPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, sequence_length, input_size)

        Returns:
            Predictions (batch, output_size)
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Usar apenas último timestep
        last_output = lstm_out[:, -1, :]

        # Fully connected
        prediction = self.fc(last_output)

        return prediction

    def get_hyperparameters(self) -> Dict:
        """Retorna dict com hiperparâmetros."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
        }

    @property
    def device(self) -> torch.device:
        """Retorna device atual do modelo."""
        return next(self.parameters()).device
