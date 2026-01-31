"""
Treinamento do modelo LSTM.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Callable, Optional

from src.models.lstm_model import LSTMPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Treina modelo LSTM com early stopping e checkpoints."""

    def __init__(
        self,
        model: LSTMPredictor,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.patience = early_stopping_patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.best_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

        logger.info(f"Trainer initialized. Device: {self.device}")

    def create_dataloader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Cria DataLoader a partir de arrays numpy."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Treina o modelo.

        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            epochs: Número de épocas
            progress_callback: Função chamada a cada época
                callback(epoch, total_epochs, train_loss, val_loss, best_loss)

        Returns:
            Dict com histórico: train_losses, val_losses, best_epoch
        """
        history = {
            "train_losses": [],
            "val_losses": [],
            "best_epoch": 0,
        }

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = self._train_epoch(train_loader)

            # Validation
            self.model.eval()
            val_loss = self._validate_epoch(val_loader)

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                history["best_epoch"] = epoch + 1
            else:
                self.epochs_without_improvement += 1

            # Logging
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {self.best_loss:.6f}"
            )

            # Callback para atualizar progresso
            if progress_callback:
                progress_callback(
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_loss=self.best_loss,
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restaurar melhor modelo
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        """Treina uma época."""
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader) -> float:
        """Valida uma época."""
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Avalia modelo no conjunto de teste.

        Returns:
            Dict com métricas: mae, rmse, mape, r2_score
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()

        # Métricas
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))

        # MAPE (evitar divisão por zero)
        mask = targets != 0
        mape = (
            np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        )

        # R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2_score": float(r2),
        }

        logger.info(
            f"Evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, "
            f"MAPE={mape:.2f}%, R²={r2:.4f}"
        )

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Salva modelo."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Carrega modelo."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
