"""
Testes para o coletor de dados com cache.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.data.collector import StockDataCollector


def test_cache_hit(test_db, sample_prices_in_cache):
    """Dados em cache válido devem ser retornados sem chamar yfinance."""
    collector = StockDataCollector()

    with patch.object(collector, "_download_from_yfinance") as mock_download:
        df = collector.download_data(test_db, "AAPL", "2024-01-01", "2024-01-10")

        mock_download.assert_not_called()
        assert len(df) > 0
        assert "Close" in df.columns


def test_cache_miss_downloads(test_db, sample_yfinance_df):
    """Cache vazio deve baixar do yfinance."""
    collector = StockDataCollector()

    with patch.object(collector, "_download_from_yfinance") as mock_download:
        mock_download.return_value = sample_yfinance_df

        df = collector.download_data(test_db, "AAPL", "2024-01-01", "2024-01-10")

        mock_download.assert_called_once()
        assert len(df) > 0


def test_cache_expired_redownloads(expired_cache, sample_yfinance_df):
    """Cache expirado (>24h) deve baixar novamente."""
    collector = StockDataCollector()

    with patch.object(collector, "_download_from_yfinance") as mock_download:
        mock_download.return_value = sample_yfinance_df

        df = collector.download_data(expired_cache, "AAPL", "2024-01-01", "2024-01-10")

        mock_download.assert_called_once()


def test_sync_data_ignores_cache(test_db, sample_prices_in_cache, sample_yfinance_df):
    """sync_data deve baixar mesmo com cache válido."""
    collector = StockDataCollector()

    with patch.object(collector, "_download_from_yfinance") as mock_download:
        mock_download.return_value = sample_yfinance_df

        df = collector.sync_data(test_db, "AAPL")

        mock_download.assert_called_once()


def test_validate_ticker_valid():
    """Ticker válido retorna True."""
    collector = StockDataCollector()

    with patch("src.data.collector.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame({"Close": [100.0]})
        mock_ticker.return_value = mock_instance

        assert collector.validate_ticker("AAPL") is True


def test_validate_ticker_invalid():
    """Ticker inválido retorna False."""
    collector = StockDataCollector()

    with patch("src.data.collector.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        assert collector.validate_ticker("INVALID") is False
