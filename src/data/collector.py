"""
Coleta de dados do yfinance com cache SQLite.
"""
import time

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from sqlalchemy.orm import Session

from src.database.repository import PriceCacheRepository
from src.monitoring.metrics import record_cache_hit, record_cache_miss
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TickerNotFoundError(Exception):
    """Ticker não encontrado no yfinance."""
    pass


class DataDownloadError(Exception):
    """Erro ao baixar dados."""
    pass


class StockDataCollector:
    """
    Coleta dados do yfinance com cache SQLite.

    LÓGICA DE CACHE:
    1. Antes de baixar, verificar SQLite
    2. Se cache válido (< CACHE_EXPIRY_HOURS), usar cache
    3. Se cache inválido/expirado, baixar do yfinance
    4. Salvar novos dados no cache
    """

    def __init__(self) -> None:
        self.cache_repo = PriceCacheRepository()

    def download_data(
        self, db: Session, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Baixa dados com cache inteligente.

        Args:
            db: Sessão SQLAlchemy
            ticker: Código da ação (ex: "PETR4.SA")
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)

        Returns:
            DataFrame com colunas: Date, Open, High, Low, Close, Adj Close, Volume

        Raises:
            TickerNotFoundError: Se ticker não existir
            DataDownloadError: Se falhar ao baixar
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        # 1. Verificar cache
        if self.cache_repo.is_cache_valid(
            db, ticker, start, end,
            max_age_hours=settings.CACHE_EXPIRY_HOURS
        ):
            logger.info(f"Cache HIT para {ticker}")
            record_cache_hit(ticker)
            return self._get_from_cache(db, ticker, start, end)

        # 2. Cache miss - baixar do yfinance
        logger.info(f"Cache MISS para {ticker} - baixando do yfinance")
        record_cache_miss(ticker)
        df = self._download_from_yfinance(ticker, start_date, end_date)

        # 3. Salvar no cache
        saved = self.cache_repo.save_prices(db, ticker, df)
        logger.info(f"Salvos {saved} registros no cache para {ticker}")

        return df

    def sync_data(self, db: Session, ticker: str, years: int = 5) -> pd.DataFrame:
        """
        Força sincronização com yfinance (ignora cache).
        Útil antes de treinar para garantir dados atualizados.

        Args:
            db: Sessão SQLAlchemy
            ticker: Código da ação
            years: Anos de histórico (default: 5)

        Returns:
            DataFrame atualizado
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

        logger.info(f"Sincronizando {ticker} (forçado)")

        # Baixar diretamente
        df = self._download_from_yfinance(ticker, start_date, end_date)

        # Atualizar cache
        self.cache_repo.delete_cache(db, ticker)
        self.cache_repo.save_prices(db, ticker, df)

        logger.info(f"Sincronizados {len(df)} registros para {ticker}")
        return df

    def get_latest_price(self, db: Session, ticker: str) -> dict:
        """
        Retorna último preço disponível.

        Returns:
            {"ticker": str, "date": str, "close": float, "source": "cache"|"yfinance"}
        """
        cache_info = self.cache_repo.get_cache_info(db, ticker)

        if cache_info and cache_info["last_updated"]:
            last_updated = datetime.fromisoformat(cache_info["last_updated"])
            if self._is_recent(last_updated):
                end_date = date.fromisoformat(cache_info["date_range"]["end"])
                prices = self.cache_repo.get_prices(db, ticker, end_date, end_date)
                if prices:
                    return {
                        "ticker": ticker,
                        "date": str(prices[-1].date),
                        "close": prices[-1].close,
                        "source": "cache",
                    }

        # Buscar do yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")

        if hist.empty:
            raise TickerNotFoundError(f"No data for {ticker}")

        return {
            "ticker": ticker,
            "date": str(hist.index[-1].date()),
            "close": float(hist["Close"].iloc[-1]),
            "source": "yfinance",
        }

    def validate_ticker(self, ticker: str) -> bool:
        """Verifica se ticker existe no yfinance."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            return not hist.empty
        except Exception:
            return False

    def _download_from_yfinance(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        Download do yfinance com retry e backoff exponencial (3 tentativas).

        Raises:
            TickerNotFoundError: Se ticker não retornar dados
            DataDownloadError: Se todas as tentativas falharem
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start, end=end)

                if df.empty:
                    raise TickerNotFoundError(f"No data found for {ticker}")

                # Resetar index para ter Date como coluna
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.date

                return df

            except TickerNotFoundError:
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1, 2, 4 segundos
                    logger.warning(f"Tentativa {attempt + 1} falhou, aguardando {wait}s")
                    time.sleep(wait)
                else:
                    raise DataDownloadError(f"Failed to download {ticker}: {e}")

    def _get_from_cache(
        self, db: Session, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """Converte registros do cache para DataFrame."""
        prices = self.cache_repo.get_prices(db, ticker, start, end)

        data = [
            {
                "Date": p.date,
                "Open": p.open,
                "High": p.high,
                "Low": p.low,
                "Close": p.close,
                "Adj Close": p.adj_close,
                "Volume": p.volume,
            }
            for p in prices
        ]

        return pd.DataFrame(data)

    def _is_recent(self, dt: datetime, hours: int = None) -> bool:
        """Verifica se datetime é recente."""
        if hours is None:
            hours = settings.CACHE_EXPIRY_HOURS
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.utcnow()
        return (now - dt).total_seconds() < hours * 3600
