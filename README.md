# Stock Prediction LSTM

Sistema de previsao de precos de acoes usando LSTM com PyTorch.

|![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)|![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)|![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)|
|:---:|:---:|:---:|

## Features

- **Predicao de precos** - Previsao de precos futuros usando LSTM
- **Treinamento via API** - Inicie treinamentos com uma requisicao
- **Inferencia rapida** - Endpoint otimizado para baixa latencia
- **Cache inteligente** - SQLite com expiracao automatica (24h)
- **Metricas Prometheus** - Monitoramento em tempo real
- **MLflow tracking** - Experimentos, metricas, parametros e artefatos
- **Docker ready** - Deploy com um comando

## Quick Start

### Instalacao Local

```bash
# Clonar repositorio
git clone <repo>
cd stock-prediction-lstm

# Instalar dependencias
pip install -e ".[dev]"

# Inicializar banco
make init-db

# Rodar API
make run
```

### Treinar Modelo

```bash
# Via CLI
python scripts/train.py --ticker PETR4.SA --epochs 100

# Via API
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "epochs": 50}'
```

### MLflow

```bash
# Subir UI local do MLflow
make mlflow-ui
```

Apos iniciar, acesse `http://localhost:5000` para acompanhar:
- Parametros de treino (arquitetura, learning rate, batch size etc.)
- Metricas por epoca (train/val loss)
- Metricas finais (MAE, RMSE, MAPE, R2)
- Artefatos gerados (`.pt`, scaler e relatorios de avaliacao)

### Fazer Predicao

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "days_ahead": 5}'
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Estrutura do Projeto
```
stock-prediction-lstm/
├── src/
│   ├── api/                 # FastAPI
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   ├── routes/          # health, training, inference, prediction
│   │   ├── schemas/         # Pydantic models
│   │   └── middleware/      # Prometheus metrics
│   ├── database/            # SQLAlchemy
│   │   ├── connection.py    # Engine, SessionLocal, get_db()
│   │   ├── models.py        # PriceCache, TrainingJob, ModelRegistry
│   │   └── repository.py    # CRUD operations
│   ├── data/
│   │   ├── collector.py     # yfinance + cache SQLite
│   │   └── preprocessor.py  # Normalização, sequences
│   ├── models/
│   │   ├── lstm_model.py    # nn.Module PyTorch
│   │   ├── trainer.py       # Treinamento com early stopping
│   │   └── predictor.py     # Inferência
│   ├── monitoring/          # Métricas Prometheus
│   └── utils/               # config.py, logger.py
├── scripts/                 # train.py, evaluate.py
├── docker/                  # Dockerfile, docker-compose.yml
├── tests/                   # pytest
├── specs/                   # Especificações detalhadas por fase
├── models/                  # Arquivos .pt e .joblib salvos
└── data/                    # SQLite database
```

## API Endpoints

### Training

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| POST | `/api/v1/training/start` | Inicia treinamento (202) |
| GET | `/api/v1/training/status/{job_id}` | Status do job |
| GET | `/api/v1/training/jobs` | Lista jobs |
| GET | `/api/v1/training/models` | Lista modelos |
| POST | `/api/v1/training/activate/{version_id}` | Ativa modelo |

### Prediction

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| POST | `/api/v1/predict` | Predicao de precos |
| GET | `/api/v1/predict/{ticker}` | Predicao rapida |
| POST | `/api/v1/predict/batch` | Multiplos tickers |

### Inference

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| POST | `/api/v1/inference` | Inferencia direta |
| POST | `/api/v1/inference/batch` | Batch inference |
| GET | `/api/v1/inference/warmup` | Aquece modelo |

### Health & Cache

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/api/v1/cache/info` | Info do cache |
| POST | `/api/v1/cache/sync/{ticker}` | Forca sync |

## Arquitetura

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────>│   FastAPI   │────>│   SQLite    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           v
                    ┌─────────────┐     ┌─────────────┐
                    │   LSTM      │<────│  yfinance   │
                    │  (PyTorch)  │     │   (cache)   │
                    └─────────────┘     └─────────────┘
```

## Cache

- Dados do yfinance sao cacheados no SQLite
- Cache expira apos **24 horas** (configuravel)
- Use `/api/v1/cache/sync/{ticker}` para forcar atualizacao

## Configuracao

Crie um arquivo `.env`:

```env
DATABASE_URL=sqlite:///./data/stock_cache.db
CACHE_EXPIRY_HOURS=24
EPOCHS=100
HIDDEN_SIZE=64
LOG_LEVEL=INFO
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=stock-prediction-lstm
```

## Desenvolvimento

```bash
# Testes
make test

# Lint
ruff check src/

# Format
black src/
```

## Licenca e Autores

### Desenvolvido por

- `Beatriz Rosa Carneiro Gomes - RM365967`
- `Cristine Scheibler - RM365433`
- `Guilherme Fernandes Dellatin - RM365508`
- `Iana Alexandre Neri - RM360484`
- `João Lucas Oliveira Hilario - RM366185`

Este projeto é apenas para fins educacionais e segue a licença MIT.
