# Stock Prediction LSTM

Sistema de previsão de preços de ações usando redes LSTM com PyTorch, servido via FastAPI, rastreado com MLflow e containerizado com Docker.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.14+-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

</div>

---

## Sumário

- [Visão Geral](#visão-geral)
- [Features](#features)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquitetura](#arquitetura)
- [Quick Start](#quick-start)
  - [Docker (recomendado)](#docker-recomendado)
  - [Instalação Local](#instalação-local)
- [API Endpoints](#api-endpoints)
- [Configuração](#configuração)
- [Desenvolvimento](#desenvolvimento)
- [Autores](#autores)

---

## Visão Geral

Este projeto implementa um pipeline completo de Machine Learning para previsão de preços de ações:

1. **Coleta** — dados históricos via `yfinance` com cache inteligente em SQLite
2. **Treinamento** — modelo LSTM (PyTorch) com early stopping, rastreado no MLflow
3. **Serviço** — API REST (FastAPI) com endpoints de treinamento, predição e inferência
4. **Monitoramento** — métricas Prometheus + Grafana e tracing MLflow
5. **Deploy** — Docker Compose com serviços `api` + `mlflow`

---

## Features

- **Predição de preços** — previsão de N dias futuros usando LSTM
- **Treinamento via API** — inicie treinamentos assíncronos com uma requisição (202 Accepted)
- **Rastreamento MLflow** — experimentos, métricas, parâmetros e artifacts por run
- **Cache inteligente** — SQLite com expiração automática configurável (padrão: 24h)
- **ModelRegistry** — versionamento de modelos com ativação por ticker
- **Métricas Prometheus** — monitoramento em tempo real com Grafana opcional
- **Resiliência** — MLflow é opcional; falhas de tracking não interrompem o treino
- **Docker ready** — deploy completo com um comando

---

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
│   └── utils/               # config.py, logger.py, mlflow_setup.py
├── scripts/                 # scripts para execução CLI train.py, evaluate.py
├── docker/                  # Dockerfile, docker-compose.yml
├── tests/                   # pytest
├── models/                  # Arquivos .pt e .joblib salvos
└── data/                    # SQLite database
```

---

## Arquitetura

O sistema é composto por dois serviços principais orquestrados via Docker Compose: a **API FastAPI** (porta 8000) e o **servidor MLflow** (porta 5000). A API recebe requisições REST, delega o treinamento para tarefas em background, persiste dados no SQLite e busca dados de mercado via yfinance com cache. O monitoramento é opcional e ativado via perfil `--profile monitoring`.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                        Docker Compose                            │
  │                                                                  │
  │   ┌─────────────┐   REST   ┌───────────────────────────────┐     │
  │   │   Client    │ ────────>│         FastAPI :8000         │     │
  │   └─────────────┘          │  /training  /predict          │     │
  │                            │  /inference /health /metrics  │     │
  │   ┌─────────────┐          └──────────────┬────────────────┘     │
  │   │  scripts/   │ train.py                │                      │
  │   │  train.py   │ ──────────────────────> │                      │
  │   │  (CLI)      │                         │                      │
  │   └─────────────┘                         │                      │
  │                              ┌────────────┼────────────┐         │
  │                              │            │            │         │
  │                              v            v            v         │
  │                    ┌─────────────┐  ┌───────────┐ ┌─────────┐    │
  │                    │    LSTM     │  │  SQLite   │ │ MLflow  │    │
  │                    │  (PyTorch)  │  │  (db)     │ │ :5000   │    │
  │                    └──────┬──────┘  │PriceCache │ │runs/    │    │
  │                           │         │TrainingJob│ │metrics  │    │
  │                           │         │ModelReg.  │ │artifacts│    │
  │                           │         └───────────┘ └─────────┘    │
  │                           │                                      │
  │                           v                                      │
  │                    ┌─────────────┐   cache     ┌─────────────┐   │
  │                    │ Preprocessor│ <────────── │  yfinance   │   │
  │                    │ MinMaxScaler│  miss→fetch │  (mercado)  │   │
  │                    └─────────────┘             └─────────────┘   │
  │                                                                  │
  │   ┌─────────────┐  scrape   ┌─────────────┐                      │
  │   │  Prometheus │ <──────── │  /metrics   │                      │
  │   │   :9090     │           │  (endpoint) │                      │
  │   └──────┬──────┘           └─────────────┘                      │
  │          │                                                       │
  │          v                                                       │
  │   ┌─────────────┐                                                │
  │   │   Grafana   │                                                │
  │   │   :3000     │  (perfil: --profile monitoring)                │
  │   └─────────────┘                                                │
  └──────────────────────────────────────────────────────────────────┘
```

> Documentação detalhada: [ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Fluxo MLflow (Orchestrator Pattern)

O treinamento segue o padrão **Orchestrator**: apenas `train.py` (CLI) e `training.py` (API) abrem runs MLflow. O `ModelTrainer` é passivo — loga métricas somente se há uma run ativa. O `version_id` é gerado antes do treino e usado de forma consistente no nome do arquivo `.pt`, na run do MLflow e no registro do SQLite. MLflow é opcional: falhas de tracking não interrompem o treinamento.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Orchestrator  │────▶│  ModelTrainer   │────▶│    MLflow       │
│  (train.py ou   │     │  (passivo)      │     │  (se ativo)     │
│   training.py)  │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                               │
        │  mlflow.start_run()                           │
        │  version_id = gerar()                         │
        │  mlflow.pytorch.log_model()                   │
        │                                               ▼
        │                                       ┌─────────────────┐
        └─────────────────────────────────────▶ │  ModelRegistry  │
                                                │    (SQLite)     │
                                                └─────────────────┘
```
D
> Documentação detalhada: [MLFLOW.md](docs/MLFLOW.md)

---

## Quick Start

### Docker (recomendado)

A forma mais simples de rodar o projeto. Requer apenas [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado.

```bash
# Clonar repositório
git clone <repo>
cd stock-prediction-lstm

# Subir API + MLflow
docker compose -f docker/docker-compose.yml up -d --build

# Subir com monitoramento (Prometheus + Grafana)
docker compose -f docker/docker-compose.yml --profile monitoring up -d --build
```

| Serviço | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

#### Treinar um modelo via API

```bash
# 1. Iniciar treinamento (retorna 202 imediatamente)
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "epochs": 50}'

# 2. Acompanhar status (substituir {job_id} pelo retornado acima)
curl http://localhost:8000/api/v1/training/status/{job_id}

# 3. Fazer predição após treino concluído
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "days_ahead": 5}'
```

> **Atenção:** É necessário ter um modelo treinado para o ticker antes de fazer predições. Caso contrário, a API retorna `404`.

---

### Instalação Local

Para desenvolvimento e execução fora do Docker. Requer Python 3.11+.

```bash
# Clonar repositório
git clone <repo>
cd stock-prediction-lstm

# Instalar dependências
pip install -e ".[dev]"

# Inicializar banco
make init-db

# Rodar API
make run
```

### Treinar Modelo Localmente

#### Via CLI

```bash
# Com tracking MLflow
python scripts/train.py --ticker PETR4.SA --epochs 100

# Sem tracking MLflow
python scripts/train.py --ticker AAPL --epochs 50 --no-mlflow

# Opções avançadas
python scripts/train.py --ticker PETR4.SA \
  --epochs 100 \
  --hidden-size 128 \
  --num-layers 2 \
  --learning-rate 0.001 \
  --sequence-length 60
```

#### Via API

```bash
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "epochs": 50}'
```

---

## API Endpoints

A API segue o padrão REST com versionamento `/api/v1/`. Treinamentos são disparados de forma assíncrona (retornam `202 Accepted`) e seu progresso é acompanhado via `job_id`. Predições exigem um modelo previamente treinado para o ticker — caso contrário retornam `404`. A documentação interativa completa (Swagger) está disponível em `http://localhost:8000/docs`.

### Training

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/training/start` | Inicia treinamento assíncrono (202) |
| GET | `/api/v1/training/status/{job_id}` | Status do job |
| GET | `/api/v1/training/jobs` | Lista todos os jobs |
| GET | `/api/v1/training/models` | Lista modelos registrados |
| POST | `/api/v1/training/activate/{version_id}` | Ativa modelo específico |

### Prediction

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/predict` | Predição de preços futuros |
| POST | `/api/v1/predict/batch` | Predição para múltiplos tickers |

### Inference

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/inference` | Inferência direta (dados pré-processados) |
| POST | `/api/v1/inference/batch` | Batch inference |
| GET | `/api/v1/inference/warmup` | Aquece modelo em memória |

### Health & Cache

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/metrics` | Métricas Prometheus |
| GET | `/api/v1/cache/info` | Informações do cache |
| POST | `/api/v1/cache/sync/{ticker}` | Força sincronização de dados |

> Documentação detalhada: [API.md](docs/API.md)

---

## Configuração

Crie um arquivo `.env` na raiz (veja `.env.example`):

```env
DATABASE_URL=sqlite:///./data/stock_cache.db
CACHE_EXPIRY_HOURS=24
MODEL_DIR=models
SEQUENCE_LENGTH=60
HIDDEN_SIZE=64
NUM_LAYERS=2
EPOCHS=100
LOG_LEVEL=INFO

# MLflow (local)
MLFLOW_TRACKING_URI=sqlite:///./data/mlflow.db
MLFLOW_EXPERIMENT_NAME=stock-prediction-lstm
MLFLOW_ARTIFACT_ROOT=./mlruns
```

---

## Desenvolvimento

```bash
# Rodar testes
make test

# Testes com cobertura
pytest --cov=src tests/

# Lint
ruff check src/

# Format
black src/
```

---

## Autores

Desenvolvido como projeto educacional — FIAP Machine Learning Tech Challenge 4.

| Nome | RM |
|------|----|
| Beatriz Rosa Carneiro Gomes | RM365967 |
| Cristine Scheibler | RM365433 |
| Guilherme Fernandes Dellatin | RM365508 |
| Iana Alexandre Neri | RM360484 |
| João Lucas Oliveira Hilario | RM366185 |

Este projeto é apenas para fins educacionais e segue a licença MIT.