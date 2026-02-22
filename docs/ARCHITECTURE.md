# Arquitetura do Sistema — Stock Prediction LSTM

## 1. Visão Geral

Sistema de previsão de preços de ações usando redes neurais LSTM (PyTorch), servido via FastAPI, com rastreamento de experimentos pelo MLflow e observabilidade via Prometheus/Grafana.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER COMPOSE                                      │
│                                                                                  │
│  ┌──────────────────┐     ┌──────────────────────────────────────────────────┐   │
│  │  MLflow Server   │     │                  API (FastAPI)                   │   │
│  │  :5000           │◀────│  :8000                                           │   │
│  │                  │     │  ┌──────────┐ ┌──────────┐ ┌────────────────┐    │   │
│  │  mlflow.db       │     │  │/training │ │/predict  │ │/inference      │    │   │
│  │  (SQLite)        │     │  └──────────┘ └──────────┘ └────────────────┘    │   │
│  │  mlruns/         │     └──────────────────────────────────────────────────┘   │
│  │  (artifacts)     │                          │                                 │
│  └──────────────────┘                          │                                 │
│                                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                          Volumes Compartilhados                         │     │
│  │  data/stock_cache.db (SQLite)  │  models/*.pt  │  models/*_scaler.joblib│     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  ┌──────────────────┐     ┌──────────────────┐                                   │
│  │   Prometheus     │     │    Grafana       │   (perfil: --profile monitoring)  │
│  │   :9090          │────▶│    :3000         │                                   │
│  └──────────────────┘     └──────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
         ▲                                ▲
         │ scrape /metrics                │ CLI (fora do Docker)
         │                                │
  ┌──────┴────────────┐     ┌─────────────┴──────────────┐
  │  API /metrics     │     │  scripts/train.py          │
  │  (Prometheus)     │     │  (Orchestrator CLI)        │
  └───────────────────┘     └────────────────────────────┘
```

---

## 2. Estrutura de Módulos

```
src/
├── api/
│   ├── main.py              # FastAPI app, lifespan, middlewares, routers
│   ├── dependencies.py      # Injeção: get_database, get_collector, get_predictor
│   ├── routes/
│   │   ├── health.py        # GET /health
│   │   ├── training.py      # POST /training/start  ← ORCHESTRATOR API
│   │   ├── prediction.py    # POST /predict (requer modelo treinado)
│   │   └── inference.py     # POST /inference (dados brutos ou normalizados)
│   ├── schemas/
│   │   ├── training.py      # TrainingRequest, TrainingResponse, ModelInfo…
│   │   ├── prediction.py    # PredictionRequest, PredictionResponse…
│   │   └── inference.py     # InferenceRequest, InferenceResponse…
│   └── middleware/
│       └── metrics.py       # MetricsMiddleware → Prometheus counters/histograms
│
├── database/
│   ├── connection.py        # Engine SQLAlchemy, SessionLocal, Base, init_db()
│   ├── models.py            # PriceCache, TrainingJob, ModelRegistry (ORM)
│   └── repository.py        # CRUD: PriceCacheRepository, TrainingJobRepository,
│                            #        ModelRegistryRepository
│
├── data/
│   ├── collector.py         # StockDataCollector → yfinance + cache SQLite
│   └── preprocessor.py      # DataPreprocessor → normalização + sequências
│
├── models/
│   ├── lstm_model.py        # LSTMPredictor (nn.Module PyTorch)
│   ├── trainer.py           # ModelTrainer → treino + early stopping (passivo MLflow)
│   └── predictor.py         # StockPredictor → inferência em produção
│
├── monitoring/
│   └── metrics.py           # Counters/Histograms Prometheus (singleton REGISTRY)
│
└── utils/
    ├── config.py            # Settings (Pydantic BaseSettings, lê .env)
    ├── logger.py            # get_logger() → logging estruturado
    ├── mlflow_setup.py      # setup_mlflow(), generate_version_id()
    └── mlflow_tracing.py    # MLflowTracing (context managers semânticos)

scripts/
└── train.py                 # CLI Orchestrator (mlflow.start_run aqui)

docker/
├── Dockerfile
├── docker-compose.yml       # api + mlflow + prometheus* + grafana*
└── prometheus.yml
```

---

## 3. Banco de Dados SQLite

Dois bancos SQLite distintos:

```
data/stock_cache.db                          data/mlflow.db
┌─────────────────────────────┐              ┌──────────────────────────┐
│         price_cache         │              │  MLflow tracking store   │
│─────────────────────────────│              │  (experiments, runs,     │
│ id         INTEGER PK       │              │   params, metrics, tags) │
│ ticker     VARCHAR(20) IDX  │              └──────────────────────────┘
│ date       DATE             │
│ open       FLOAT            │       mlruns/   (artifact store)
│ high       FLOAT            │       ├── <experiment_id>/
│ low        FLOAT            │       │   └── <run_id>/
│ close      FLOAT  NOT NULL  │       │       ├── model/*.pt
│ adj_close  FLOAT            │       │       ├── model/*_scaler.joblib
│ volume     BIGINT           │       │       └── lstm_model/ (PyTorch)
│ created_at DATETIME         │
│ updated_at DATETIME         │
│ UNIQUE(ticker, date)        │
└─────────────────────────────┘

┌─────────────────────────────┐    ┌────────────────────────────────────┐
│        training_jobs        │    │          model_registry            │
│─────────────────────────────│    │────────────────────────────────────│
│ id            INTEGER PK    │    │ id            INTEGER PK           │
│ job_id        VARCHAR(36)   │    │ version_id    VARCHAR(50) UNIQUE   │
│ ticker        VARCHAR(20)   │    │ ticker        VARCHAR(20)          │
│ status        VARCHAR(20)   │    │ model_path    VARCHAR(255)         │
│   pending / running /       │    │ scaler_path   VARCHAR(255)         │
│   completed / failed        │    │ mae           FLOAT                │
│ epochs_total  INTEGER       │    │ rmse          FLOAT                │
│ epochs_completed INTEGER    │    │ mape          FLOAT                │
│ current_loss  FLOAT         │    │ r2_score      FLOAT                │
│ best_loss     FLOAT         │    │ epochs_trained INTEGER             │
│ error_message TEXT          │    │ hyperparameters TEXT (JSON)        │
│ model_version_id VARCHAR    │    │ is_active     BOOLEAN              │
│ hyperparameters TEXT (JSON) │    │ mlflow_run_id VARCHAR (opcional)   │
│ created_at    DATETIME      │    │ created_at    DATETIME             │
│ started_at    DATETIME      │    └────────────────────────────────────┘
│ completed_at  DATETIME      │
└─────────────────────────────┘
```

> **Fonte de verdade**: `model_registry.is_active` determina qual modelo serve
> predições. O MLflow é complementar — armazena métricas e artefatos para
> experimentação, mas **não** decide qual modelo está em produção.

---

## 4. Pipeline de Treinamento

Existem dois orquestradores: **CLI** (`scripts/train.py`) e **API** (`routes/training.py`). Ambos seguem o mesmo fluxo.

```
┌────────────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR  (CLI ou API — training.py)              │
│                                                                    │
│  1. setup_mlflow()          ← configura tracking URI + experiment  │
│  2. version_id = generate_version_id(ticker)                       │
│     ex: "AAPL_20240315_143022"                                     │
│  3. mlflow.start_run(run_name=version_id)   ← abre a run           │
│  4. mlflow.log_params(hyperparams)                                 │
│  5. mlflow.set_tags({source, ticker, version_id})                  │
│                                                                    │
│  ┌─── tracing.pipeline("training_pipeline") ──────────────────┐    │
│  │                                                            │    │
│  │  [fetch_historical_data]                                   │    │
│  │      StockDataCollector.sync_data()                        │    │
│  │          ├── cache válido? → lê PriceCache (SQLite)        │    │
│  │          └── expirado?    → yfinance → atualiza cache      │    │
│  │                                                            │    │
│  │  [preprocessing]                                           │    │
│  │      DataPreprocessor.fit_transform()  → MinMaxScaler      │    │
│  │      DataPreprocessor.create_sequences(seq_len=60)         │    │
│  │      DataPreprocessor.split_data()  → 70/15/15             │    │
│  │                                                            │    │
│  │  [model_creation]                                          │    │
│  │      LSTMPredictor(input=1, hidden=64, layers=2, drop=0.2) │    │
│  │                                                            │    │
│  │  [model_training]     ← ModelTrainer (PASSIVO)             │    │
│  │      trainer.train(train_loader, val_loader, epochs)       │    │
│  │          ├── por época: MSELoss, Adam, ReduceLROnPlateau   │    │
│  │          ├── early stopping (patience=10)                  │    │
│  │          └── se mlflow.active_run(): log train/val_loss    │    │
│  │                                                            │    │
│  │  [model_evaluation]                                        │    │
│  │      trainer.evaluate(test_loader) → MAE, RMSE, MAPE, R²   │    │
│  │      se mlflow.active_run(): log eval_mae, eval_rmse…      │    │
│  │                                                            │    │
│  │  [save_model]                                              │    │
│  │      trainer.save_checkpoint(models/{version_id}.pt)       │    │
│  │      preprocessor.save_scaler(models/{version_id}_scaler…) │    │
│  │                                                            │    │
│  │  [artifact_logging]  (se mlflow habilitado)                │    │
│  │      mlflow.log_artifact(.pt)                              │    │
│  │      mlflow.log_artifact(_scaler.joblib)                   │    │
│  │      mlflow.pytorch.log_model(model, "lstm_model")         │    │
│  │      mlflow.set_tag("status", "completed")                 │    │
│  │                                                            │    │
│  │  [model_registration]                                      │    │
│  │      ModelRegistryRepository.register_model()              │    │
│  │          → INSERT INTO model_registry (version_id, paths,  │    │
│  │                         metrics, mlflow_run_id)            │    │
│  │      ModelRegistryRepository.set_active_model()            │    │
│  │          → UPDATE is_active=True  para version_id          │    │
│  │          → UPDATE is_active=False para modelos anteriores  │    │
│  │                                                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  Em caso de falha:                                                 │
│      mlflow.set_tag("status", "failed")                            │
│      mlflow.set_tag("error", str(e)[:250])                         │
│      TrainingJobRepository.update_job(status="failed")  [API only] │
└────────────────────────────────────────────────────────────────────┘
```

### Diferença CLI vs API

| Aspecto | CLI (`train.py`) | API (`training.py`) |
|---------|-----------------|---------------------|
| Entrada | `argparse` | `TrainingRequest` (JSON) |
| Execução | Síncrona | `BackgroundTasks` (async) |
| Resposta | Log no console | HTTP 202 + `job_id` |
| Progresso | Log por época | `TrainingJob` atualizado por callback |
| Flag MLflow | `--no-mlflow` | Sempre tenta (fallback silencioso) |
| Fonte | tag `source=cli` | tag `source=api` |

---

## 5. Padrão Orchestrator + Trainer Passivo

```
┌──────────────────────────────────────────────────────────────────┐
│  REGRA: mlflow.start_run() APENAS no Orchestrator                │
│                                                                  │
│  Orchestrator                   ModelTrainer (passivo)           │
│  ─────────────                  ───────────────────────          │
│  mlflow.start_run()  ────────▶  if _MLFLOW_AVAILABLE             │
│                                    and mlflow.active_run():      │
│                                      mlflow.log_metric(...)      │
│                                                                  │
│  Falha de log nunca interrompe o treino (try-except silencioso)  │
│  MLflow não instalado → _MLFLOW_AVAILABLE=False → sem logging    │
│  --no-mlflow → nullcontext() → active_run()=None → sem logging   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Pipeline de Predição

```
Cliente
  │
  │  POST /api/v1/predict
  │  { "ticker": "AAPL", "days_ahead": 5 }
  ▼
┌─────────────────────────────────────────────────────────────┐
│  prediction.py  (tracing.safe_span "prediction_pipeline")   │
│                                                             │
│  [load_model]                                               │
│      ModelRegistryRepository.get_active_model(ticker)       │
│      ├── Encontrou? → continua                              │
│      └── Não encontrou? → HTTP 404                          │
│          { "error": "No trained model found for AAPL",      │
│            "suggestion": "Train a model first..." }         │
│                                                             │
│  [ensure_model_loaded]                                      │
│      StockPredictor.is_loaded() && ticker correto?          │
│      └── Não → predictor.reload_model(.pt, _scaler.joblib)  │
│                                                             │
│  [fetch_historical_data]                                    │
│      StockDataCollector.download_data(ticker, -365d, hoje)  │
│      ├── cache válido → SQLite                              │
│      └── expirado → yfinance                                │
│                                                             │
│  [model_predict]                                            │
│      StockPredictor.predict(df, days_ahead=5)               │
│      → retorna lista de preços desnormalizados              │
│                                                             │
│  [format_response]                                          │
│      Gera datas futuras (pula fins de semana)               │
│      → PredictionResponse { predictions: [...] }            │
└─────────────────────────────────────────────────────────────┘
  │
  │  HTTP 200
  │  { "ticker": "AAPL",
  │    "model_version": "AAPL_20240315_143022",
  │    "predictions": [
  │      { "date": "2024-03-18", "price": 182.45 },
  │      ...
  │    ]
  │  }
  ▼
Cliente
```

---

## 7. Pipeline de Inferência Direta

```
POST /api/v1/inference
  │
  ├── raw_prices: [182.1, 183.5, ...]   → predict() sobre preços reais
  │                                        retorna preço desnormalizado
  └── data: [0.82, 0.83, ...]           → inference() sobre tensor normalizado
                                           retorna normalizado ou desnormalizado
                                           (return_normalized flag)

POST /api/v1/inference/batch
  │
  └── sequences: [[...], [...], ...]    → inference() em batch
                                           retorna lista de predições normalizadas
```

---

## 8. Arquitetura do Modelo LSTM

```
Input
(batch_size, sequence_length=60, input_size=1)
         │
         ▼
┌─────────────────────────────────┐
│  nn.LSTM                        │
│  input_size  = 1                │
│  hidden_size = 64               │
│  num_layers  = 2                │
│  dropout     = 0.2 (entre cam.) │
│  batch_first = True             │
└─────────────────────────────────┘
         │
         │  output: (batch, seq_len, hidden_size=64)
         │  usa apenas último timestep → (batch, 64)
         ▼
┌─────────────────────────────────┐
│  nn.Linear(64 → 1)              │
└─────────────────────────────────┘
         │
         ▼
Output
(batch_size, 1)  — preço normalizado [0, 1]
```

### Hiperparâmetros padrão

| Parâmetro | Valor |
|-----------|-------|
| `sequence_length` | 60 dias |
| `hidden_size` | 64 |
| `num_layers` | 2 |
| `dropout` | 0.2 |
| `learning_rate` | 0.001 |
| `batch_size` | 32 |
| `epochs` | 100 |
| `early_stopping_patience` | 10 |

### Otimização

```
Loss:       MSELoss
Optimizer:  Adam (lr=0.001)
Scheduler:  ReduceLROnPlateau (factor=0.5, patience=5)
Device:     CUDA (se disponível) → fallback CPU
```

---

## 9. Cache de Dados (Lógica Crítica)

```
StockDataCollector.download_data(db, ticker, start, end)
         │
         ▼
   Verificar SQLite (PriceCache)
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
Cache         Cache ausente
existe?       ou expirado
   │            │
   ▼            ▼
(now - updated_at)    yfinance.download()
   < 24h?               │
   │                    ▼
   ├── Sim → DataFrame  INSERT/UPDATE
   │         do cache   PriceCache
   └── Não ─────────────┘
                │
                ▼
          DataFrame retornado
          ao caller
```

A mesma lógica se aplica em `sync_data()` (treino) e `download_data()` (predição).

---

## 10. Integração MLflow

### Tracking (Experimentos)

```
MLflow Experiment: "stock-prediction-lstm"
   │
   └── Run (run_name = version_id)
         ├── Params:  ticker, epochs, hidden_size, lr, ...
         ├── Tags:    source, version_id, status
         ├── Metrics: train_loss (por step/epoch)
         │            val_loss (por step/epoch)
         │            eval_mae, eval_rmse, eval_mape, eval_r2_score
         │            dataset_size, train_size, val_size, test_size
         └── Artifacts:
               model/{version_id}.pt
               model/{version_id}_scaler.joblib
               lstm_model/ (PyTorch MLflow format)
```

### Tracing (Spans por Operação)

```
prediction_pipeline  (safe_span — nunca quebra)
├── load_model
├── ensure_model_loaded
├── fetch_historical_data
├── model_predict
└── format_response

inference_pipeline  (safe_span)
├── load_model
└── model_inference | batch_model_inference

training_pipeline  (span — dentro da run)
├── fetch_historical_data
├── preprocessing
├── model_creation
├── model_training
├── model_evaluation
├── save_model
├── artifact_logging
└── model_registration
```

### Consistência de Versão

```
version_id gerado ANTES do treino
     │
     ├──▶  models/{version_id}.pt             (arquivo no disco)
     ├──▶  models/{version_id}_scaler.joblib   (arquivo no disco)
     ├──▶  mlflow.start_run(run_name=version_id)
     ├──▶  mlflow.set_tag("version_id", version_id)
     └──▶  model_registry.version_id = version_id  (UNIQUE constraint)
```

---

## 11. Camada de API

### Endpoints

| Método | Rota | Status | Descrição |
|--------|------|--------|-----------|
| `GET` | `/` | 200 | Info da API |
| `GET` | `/health` | 200 | Health check |
| `GET` | `/metrics` | 200 | Métricas Prometheus |
| `POST` | `/api/v1/training/start` | **202** | Inicia treino (background) |
| `GET` | `/api/v1/training/status/{job_id}` | 200/404 | Status do job |
| `GET` | `/api/v1/training/jobs` | 200 | Lista jobs |
| `GET` | `/api/v1/training/models` | 200 | Lista modelos |
| `POST` | `/api/v1/training/activate/{version_id}` | 200 | Ativa modelo |
| `POST` | `/api/v1/predict` | 200/**404** | Predição futura |
| `GET` | `/api/v1/predict/{ticker}` | 200/**404** | Predição via GET |
| `POST` | `/api/v1/predict/batch` | 200 | Predição em lote |
| `POST` | `/api/v1/inference` | 200/404 | Inferência direta |
| `POST` | `/api/v1/inference/batch` | 200/404 | Batch inference |
| `GET` | `/api/v1/inference/warmup` | 200 | Aquece modelo |

### Middlewares e Startup

```
FastAPI app (lifespan)
  │
  ├── Startup:
  │     init_db()          → cria tabelas SQLite se não existirem
  │     setup_mlflow()     → configura experiment (warning se falhar)
  │
  ├── MetricsMiddleware    → contabiliza requests Prometheus
  └── CORSMiddleware       → origins configuráveis via .env
```

---

## 12. Monitoramento

```
┌────────────────────────────────────────────────────────────┐
│                    Métricas Prometheus                     │
│                                                            │
│  HTTP:                                                     │
│    http_requests_total{method, endpoint, status}           │
│    http_request_duration_seconds{method, endpoint}         │
│                                                            │
│  Modelo:                                                   │
│    model_predictions_total{ticker}                         │
│    model_inference_duration_seconds{ticker}                │
│                                                            │
│  Treino:                                                   │
│    training_jobs_total{status}                             │
│    training_duration_seconds                               │
│                                                            │
│  Cache:                                                    │
│    cache_hits_total{ticker}                                │
│    cache_misses_total{ticker}                              │
└────────────────────────────────────────────────────────────┘
         │
         │ scrape :8000/metrics
         ▼
    Prometheus :9090
         │
         ▼
    Grafana :3000
```

---

## 13. Infraestrutura Docker

```
docker-compose.yml
│
├── mlflow  (:5000)
│     Image:   Dockerfile (mesma base da API)
│     Command: mlflow server --backend-store-uri sqlite:///./data/mlflow.db
│     Volumes: data/ (mlflow.db), mlruns/ (artifacts)
│     Health:  GET /health
│
├── api  (:8000)
│     Image:   Dockerfile
│     Depends: mlflow (healthy)
│     Env:     DATABASE_URL, MLFLOW_TRACKING_URI=http://mlflow:5000, ...
│     Volumes: models/, data/, mlruns/
│     Health:  GET /health
│
├── prometheus  (:9090)  [profile: monitoring]
│     Image:   prom/prometheus
│     Config:  docker/prometheus.yml
│
└── grafana  (:3000)  [profile: monitoring]
      Image:   grafana/grafana
      Depends: prometheus
```

### Iniciar o sistema

```bash
# Apenas API + MLflow
docker compose up

# Com monitoramento
docker compose --profile monitoring up

# Treinar via CLI (fora do Docker)
python scripts/train.py --ticker AAPL --epochs 50

# Treinar sem MLflow tracking
python scripts/train.py --ticker AAPL --epochs 50 --no-mlflow
```

---

## 14. Fluxo Completo End-to-End

```
  Usuário / Sistema Externo
         │
         │  1. POST /api/v1/training/start
         │     { "ticker": "PETR4.SA", "epochs": 100 }
         ▼
  ┌──────────────────┐
  │   FastAPI API    │  → HTTP 202 { "job_id": "uuid" }
  │                  │
  │  BackgroundTask  │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────┐
  │  train_model_task()  [Orchestrator API]          │
  │                                                  │
  │  setup_mlflow()  →  MLflow Server (:5000)        │
  │  version_id = "PETR4.SA_20240315_143022"         │
  │  mlflow.start_run(run_name=version_id)           │
  │                                                  │
  │  StockDataCollector.sync_data()                  │
  │    └── cache miss → yfinance → PriceCache        │
  │                                                  │
  │  DataPreprocessor → MinMaxScaler → sequences     │
  │                                                  │
  │  LSTMPredictor (PyTorch) criado                  │
  │                                                  │
  │ ModelTrainer.train() ←─ log por época ──▶ MLflow │
  │    └── early stopping                            │
  │                                                  │
  │  ModelTrainer.evaluate() → MAE/RMSE/MAPE/R²      │
  │                                                  │
  │  salva .pt + .joblib em models/                  │
  │  mlflow.pytorch.log_model()  →  mlruns/          │
  │                                                  │
  │  ModelRegistryRepository.register_model()        │
  │    INSERT model_registry (version_id, is_active) │
  │  set_active_model()                              │
  │    UPDATE is_active = True  (novo)               │
  │    UPDATE is_active = False (anterior)           │
  └──────────────────────────────────────────────────┘
           │
           │  2. GET /api/v1/training/status/{job_id}
           │     → { "status": "completed", "mae": 0.0042, ... }
           │
           │  3. POST /api/v1/predict
           │     { "ticker": "PETR4.SA", "days_ahead": 5 }
           ▼
  ┌──────────────────────────────────────────────────┐
  │  prediction.py                                   │
  │                                                  │
  │  get_active_model("PETR4.SA") → model_registry   │
  │  reload_model(.pt, _scaler.joblib)  [se mudou]   │
  │  download_data()  → cache ou yfinance            │
  │  predictor.predict(df, days_ahead=5)             │
  │    └── retorna [29.45, 29.67, 29.82, 30.1, 30.3] │
  └──────────────────────────────────────────────────┘
           │
           │  HTTP 200
           │  { "predictions": [
           │      {"date": "2024-03-18", "price": 29.45},
           │      ...
           │    ]
           │  }
           ▼
  Usuário / Sistema Externo
```