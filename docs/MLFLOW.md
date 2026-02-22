# MLflow Integration Guide

## Overview

Este projeto usa MLflow para:
- Rastreamento de experimentos
- Comparação de runs
- Armazenamento de artifacts (modelos)
- Serving de modelos via `mlflow.pytorch.log_model`

O SQLite (`ModelRegistry`) continua sendo a fonte de verdade para modelos ativos.

## Arquitetura

### Padrão Orchestrator

```
Orchestrator (train.py ou training.py)
    │
    ├── 1. Gera version_id
    ├── 2. Abre mlflow.start_run(run_name=version_id)
    ├── 3. Loga params
    │
    │   ModelTrainer (passivo)
    │       │
    │       └── Loga métricas SE _MLFLOW_AVAILABLE E run ativa
    │
    ├── 4. Loga artifacts
    ├── 5. Loga modelo PyTorch (mlflow.pytorch.log_model)
    ├── 6. Registra no ModelRegistry com MESMO version_id
    └── 7. Se falhar: tag "status=failed" (em ambos orquestradores)
```

### Por que este padrão?

1. **Separação de responsabilidades**: Trainer não precisa saber sobre MLflow
2. **Testabilidade**: Trainer funciona sem MLflow instalado (import protegido)
3. **Consistência**: Um único version_id para tudo (UNIQUE no banco)
4. **Resiliência**: Falha de MLflow não impede treino (nullcontext fallback)
5. **Flexibilidade**: Fácil adicionar outros trackers

## Configuração

### Variáveis de Ambiente

```env
MLFLOW_TRACKING_URI=sqlite:///./data/mlflow.db
MLFLOW_EXPERIMENT_NAME=stock-prediction-lstm
MLFLOW_ARTIFACT_ROOT=./mlruns
```

### Comandos

```bash
# Servidor local
make mlflow-server

# UI
open http://localhost:5000
```

## Uso

### CLI

```bash
# Com tracking (padrão)
python scripts/train.py --ticker AAPL --epochs 100

# Sem tracking
python scripts/train.py --ticker AAPL --epochs 100 --no-mlflow
```

### API

```bash
# Sempre com tracking (fallback automático se MLflow indisponível)
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "epochs": 100}'
```

## Rastreabilidade

### SQLite → MLflow

```python
from src.database.repository import ModelRegistryRepository
import json

db = SessionLocal()
model = repo.get_active_model(db, "AAPL")
params = json.loads(model.hyperparameters)
mlflow_run_id = params.get("mlflow_run_id")
print(f"MLflow Run: {mlflow_run_id}")
```

### MLflow → SQLite

No MLflow UI, procure pela tag `version_id` e busque no ModelRegistry.

### Filtrar Runs Falhadas

```python
# No MLflow UI ou via API:
runs = mlflow.search_runs(
    filter_string="tags.status = 'failed'",
    order_by=["start_time DESC"]
)
```

## Comparação de Runs

### Via UI

1. Acesse http://localhost:5000
2. Selecione experimento "stock-prediction-lstm"
3. Compare métricas entre runs

### Via Python

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///./data/mlflow.db")

# Buscar melhores runs
runs = mlflow.search_runs(
    experiment_names=["stock-prediction-lstm"],
    filter_string="metrics.eval_mae < 1.0 and tags.status = 'completed'",
    order_by=["metrics.eval_mae ASC"]
)
print(runs[["run_id", "params.ticker", "metrics.eval_mae"]])
```

## Troubleshooting

### MLflow não está logando

1. Verifique se `setup_mlflow()` foi chamado
2. Verifique se a run está ativa: `mlflow.active_run()`
3. Verifique se `_MLFLOW_AVAILABLE` é True no Trainer
4. Verifique logs de warning

### Runs duplicadas

Isso acontece se `setup_mlflow()` for chamado múltiplas vezes dentro de uma run. O módulo `mlflow_setup.py` previne isso com flag `_mlflow_configured`.

### Runs ficam como "RUNNING" para sempre

Isso indica que o treino falhou sem logar a tag de falha. Verifique se ambos orquestradores (CLI e API) têm o bloco `except` com `mlflow.set_tag("status", "failed")`.

### Artifacts não aparecem

Verifique o `MLFLOW_ARTIFACT_ROOT` e permissões da pasta `./mlruns`.