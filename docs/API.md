# API Reference

## Base URL

```
http://localhost:8000
```

## Autenticacao

Nao requerida (adicione em producao se necessario).

---

## Training

### POST /api/v1/training/start

Inicia um job de treinamento em background.

**Request:**
```json
{
  "ticker": "PETR4.SA",
  "epochs": 100,
  "batch_size": 32,
  "hidden_size": 64,
  "num_layers": 2,
  "learning_rate": 0.001
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted",
  "message": "Training job started for PETR4.SA",
  "ticker": "PETR4.SA"
}
```

### GET /api/v1/training/status/{job_id}

Consulta o status de um job de treinamento.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "ticker": "PETR4.SA",
  "status": "running",
  "progress_percent": 45.0,
  "epochs_completed": 45,
  "epochs_total": 100,
  "current_loss": 0.0023,
  "best_loss": 0.0019
}
```

---

## Prediction

### POST /api/v1/predict

Realiza predicao de precos futuros para um ticker.

**Request:**
```json
{
  "ticker": "PETR4.SA",
  "days_ahead": 5
}
```

**Response (200):**
```json
{
  "ticker": "PETR4.SA",
  "model_version": "PETR4.SA_20240115_143022",
  "predictions": [
    {"date": "2024-01-16", "price": 38.50},
    {"date": "2024-01-17", "price": 38.75}
  ],
  "generated_at": "2024-01-15T16:45:00Z"
}
```

**Response (404) - Sem modelo:**
```json
{
  "error": "No trained model found for ticker PETR4.SA",
  "suggestion": "Train a model first using POST /api/v1/training/start",
  "ticker": "PETR4.SA"
}
```

---

## Inference

### POST /api/v1/inference

Inferencia direta com precos brutos.

**Request:**
```json
{
  "ticker": "PETR4.SA",
  "raw_prices": [38.5, 39.0, 38.2, 38.8]
}
```

**Response:**
```json
{
  "ticker": "PETR4.SA",
  "prediction": 39.15,
  "is_normalized": false,
  "model_version": "PETR4.SA_20240115_143022",
  "inference_time_ms": 5.2
}
```

---

## Health & Monitoring

### GET /health

Retorna status de saude da aplicacao.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "models_loaded": 2
}
```

### GET /metrics

Retorna metricas no formato Prometheus.

---

## Errors

| Status | Descricao |
|--------|-----------|
| 400 | Bad Request - Validacao falhou |
| 404 | Not Found - Recurso nao existe |
| 500 | Internal Server Error |
