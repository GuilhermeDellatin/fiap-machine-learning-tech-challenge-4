# Arquitetura

## Visao Geral

```
stock-prediction-lstm/
├── src/
│   ├── api/           # FastAPI endpoints
│   ├── database/      # SQLAlchemy + SQLite
│   ├── data/          # Coleta e preprocessamento
│   ├── models/        # LSTM PyTorch
│   └── monitoring/    # Prometheus
├── scripts/           # CLI tools
└── docker/            # Containerizacao
```

## Fluxo de Dados

### Treinamento

```
1. POST /training/start
   |
2. BackgroundTask iniciado
   |
3. sync_data() -> yfinance -> SQLite
   |
4. Preprocessor -> Sequences
   |
5. LSTM Training
   |
6. Save .pt + .joblib
   |
7. Register in ModelRegistry
   |
8. Activate model
```

### Predicao

```
1. POST /predict
   |
2. Check ModelRegistry (404 se nao existe)
   |
3. Load model (se necessario)
   |
4. download_data() (cache check)
   |
5. Predict
   |
6. Return predictions
```

## Cache SQLite

### Tabelas

- **PriceCache**: Dados do yfinance (ticker, date, OHLCV, updated_at)
- **TrainingJob**: Jobs de treinamento (job_id, status, progress)
- **ModelRegistry**: Modelos treinados (version_id, paths, metrics, is_active)

### Regra de Expiracao

```python
is_valid = (now - updated_at) < CACHE_EXPIRY_HOURS
```

Antes de baixar do yfinance, o sistema verifica:
1. Se dados existem no SQLite (tabela PriceCache)
2. Se o campo `updated_at` indica cache valido (< 24h por padrao)
3. Se expirado, baixa novamente e atualiza o cache

## Modelo LSTM

```
Input (batch, seq_len, 1)
    |
LSTM (num_layers, hidden_size)
    |
FC Layer
    |
Output (batch, 1)
```

### Parametros Padrao

| Parametro | Valor |
|-----------|-------|
| sequence_length | 60 |
| hidden_size | 64 |
| num_layers | 2 |
| epochs | 100 |
| learning_rate | 0.001 |

### Pos-Treinamento

Apos treinamento bem-sucedido:
1. Salva modelo: `models/{ticker}_{timestamp}.pt`
2. Salva scaler: `models/{ticker}_{timestamp}_scaler.joblib`
3. Registra no ModelRegistry com metricas
4. Seta `is_active=True` para o novo modelo
5. Seta `is_active=False` para modelo anterior do mesmo ticker

## Stack Tecnologica

| Componente | Tecnologia |
|------------|-----------|
| API | FastAPI |
| ML Framework | PyTorch |
| Database | SQLite + SQLAlchemy |
| Dados | yfinance |
| Metricas | Prometheus |
| Container | Docker |
