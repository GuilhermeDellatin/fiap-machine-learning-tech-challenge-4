# API Reference — Stock Prediction LSTM

## Informações Gerais

| Item | Valor |
|------|-------|
| Base URL | `http://localhost:8000` |
| Formato | JSON (`Content-Type: application/json`) |
| Autenticação | Não requerida |
| Documentação interativa | `http://localhost:8000/docs` (Swagger UI) |
| Schema OpenAPI | `http://localhost:8000/openapi.json` |

---

## Índice de Endpoints

| Grupo | Método | Rota | Status | Descrição |
|-------|--------|------|--------|-----------|
| Training | `POST` | `/api/v1/training/start` | 202 | Inicia treinamento em background |
| Training | `GET` | `/api/v1/training/status/{job_id}` | 200/404 | Status detalhado de um job |
| Training | `GET` | `/api/v1/training/jobs` | 200 | Lista jobs de treinamento |
| Training | `GET` | `/api/v1/training/models` | 200 | Lista modelos registrados |
| Training | `POST` | `/api/v1/training/activate/{version_id}` | 200/404 | Ativa um modelo específico |
| Prediction | `POST` | `/api/v1/predict` | 200/404 | Predição de preços futuros |
| Prediction | `POST` | `/api/v1/predict/batch` | 200 | Predição em lote para múltiplos tickers |
| Inference | `POST` | `/api/v1/inference` | 200/404 | Inferência direta sobre dados fornecidos |
| Inference | `POST` | `/api/v1/inference/batch` | 200/404 | Inferência em batch |
| Inference | `GET` | `/api/v1/inference/warmup` | 200 | Aquece o modelo em memória |
| Info | `GET` | `/api/v1/model/info/{ticker}` | 200/404 | Info do modelo ativo de um ticker |
| Cache | `GET` | `/api/v1/cache/info` | 200 | Informações do cache SQLite |
| Cache | `POST` | `/api/v1/cache/sync/{ticker}` | 200 | Força sincronização do cache |
| Cache | `DELETE` | `/api/v1/cache/{ticker}` | 200 | Remove cache de um ticker |
| Health | `GET` | `/health` | 200 | Health check da API |
| Monitoring | `GET` | `/metrics` | 200 | Métricas Prometheus |

---

## Códigos de Status HTTP

| Código | Significado | Quando ocorre |
|--------|-------------|---------------|
| `200 OK` | Sucesso | Requisição processada com sucesso |
| `202 Accepted` | Aceito | Job de treinamento aceito para processamento em background |
| `400 Bad Request` | Erro de validação | Campos inválidos, tipos errados ou regras de negócio violadas |
| `404 Not Found` | Não encontrado | Ticker sem modelo treinado, job_id inválido, version_id inválido |
| `422 Unprocessable Entity` | Erro de schema | JSON malformado ou campos obrigatórios ausentes |
| `500 Internal Server Error` | Erro interno | Falha inesperada no servidor |

---

## Formato de Erros

Todos os erros 4xx/5xx retornam um objeto JSON com `detail`:

```json
{
  "detail": "Mensagem de erro"
}
```

Erros 404 de modelo não encontrado retornam `detail` como objeto:

```json
{
  "detail": {
    "error": "No trained model found for ticker AAPL",
    "suggestion": "Train a model first using POST /api/v1/training/start",
    "ticker": "AAPL"
  }
}
```

---

## Grupo: Training

### POST /api/v1/training/start

Inicia um job de treinamento assíncrono para um ticker. O endpoint retorna imediatamente com HTTP 202 e o treinamento ocorre em background. Use `GET /api/v1/training/status/{job_id}` para acompanhar o progresso.

O treinamento também é rastreado automaticamente no MLflow (se disponível).

#### Request Body

```json
{
  "ticker": "PETR4.SA",
  "epochs": 100,
  "batch_size": 32,
  "sequence_length": 60,
  "hidden_size": 64,
  "num_layers": 2,
  "dropout": 0.2,
  "learning_rate": 0.001
}
```

#### Campos do Request

| Campo | Tipo | Obrigatório | Padrão | Restrições | Descrição |
|-------|------|-------------|--------|------------|-----------|
| `ticker` | `string` | Sim | — | 1–20 chars | Código do ativo financeiro. Convertido automaticamente para maiúsculas. Exemplos: `"PETR4.SA"`, `"AAPL"`, `"MSFT"` |
| `epochs` | `integer` | Não | `100` | 1–1000 | Número máximo de épocas de treinamento. O early stopping pode interromper antes de atingir esse valor |
| `batch_size` | `integer` | Não | `32` | 8–256 | Quantidade de amostras processadas por iteração do otimizador. Valores maiores aceleram o treino mas consomem mais memória |
| `sequence_length` | `integer` | Não | `60` | 10–200 | Janela de dias históricos que o modelo usa como contexto para cada predição. Equivale ao comprimento das sequências de entrada da LSTM |
| `hidden_size` | `integer` | Não | `64` | 16–512 | Número de neurônios em cada camada LSTM. Valores maiores aumentam a capacidade do modelo mas aumentam o risco de overfitting |
| `num_layers` | `integer` | Não | `2` | 1–5 | Quantidade de camadas LSTM empilhadas. Mais camadas capturam padrões mais abstratos |
| `dropout` | `float` | Não | `0.2` | 0.0–0.5 | Taxa de dropout aplicada entre as camadas LSTM (não aplicado se `num_layers=1`). Ajuda a regularizar o modelo |
| `learning_rate` | `float` | Não | `0.001` | 0.00001–0.1 | Taxa de aprendizado do otimizador Adam. Controla o tamanho dos passos na descida do gradiente |

#### Response 202 — Job aceito

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted",
  "message": "Training job started for PETR4.SA. Track in MLflow UI.",
  "ticker": "PETR4.SA"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `job_id` | `string` (UUID v4) | Identificador único do job. Use para consultar o status em `GET /api/v1/training/status/{job_id}` |
| `status` | `string` | Sempre `"accepted"` neste momento |
| `message` | `string` | Mensagem informativa com o ticker e o link ao MLflow |
| `ticker` | `string` | Ticker normalizado (maiúsculas) |

#### Response 404 — Ticker inválido

```json
{
  "detail": "Ticker INVALIDO not found in yfinance"
}
```

---

### GET /api/v1/training/status/{job_id}

Retorna o estado atual de um job de treinamento, incluindo progresso por época, perda atual e métricas finais quando concluído.

#### Path Parameter

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `job_id` | `string` (UUID) | ID retornado por `POST /api/v1/training/start` |

#### Response 200 — Job encontrado

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "ticker": "PETR4.SA",
  "status": "running",
  "progress_percent": 45.0,
  "epochs_completed": 45,
  "epochs_total": 100,
  "current_loss": 0.0023,
  "best_loss": 0.0019,
  "error_message": null,
  "model_version_id": null,
  "started_at": "2024-03-15T14:30:22Z",
  "completed_at": null
}
```

| Campo | Tipo | Nullable | Descrição |
|-------|------|----------|-----------|
| `job_id` | `string` | Não | UUID do job |
| `ticker` | `string` | Não | Código do ativo |
| `status` | `string` | Não | Estado atual: `"pending"` → `"running"` → `"completed"` ou `"failed"` |
| `progress_percent` | `float` | Não | Percentual de conclusão (0.0 a 100.0), calculado como `epochs_completed / epochs_total * 100` |
| `epochs_completed` | `integer` | Não | Épocas já processadas |
| `epochs_total` | `integer` | Não | Total de épocas configuradas |
| `current_loss` | `float` | Sim | Perda de validação da última época processada (MSE normalizado) |
| `best_loss` | `float` | Sim | Melhor perda de validação já atingida (base para early stopping) |
| `error_message` | `string` | Sim | Mensagem de erro truncada em 500 chars. Presente apenas quando `status="failed"` |
| `model_version_id` | `string` | Sim | `version_id` do modelo gerado (ex: `"PETR4.SA_20240315_143022"`). Presente apenas quando `status="completed"` |
| `started_at` | `datetime` | Sim | Timestamp ISO 8601 UTC de início efetivo do treino (após aceite do job) |
| `completed_at` | `datetime` | Sim | Timestamp ISO 8601 UTC de conclusão ou falha |

**Valores possíveis de `status`:**

| Valor | Descrição |
|-------|-----------|
| `"pending"` | Job criado, aguardando início do background task |
| `"running"` | Treino em andamento |
| `"completed"` | Treino concluído com sucesso. Modelo ativado e disponível para predições |
| `"failed"` | Treino falhou. Verificar `error_message` |

#### Response 404 — Job não encontrado

```json
{
  "detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

---

### GET /api/v1/training/jobs

Lista todos os jobs de treinamento com suporte a filtros.

#### Query Parameters

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
|-----------|------|-------------|--------|-----------|
| `ticker` | `string` | Não | — | Filtra por ticker (case-sensitive, usar maiúsculas) |
| `status` | `string` | Não | — | Filtra por status: `pending`, `running`, `completed`, `failed` |
| `limit` | `integer` | Não | `50` | Máximo de jobs retornados |

#### Exemplos de chamada

```
GET /api/v1/training/jobs
GET /api/v1/training/jobs?ticker=AAPL
GET /api/v1/training/jobs?status=completed&limit=10
GET /api/v1/training/jobs?ticker=PETR4.SA&status=failed
```

#### Response 200

```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "ticker": "PETR4.SA",
      "status": "completed",
      "created_at": "2024-03-15T14:30:00Z"
    },
    {
      "job_id": "660f9500-f30c-52e5-b827-557766551111",
      "ticker": "AAPL",
      "status": "running",
      "created_at": "2024-03-15T15:00:00Z"
    }
  ],
  "total": 2
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `jobs` | `array` | Lista de jobs (resumo) |
| `jobs[].job_id` | `string` | UUID do job |
| `jobs[].ticker` | `string` | Código do ativo |
| `jobs[].status` | `string` | Estado atual do job |
| `jobs[].created_at` | `datetime` | Timestamp ISO 8601 UTC de criação do job |
| `total` | `integer` | Total de jobs na lista retornada |

---

### GET /api/v1/training/models

Lista todos os modelos registrados no ModelRegistry (SQLite), com métricas e hiperparâmetros.

#### Query Parameters

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
|-----------|------|-------------|--------|-----------|
| `ticker` | `string` | Não | — | Filtra por ticker |
| `active_only` | `boolean` | Não | `false` | Se `true`, retorna apenas o modelo ativo de cada ticker |

#### Exemplos de chamada

```
GET /api/v1/training/models
GET /api/v1/training/models?ticker=AAPL
GET /api/v1/training/models?active_only=true
```

#### Response 200

```json
{
  "models": [
    {
      "version_id": "AAPL_20240315_143022",
      "ticker": "AAPL",
      "mae": 0.0042,
      "rmse": 0.0061,
      "mape": 1.83,
      "r2_score": 0.9712,
      "epochs_trained": 73,
      "hyperparameters": {
        "ticker": "AAPL",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "sequence_length": 60,
        "batch_size": 32,
        "epochs": 100
      },
      "is_active": true,
      "created_at": "2024-03-15T14:30:22Z"
    }
  ]
}
```

| Campo | Tipo | Nullable | Descrição |
|-------|------|----------|-----------|
| `version_id` | `string` | Não | Identificador único do modelo no formato `{TICKER}_{YYYYMMDD}_{HHMMSS}` |
| `ticker` | `string` | Não | Código do ativo |
| `mae` | `float` | Sim | Mean Absolute Error no conjunto de teste (escala normalizada) |
| `rmse` | `float` | Sim | Root Mean Square Error no conjunto de teste |
| `mape` | `float` | Sim | Mean Absolute Percentage Error em % no conjunto de teste |
| `r2_score` | `float` | Sim | Coeficiente de determinação R² (0 a 1; quanto maior, melhor) |
| `epochs_trained` | `integer` | Sim | Época em que o melhor modelo foi obtido (pode ser menor que `epochs` por early stopping) |
| `hyperparameters` | `object` | Sim | Dicionário com todos os hiperparâmetros usados no treinamento |
| `is_active` | `boolean` | Não | `true` se este é o modelo atualmente ativo para predições do ticker |
| `created_at` | `datetime` | Não | Timestamp ISO 8601 UTC de registro do modelo |

---

### POST /api/v1/training/activate/{version_id}

Ativa um modelo específico para um ticker, desativando o modelo anteriormente ativo. Útil para fazer rollback para uma versão anterior ou promover manualmente um modelo.

#### Path Parameter

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `version_id` | `string` | ID do modelo a ativar (ex: `"AAPL_20240315_143022"`) |

#### Response 200

```json
{
  "status": "activated",
  "version_id": "AAPL_20240315_143022",
  "ticker": "AAPL",
  "previous_active": "AAPL_20240310_091500"
}
```

| Campo | Tipo | Nullable | Descrição |
|-------|------|----------|-----------|
| `status` | `string` | Não | Sempre `"activated"` |
| `version_id` | `string` | Não | ID do modelo recém-ativado |
| `ticker` | `string` | Não | Ticker ao qual o modelo pertence |
| `previous_active` | `string` | Sim | ID do modelo que estava ativo antes. `null` se não havia modelo ativo |

#### Response 404 — Modelo não encontrado

```json
{
  "detail": "Model AAPL_20240101_000000 not found"
}
```

---

## Grupo: Prediction

### POST /api/v1/predict

Realiza predição de preços futuros para os próximos N dias úteis usando o modelo LSTM ativo do ticker. O modelo busca dados históricos recentes (último ano) e projeta os preços futuros.

> **Regra crítica:** Se não houver modelo ativo treinado para o ticker, retorna HTTP 404.

#### Request Body

```json
{
  "ticker": "PETR4.SA",
  "days_ahead": 5
}
```

| Campo | Tipo | Obrigatório | Padrão | Restrições | Descrição |
|-------|------|-------------|--------|------------|-----------|
| `ticker` | `string` | Sim | — | 1–20 chars | Código do ativo. Convertido automaticamente para maiúsculas |
| `days_ahead` | `integer` | Não | `1` | 1–30 | Quantidade de dias úteis futuros a predizer. Fins de semana são pulados automaticamente na resposta |

#### Response 200 — Predição realizada

```json
{
  "ticker": "PETR4.SA",
  "model_version": "PETR4.SA_20240315_143022",
  "predictions": [
    { "date": "2024-03-18", "price": 38.52 },
    { "date": "2024-03-19", "price": 38.74 },
    { "date": "2024-03-20", "price": 38.91 },
    { "date": "2024-03-21", "price": 39.08 },
    { "date": "2024-03-22", "price": 39.25 }
  ],
  "generated_at": "2024-03-15T16:45:00Z"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ticker` | `string` | Código do ativo |
| `model_version` | `string` | `version_id` do modelo utilizado |
| `predictions` | `array` | Lista de predições diárias ordenadas cronologicamente |
| `predictions[].date` | `string` | Data no formato `YYYY-MM-DD` (apenas dias úteis) |
| `predictions[].price` | `float` | Preço predito desnormalizado na moeda original do ativo (arredondado a 2 casas decimais) |
| `generated_at` | `datetime` | Timestamp ISO 8601 UTC de geração da resposta |

#### Response 404 — Sem modelo treinado

```json
{
  "detail": {
    "error": "No trained model found for ticker PETR4.SA",
    "suggestion": "Train a model first using POST /api/v1/training/start",
    "ticker": "PETR4.SA"
  }
}
```

---

### GET /api/v1/predict/{ticker}

Variante GET do endpoint de predição. Conveniente para testes rápidos via browser ou curl.

#### Path Parameter

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `ticker` | `string` | Código do ativo (ex: `AAPL`, `PETR4.SA`) |

#### Query Parameter

| Parâmetro | Tipo | Obrigatório | Padrão | Restrições | Descrição |
|-----------|------|-------------|--------|------------|-----------|
| `days_ahead` | `integer` | Não | `1` | 1–30 | Dias úteis a predizer |

#### Exemplos de chamada

```
GET /api/v1/predict/AAPL
GET /api/v1/predict/PETR4.SA?days_ahead=10
```

Response idêntico ao `POST /api/v1/predict`.

---

### POST /api/v1/predict/batch

Executa predições para múltiplos tickers em uma única requisição. Tickers sem modelo treinado são retornados na lista `failed` sem interromper as demais predições.

#### Request Body

```json
{
  "tickers": ["AAPL", "PETR4.SA", "MSFT"],
  "days_ahead": 3
}
```

| Campo | Tipo | Obrigatório | Padrão | Restrições | Descrição |
|-------|------|-------------|--------|------------|-----------|
| `tickers` | `array[string]` | Sim | — | Mínimo 1 item | Lista de códigos de ativos a predizer |
| `days_ahead` | `integer` | Não | `1` | 1–30 | Número de dias úteis a predizer para todos os tickers |

#### Response 200

```json
{
  "predictions": {
    "AAPL": {
      "ticker": "AAPL",
      "model_version": "AAPL_20240315_143022",
      "predictions": [
        { "date": "2024-03-18", "price": 182.45 },
        { "date": "2024-03-19", "price": 183.10 },
        { "date": "2024-03-20", "price": 183.75 }
      ],
      "generated_at": "2024-03-15T16:45:00Z"
    },
    "MSFT": {
      "ticker": "MSFT",
      "model_version": "MSFT_20240314_100000",
      "predictions": [
        { "date": "2024-03-18", "price": 415.30 },
        { "date": "2024-03-19", "price": 416.80 },
        { "date": "2024-03-20", "price": 418.20 }
      ],
      "generated_at": "2024-03-15T16:45:00Z"
    }
  },
  "failed": [
    {
      "ticker": "PETR4.SA",
      "error": "404: {'error': 'No trained model found for ticker PETR4.SA', ...}"
    }
  ],
  "generated_at": "2024-03-15T16:45:00Z"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `predictions` | `object` | Mapa de ticker → `PredictionResponse`. Contém apenas os tickers que tiveram sucesso |
| `failed` | `array` | Lista de tickers que falharam (sem modelo ou erro inesperado) |
| `failed[].ticker` | `string` | Código do ativo que falhou |
| `failed[].error` | `string` | Mensagem de erro |
| `generated_at` | `datetime` | Timestamp ISO 8601 UTC |

---

## Grupo: Inference

### POST /api/v1/inference

Inferência direta: você fornece os dados, o modelo processa e retorna uma predição imediata. Diferente do `/predict`, não busca dados históricos automaticamente — você controla o input.

Aceita dois modos mutuamente exclusivos:

- **`raw_prices`**: lista de preços de fechamento históricos na moeda original. O sistema normaliza internamente e retorna o preço desnormalizado.
- **`data`**: sequências já normalizadas no formato esperado pela LSTM (valores entre 0 e 1). Retorna valor normalizado ou desnormalizado conforme `return_normalized`.

> **Regra:** Exatamente um de `raw_prices` ou `data` deve ser fornecido.

#### Request Body — Modo raw_prices

```json
{
  "ticker": "PETR4.SA",
  "raw_prices": [36.5, 37.0, 37.8, 38.2, 38.5, 38.1, 37.9]
}
```

#### Request Body — Modo data (normalizado)

```json
{
  "ticker": "PETR4.SA",
  "data": [
    [0.72], [0.73], [0.75], [0.76], [0.77], [0.76], [0.75]
  ],
  "return_normalized": false
}
```

#### Campos do Request

| Campo | Tipo | Obrigatório | Padrão | Descrição |
|-------|------|-------------|--------|-----------|
| `ticker` | `string` | Sim | — | Código do ativo. Convertido para maiúsculas internamente |
| `raw_prices` | `array[float]` | Condicional | `null` | Lista de preços de fechamento históricos na moeda original (ex: USD, BRL). Mínimo recomendado: `sequence_length` (padrão 60) valores |
| `data` | `array[array[float]]` | Condicional | `null` | Sequência normalizada no formato `[[v1], [v2], ...]` onde cada sublista tem um único valor float entre 0 e 1. Comprimento deve ser `sequence_length` (padrão 60) |
| `return_normalized` | `boolean` | Não | `false` | Somente aplicável ao modo `data`. Se `true`, retorna a predição no espaço normalizado [0, 1] sem desnormalizar |

#### Response 200

```json
{
  "ticker": "PETR4.SA",
  "prediction": 38.91,
  "is_normalized": false,
  "model_version": "PETR4.SA_20240315_143022",
  "inference_time_ms": 4.73
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ticker` | `string` | Código do ativo (maiúsculas) |
| `prediction` | `float` | Valor predito. Se `is_normalized=false`: preço na moeda original. Se `is_normalized=true`: valor no intervalo [0, 1] do MinMaxScaler |
| `is_normalized` | `boolean` | Indica se `prediction` está no espaço normalizado. `false` no modo `raw_prices` ou quando `return_normalized=false` |
| `model_version` | `string` | `version_id` do modelo utilizado |
| `inference_time_ms` | `float` | Tempo de processamento em milissegundos (arredondado a 2 casas decimais) |

#### Response 404 — Sem modelo

```json
{
  "detail": "No model for PETR4.SA"
}
```

#### Response 400 — Validação dos dados

```json
{
  "detail": [
    {
      "type": "value_error",
      "msg": "Value error, Must provide 'data' or 'raw_prices'"
    }
  ]
}
```

---

### POST /api/v1/inference/batch

Executa inferência em lote: múltiplas sequências normalizadas são processadas simultaneamente pelo modelo em uma única passagem forward, maximizando eficiência.

> Todas as sequências devem ser para o mesmo ticker.

#### Request Body

```json
{
  "ticker": "AAPL",
  "sequences": [
    [[0.72], [0.73], [0.75], [0.76], [0.77], [0.76], [0.75]],
    [[0.80], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86]],
    [[0.65], [0.66], [0.67], [0.68], [0.69], [0.70], [0.71]]
  ]
}
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `ticker` | `string` | Sim | Código do ativo |
| `sequences` | `array[array[array[float]]]` | Sim | Lista de sequências. Cada sequência é `[[v1], [v2], ..., [vN]]` com shape `(sequence_length, 1)`. Os valores devem estar normalizados no intervalo [0, 1] |

#### Response 200

```json
{
  "ticker": "AAPL",
  "predictions": [0.7823, 0.8902, 0.7012],
  "batch_size": 3,
  "inference_time_ms": 8.41
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ticker` | `string` | Código do ativo |
| `predictions` | `array[float]` | Lista de predições normalizadas [0, 1], uma por sequência de entrada, na mesma ordem |
| `batch_size` | `integer` | Número de sequências processadas |
| `inference_time_ms` | `float` | Tempo total de inferência em milissegundos |

---

### GET /api/v1/inference/warmup

Executa uma inferência com dados sintéticos (dummy tensor) para garantir que o modelo está carregado em memória e os pesos estão nos kernels da GPU/CPU. Útil em cenários de health check ou antes de SLAs críticos de latência.

#### Response 200 — Modelo aquecido

```json
{
  "status": "warmed_up",
  "inference_time_ms": 3.12,
  "device": "cpu"
}
```

#### Response 200 — Sem modelo carregado

```json
{
  "status": "no_model_loaded",
  "inference_time_ms": 0,
  "device": "unknown"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `status` | `string` | `"warmed_up"` se o modelo foi aquecido com sucesso; `"no_model_loaded"` se nenhum modelo está em memória |
| `inference_time_ms` | `float` | Tempo da inferência dummy em ms. `0` quando sem modelo |
| `device` | `string` | Dispositivo em uso: `"cpu"`, `"cuda"`, `"cuda:0"` etc. `"unknown"` quando sem modelo |

---

## Grupo: Informações e Cache

### GET /api/v1/model/info/{ticker}

Retorna informações detalhadas do modelo ativo para um ticker.

#### Path Parameter

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `ticker` | `string` | Código do ativo (case-insensitive; convertido para maiúsculas internamente) |

#### Response 200

```json
{
  "ticker": "AAPL",
  "active_model": {
    "version_id": "AAPL_20240315_143022",
    "mae": 0.0042,
    "rmse": 0.0061,
    "mape": 1.83,
    "r2_score": 0.9712,
    "epochs_trained": 73,
    "created_at": "2024-03-15T14:30:22Z"
  }
}
```

| Campo | Tipo | Nullable | Descrição |
|-------|------|----------|-----------|
| `ticker` | `string` | Não | Código do ativo |
| `active_model.version_id` | `string` | Não | ID único do modelo ativo |
| `active_model.mae` | `float` | Sim | Mean Absolute Error no teste |
| `active_model.rmse` | `float` | Sim | Root Mean Square Error no teste |
| `active_model.mape` | `float` | Sim | Mean Absolute Percentage Error (%) no teste |
| `active_model.r2_score` | `float` | Sim | R² no teste (quanto mais próximo de 1, melhor) |
| `active_model.epochs_trained` | `integer` | Sim | Época do melhor checkpoint |
| `active_model.created_at` | `datetime` | Não | Timestamp de criação do modelo |

#### Response 404

```json
{
  "detail": {
    "error": "No model found for ticker AAPL",
    "suggestion": "Train a model using POST /api/v1/training/start"
  }
}
```

---

### GET /api/v1/cache/info

Retorna informações sobre os dados históricos em cache no SQLite. Útil para verificar se os dados estão atualizados antes de iniciar um treinamento.

#### Query Parameter

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `ticker` | `string` | Não | Filtra por ticker. Se omitido, retorna lista vazia (comportamento atual) |

#### Exemplo de chamada

```
GET /api/v1/cache/info?ticker=AAPL
```

#### Response 200

```json
{
  "caches": [
    {
      "ticker": "AAPL",
      "records_count": 1258,
      "date_range": {
        "start": "2019-03-15",
        "end": "2024-03-15"
      },
      "last_updated": "2024-03-15T14:30:00Z",
      "is_valid": true,
      "expires_in_hours": 21.5
    }
  ]
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `caches` | `array` | Lista de entradas de cache |
| `caches[].ticker` | `string` | Código do ativo |
| `caches[].records_count` | `integer` | Total de registros OHLCV no SQLite para este ticker |
| `caches[].date_range` | `object` | Intervalo de datas disponíveis no cache |
| `caches[].date_range.start` | `string` | Data mais antiga (`YYYY-MM-DD`) |
| `caches[].date_range.end` | `string` | Data mais recente (`YYYY-MM-DD`) |
| `caches[].last_updated` | `datetime` | Timestamp ISO 8601 UTC da última atualização do cache |
| `caches[].is_valid` | `boolean` | `true` se `(now - last_updated) < CACHE_EXPIRY_HOURS` (padrão: 24h) |
| `caches[].expires_in_hours` | `float` | Horas restantes até o cache expirar. `0` se já expirado |

---

### POST /api/v1/cache/sync/{ticker}

Força sincronização imediata do cache com o yfinance, independentemente da validade atual. Baixa dados históricos (padrão: 5 anos) e atualiza a tabela `price_cache`.

#### Path Parameter

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `ticker` | `string` | Código do ativo a sincronizar |

#### Response 200

```json
{
  "ticker": "AAPL",
  "records_updated": 1258,
  "message": "Cache synchronized successfully"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ticker` | `string` | Código do ativo sincronizado |
| `records_updated` | `integer` | Total de registros no cache após a sincronização |
| `message` | `string` | Mensagem de confirmação |

---

### DELETE /api/v1/cache/{ticker}

Remove registros de cache do SQLite para um ticker específico ou para todos os tickers.

#### Path Parameter

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `ticker` | `string` | Código do ativo a limpar. Use `all` para remover o cache de todos os tickers |

#### Exemplos de chamada

```
DELETE /api/v1/cache/AAPL
DELETE /api/v1/cache/all
```

#### Response 200

```json
{
  "deleted_records": 1258,
  "ticker": "AAPL"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `deleted_records` | `integer` | Número de registros removidos do banco |
| `ticker` | `string` | Ticker informado (ou `"all"`) |

---

## Grupo: Health & Monitoring

### GET /health

Verifica a disponibilidade da API e a conectividade com o banco de dados SQLite.

#### Response 200 — Saudável

```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2024-03-15T16:45:00Z"
}
```

#### Response 200 — Degradado (banco inacessível)

```json
{
  "status": "unhealthy",
  "database": "unable to open database file",
  "timestamp": "2024-03-15T16:45:00Z"
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `status` | `string` | `"healthy"` se banco acessível; `"unhealthy"` caso contrário |
| `database` | `string` | `"connected"` ou mensagem de erro do SQLAlchemy |
| `timestamp` | `datetime` | Timestamp ISO 8601 UTC do momento do check |

> Retorna sempre HTTP 200 mesmo quando `status="unhealthy"` — o campo `status` no body deve ser verificado pelo orquestrador.

---

### GET /metrics

Retorna métricas operacionais no formato texto do Prometheus (exposition format). Consumido pelo Prometheus Server em `http://localhost:9090`.

#### Response 200

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/api/v1/predict",method="POST",status="200"} 142.0

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{endpoint="/api/v1/predict",method="POST",le="0.005"} 10.0
...

# HELP model_predictions_total Total predictions made
# TYPE model_predictions_total counter
model_predictions_total{ticker="AAPL"} 87.0

# HELP model_inference_duration_seconds Model inference duration
# TYPE model_inference_duration_seconds histogram
model_inference_duration_seconds_bucket{ticker="AAPL",le="0.005"} 85.0
...

# HELP training_jobs_total Training jobs
# TYPE training_jobs_total counter
training_jobs_total{status="completed"} 5.0
training_jobs_total{status="failed"} 1.0

# HELP cache_hits_total Cache hits
# TYPE cache_hits_total counter
cache_hits_total{ticker="AAPL"} 230.0

# HELP cache_misses_total Cache misses
# TYPE cache_misses_total counter
cache_misses_total{ticker="AAPL"} 3.0
```

**Métricas disponíveis:**

| Métrica | Tipo | Labels | Descrição |
|---------|------|--------|-----------|
| `http_requests_total` | Counter | `method`, `endpoint`, `status` | Total de requisições HTTP por rota e código de status |
| `http_request_duration_seconds` | Histogram | `method`, `endpoint` | Distribuição de latência das requisições |
| `model_predictions_total` | Counter | `ticker` | Total de predições realizadas por ticker |
| `model_inference_duration_seconds` | Histogram | `ticker` | Distribuição de latência de inferência por ticker |
| `training_jobs_total` | Counter | `status` | Total de jobs de treinamento por status final |
| `training_duration_seconds` | Histogram | — | Distribuição de duração dos treinamentos |
| `cache_hits_total` | Counter | `ticker` | Acessos ao cache SQLite que encontraram dados válidos |
| `cache_misses_total` | Counter | `ticker` | Acessos ao cache que precisaram buscar no yfinance |

---

## Exemplos Práticos

### Fluxo completo: treinar e predizer

```bash
# 1. Iniciar treinamento
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "epochs": 50}'

# Resposta: {"job_id": "550e8400-...", "status": "accepted", ...}

# 2. Acompanhar progresso
curl http://localhost:8000/api/v1/training/status/550e8400-e29b-41d4-a716-446655440000

# 3. Quando status="completed", predizer
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days_ahead": 5}'
```

### Verificar cache antes de treinar

```bash
# Ver se há dados em cache (evita download desnecessário)
curl "http://localhost:8000/api/v1/cache/info?ticker=AAPL"

# Forçar atualização do cache
curl -X POST http://localhost:8000/api/v1/cache/sync/AAPL
```

### Rollback para versão anterior

```bash
# Listar modelos disponíveis
curl "http://localhost:8000/api/v1/training/models?ticker=AAPL"

# Ativar versão anterior
curl -X POST http://localhost:8000/api/v1/training/activate/AAPL_20240310_091500
```

### Inferência direta com preços brutos

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "raw_prices": [182.1, 183.5, 181.9, 184.2, 185.0, 184.7, 183.8]
  }'
```