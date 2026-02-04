.PHONY: install init-db train train-quick evaluate run test docker-build docker-run clean

install:
	pip install -e ".[dev]"

init-db:
	python -c "from src.database.connection import init_db; init_db()"

train:
	python scripts/train.py --ticker PETR4.SA --epochs 100

train-quick:
	python scripts/train.py --ticker AAPL --epochs 5 --no-sync

evaluate:
	python scripts/evaluate.py --ticker PETR4.SA

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --log-level info

test:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

docker-build:
	docker build -f docker/Dockerfile -t stock-prediction-lstm .

docker-run:
	docker-compose -f docker/docker-compose.yml up

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
