.PHONY: install init-db train run test docker-build docker-run clean

install:
	pip install -e ".[dev]"

init-db:
	python -c "from src.database.connection import init_db; init_db()"

train:
	python scripts/train.py --ticker PETR4.SA --epochs 100

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

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
