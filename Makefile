.PHONY: help install test benchmark serve dev tunnel reproduce clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn

help:
	@echo "OptiServe Developer Workflow Targets:"
	@echo "  make install    - Initialize venv and install dependencies via uv"
	@echo "  make test       - Execute automated test suite (15 tests)"
	@echo "  make benchmark  - Run 5-way quantization bake-off on RTX 4060"
	@echo "  make serve      - Launch FastAPI async continuous batching microservice"
	@echo "  make dev        - Launch Next.js 14 Interactive Studio frontend"
	@echo "  make tunnel     - Open Cloudflare Tunnel to expose local GPU engine"
	@echo "  make reproduce  - Execute full benchmark, tests, and build"
	@echo "  make clean      - Clean cache and temporary build artifacts"

install:
	uv venv --python 3.11 .venv
	uv pip install -e ".[dev]"
	cd frontend && npm install

test:
	$(PYTEST) tests/ -v --durations=10

benchmark:
	$(PYTHON) -m optiserve.benchmarks.quant_bakeoff

serve:
	$(UVICORN) optiserve.api.server:app --host 0.0.0.0 --port 8000 --reload

dev:
	cd frontend && npm run dev

tunnel:
	cloudflared tunnel --url http://localhost:8000

reproduce:
	$(PYTHON) -m optiserve.benchmarks.latency_profiler
	$(PYTHON) -m optiserve.benchmarks.quant_bakeoff
	$(PYTEST) tests/ -v
	cd frontend && npm run build

clean:
	rm -rf .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf data/cache/airllm_shards
