PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev test check run app docker-build docker-run research-exp research-ablation research-robustness research-bundle

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test:
	PYTHONPATH=. pytest -q

check:
	PYTHONPATH=. $(PYTHON) -m compileall src app app.py
	PYTHONPATH=. pytest -q

run:
	PYTHONPATH=. $(PYTHON) -m src.pipeline.run_pipeline --config configs/config.yml --output outputs/latest

app:
	PYTHONPATH=. streamlit run app.py

research-exp:
	PYTHONPATH=. $(PYTHON) -m src.research.run_research_packaging run-experiment --config configs/config.yml --experiment-id exp_baseline --output-root experiments

research-ablation:
	PYTHONPATH=. $(PYTHON) -m src.research.run_research_packaging run-ablation --config configs/config.yml --output-root experiments

research-robustness:
	PYTHONPATH=. $(PYTHON) -m src.research.run_research_packaging run-robustness --config configs/config.yml --output-root experiments

research-bundle:
	PYTHONPATH=. $(PYTHON) -m src.research.run_research_packaging bundle --experiments-dir experiments

docker-build:
	docker build -t marine-regime-app .

docker-run:
	docker run --rm -p 8501:8501 marine-regime-app
