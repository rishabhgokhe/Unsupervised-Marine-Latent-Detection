PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev test check run app docker-build docker-run

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

docker-build:
	docker build -t marine-regime-app .

docker-run:
	docker run --rm -p 8501:8501 marine-regime-app
