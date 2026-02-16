# Unsupervised Marine Hidden Regime Discovery

Production-grade, config-driven pipeline for hidden regime discovery in multivariate marine time-series.

Author: [Rishabh Gokhe](https://github.com/rishabhgokhe)

## Features

- Data ingestion, validation, station-wise resampling
- Quality preprocessing (gap handling, directional encoding, outlier clipping)
- Sliding window feature generation
- Baseline unsupervised models: KMeans, GMM, optional HMM, optional change-point detection
- Optional deep segmentation scaffold: LSTM autoencoder + latent clustering
- Streamlit dashboard for interactive runs and visual diagnostics
- Artifact persistence for reproducibility
- MLflow experiment tracking (config controlled)
- CI + pytest + Docker + Makefile support

## Project Layout

```text
.
├── app.py
├── configs/
│   └── config.yml
├── data/raw/
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── logging_utils.py
│   │   └── tracking.py
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   │   ├── clustering.py
│   │   ├── hmm_model.py
│   │   ├── changepoint.py
│   │   └── deep_autoencoder.py
│   ├── pipeline/
│   └── visualization/
├── tests/
├── Dockerfile
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── requirements-deep.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
```

For deep model support:

```bash
pip install -r requirements-deep.txt
```

## Configuration

Main config: `configs/config.yml`

Key blocks:
- `data`: input path + schema + resampling
- `preprocess`: gap handling and clipping rules
- `features`: window settings
- `models`: state search + post-processing
- `tracking`: MLflow flags
- `deep`: LSTM autoencoder params

## Run Pipeline (CLI)

```bash
PYTHONPATH=. python3 -m src.pipeline.run_pipeline --config configs/config.yml --output outputs/latest
```

Generated artifacts:
- `processed_data.csv`
- `window_features.csv`
- `window_regimes.csv`
- `model_metrics.json`
- `quality_report.json`
- `changepoints.json` (if enabled dependency available)

## Run Streamlit

```bash
PYTHONPATH=. streamlit run app.py
```

Dashboard supports:
- config-path based runs
- CSV upload override
- deep model toggle (sidebar)
- metrics table, regime timeline, downloadable outputs

## MLflow Tracking

Enable in config:

```yaml
tracking:
  enabled: true
  tracking_uri: file:./mlruns
  experiment_name: marine-regime-discovery
  run_name: baseline-run
```

Then run CLI pipeline. Params, metrics and artifacts will be logged.

## Automation Commands

```bash
make install
make install-dev
make check
make test
make run
make app
```

## Docker

```bash
docker build -t marine-regime-app .
docker run --rm -p 8501:8501 marine-regime-app
```

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` runs:
- dependency install
- Python compile check
- pytest suite

## Phase Roadmap

1. Phase 1: Data and Foundation
2. Phase 2: Feature Engineering
3. Phase 3: Autoencoder and VAE Ablation
4. Phase 4: HMM Regime Modeling
5. Phase 5: Hierarchical Regime Grouping (Agglomerative)
6. Phase 6: Evaluation and Comparison
7. Phase 7: Streamlit App

## Notebook Workflow

Notebook files are available in `notebooks/`:
- `notebooks/00_data_ingest_qc.ipynb`
- `notebooks/01_eda_preprocess.ipynb`
- `notebooks/02_feature_engineering.ipynb`
- `notebooks/03_baseline_kmeans_gmm.ipynb`
- `notebooks/04_hmm_regimes.ipynb`
- `notebooks/05_autoencoder_vae_ablation.ipynb`
- `notebooks/06_hierarchical_regime_analysis.ipynb`
- `notebooks/07_evaluation_and_comparison.ipynb`
- `notebooks/08_streamlit_demo_checks.ipynb`
