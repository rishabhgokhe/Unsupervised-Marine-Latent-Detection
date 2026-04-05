# Unsupervised Marine Latent Regime Detection

Final-year CSE project (7th semester) focused on discovering hidden operational regimes in multivariate marine time-series. This repository includes a full offline pipeline, a Streamlit inference dashboard, and a research packaging system for experiment tracking.

If you read this README once, you should understand exactly what the project does, why each component exists, and how the outputs are produced.

---

**Problem Statement**

Marine telemetry is high-frequency, noisy, and multi-dimensional. The goal is to learn latent “regimes” (operational or environmental states) without labels, and to expose those regimes in a form that supports monitoring, risk detection, and downstream decisions.

---

**High-Level Pipeline**

1. Ingest CSV/Parquet telemetry.
2. Station-wise resampling and quality preprocessing.
3. Sliding-window or multiscale feature extraction.
4. Standardization.
5. Baseline clustering (KMeans, GMM).
6. Optional HMM and changepoint diagnostics.
7. Optional deep latent models (dense AE, LSTM AE, VAE ablation).
8. Hierarchical regime mapping (micro -> macro).
9. Artifact packaging for reproducible inference.

---

**Project Layout**

```text
.
├── app.py                      # Streamlit entrypoint
├── app/                        # UI, inference, visualization helpers
├── configs/                    # Config presets
├── data/                       # Sample datasets
├── experiments/                # Research outputs
├── src/                        # Pipeline, models, features, evaluation
├── tests/                      # Pytest suite
└── requirements*.txt
```

---

**Installation**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dev + tests:

```bash
pip install -r requirements-dev.txt
```

Note:
- `requirements.txt` includes `requirements-deep.txt`, so PyTorch is installed by default.

---

**Data Expectations**

Supported formats:
- `.csv`
- `.parquet`

Required schema is defined in config:
- `timestamp_col` (e.g., `DATE`)
- `station_col` (e.g., `STATION`)
- `numeric_columns` (e.g., `WIND_SPEED`, `SEA_LVL_PRES`, `WAVE_HGT`)
- `directional_columns` (e.g., `WIND_DIR`, `WAVE_DIR`, `SWELL_DIR`)

Sentinel values replaced with NaN:
- `99`, `999`, `9999`, `99999`

Sample data included in repo:
- `data/raw/real data.csv`
- `data/raw/synthetic_marine_extended_unlabelled_dataset.csv`
- `data/raw/merged final.parquet`

---

**Preprocessing Details**

Station-wise resampling:
- Numeric columns: arithmetic mean per resample interval.
- Directional columns: circular mean.

Circular mean for a directional column `d`:

```text
theta = deg2rad(d)
mean_dir = atan2(mean(sin(theta)), mean(cos(theta)))
```

Missing value handling:
- Forward/backward fill for short gaps (`small_gap_limit`).
- Linear interpolation for medium gaps (`medium_gap_limit`).
- Optional drop for large gaps (`drop_large_gap_rows`).

Directional encoding:
- For each directional feature `DIR`, add `DIR_SIN` and `DIR_COS` after interpolation.

Outlier clipping:
- For each numeric column `x`, clamp to quantiles:

```text
low = quantile(x, clip_quantile_low)
high = quantile(x, clip_quantile_high)
x = clip(x, low, high)
```

---

**Windowing and Feature Engineering**

Windowing is done per station with past-only windows.

Two modes:
- Single-scale sliding windows (`build_sliding_windows`).
- Multiscale windows (`generate_multiscale_window_features`).

For each window, per feature, we compute:
- `mean`
- `std`
- `trend` = `last_value - first_value`
- `energy` = `sum(x^2) / window_length`
- `max`
- `min`

Multiscale windows:
- Multiple window sizes are computed.
- Feature names include scale prefix, e.g., `WIND_SPEED_s24_mean`.
- All scales are concatenated at the same window end index.

---

**Scaling**

All window features are standardized with `StandardScaler`:

```text
z = (x - mean) / std
```

This scaled matrix is used by all clustering and HMM steps.

---

**Models and Why They Are Used**

**KMeans**
- Purpose: simple baseline to capture spherical clusters.
- Selection: choose `k` maximizing silhouette.

Silhouette for sample `i`:

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Gaussian Mixture Model (GMM)**
- Purpose: soft clustering with ellipsoidal clusters.
- Selection: choose `k` minimizing BIC.
- AIC is recorded for diagnostics.

**Hidden Markov Model (HMM)**
- Purpose: temporal regime modeling with transition structure.
- Model: GaussianHMM with diagonal covariance.
- Selection: choose `k` minimizing BIC approximation.

BIC approximation:

```text
BIC = -2 * logL + p * ln(n)
p = n_states^2 + 2 * n_states * n_features
```

**Changepoint Detection**
- Purpose: detect statistical regime shifts independently of clustering.
- Method: PELT algorithm with RBF cost (`ruptures`).
- Aligns change points with regime boundaries for diagnostics.

**Dense Autoencoder (AE)**
- Purpose: learn compact latent embedding for non-linear structure.
- Architecture:
  - Encoder: Linear -> ReLU -> Linear -> ReLU -> Linear(latent)
  - Decoder: Linear -> ReLU -> Linear -> ReLU -> Linear(input)
- Loss: MSE reconstruction loss.
- Outputs:
  - Latent embeddings
  - Reconstruction error per window

Dense AE outputs are used for:
- KMeans in latent space.
- HMM in latent space.

**LSTM Autoencoder**
- Purpose: capture sequential dynamics in window sequences.
- Encoder: LSTM over sequences.
- Latent: last hidden state projected to latent space.
- Decoder: LSTM driven by zero input to reconstruct.

**Variational Autoencoder (VAE)**
- Purpose: probabilistic latent structure for ablation study.
- Encoder outputs `mu` and `logvar`.
- Reparameterization:

```text
z = mu + eps * exp(0.5 * logvar)
```

- Loss:

```text
loss = recon_mse + beta * KL
```

Where:

```text
KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
```

---

**Hierarchical Regime Mapping (Micro -> Macro)**

Two stages:
- Micro regimes from base model (e.g., HMM).
- Macro regimes formed by clustering micro-state means.

Macro mapping:
- Use KMeans over micro-state means.
- Map each micro state to one macro state.

---

**Diagnostics and Evaluation**

**Cluster Quality**
- Silhouette score.
- Davies-Bouldin index.

**Temporal Diagnostics**
- Transition matrix (row-normalized).
- Transition entropy:

```text
H = mean( -sum(p * log(p)) ) over rows
```

**Regime Durations**
- Mean, median, p90, min, max.

**HMM Seed Stability**
- Fit HMM with multiple random seeds.
- Compare label consistency using ARI.

ARI (Adjusted Rand Index) is used to measure label agreement.

**Changepoint Alignment**
- Convert regime labels to boundaries.
- Measure boundary precision/recall with tolerance.

**Pressure Drop Analysis**
- For `SEA_LVL_PRES` mean feature, compute post-transition drop vs pre-transition mean.

---

**Artifacts Produced**

Saved to the output directory (default `outputs/latest`):

- `processed_data.csv`: cleaned and resampled data.
- `window_features.csv`: feature matrix for windows.
- `feature_scaler.pkl`: fitted `StandardScaler`.
- `window_regimes.csv`: labels for each window.
- `model_metrics.json`: metrics by model.
- `quality_report.json`: missingness and clipping.
- `model_diagnostics.json`: diagnostics (entropy, transitions, stability).
- `changepoints.json`: if changepoints enabled.
- `dense_autoencoder_config.json`: dense AE hyperparameters.
- `autoencoder_dense.pt`: dense AE weights (PyTorch).
- `dense_latent_projection.csv`: PCA of latent space.
- `macro_mapping.pkl`: micro -> macro mapping.
- `macro_regime_characterization.json`: macro regime profiles.
- `hmm.pkl`: trained HMM model.
- `inference_config.json`: columns + windowing details for inference.
- `config.yaml`: copy of config used.

The Streamlit app expects inference artifacts under `artifacts/latest` by default.

---

**Inference Logic**

Inference runs without retraining:

1. Load artifacts:
   - `feature_scaler.pkl`
   - `hmm.pkl`
   - Optional dense AE artifacts
2. Align input columns to config + artifacts.
3. Apply the same windowing as training (from `inference_config.json`).
4. Scale using saved scaler.
5. Compute latent embeddings if dense AE is available.
6. Predict HMM micro states.
7. Map to macro states if mapping exists.

---

**Streamlit App**

File: `app/app.py`

Main features:
- Upload CSV/Parquet or use bundled sample dataset.
- Dashboard metrics for rows, windows, state counts, and duration.
- Visual diagnostics: timeline scatter, distributions, transition heatmaps.
- Early warning signals and operational planning summaries.
- Download labeled datasets and reports.
- Project Docs view (renders this README).

---

**GPU Acceleration**

PyTorch components use GPU automatically if CUDA is available. You can override with:

```bash
export UMDL_DEVICE=auto   # auto|cuda|cpu
```

GPU-accelerated components:
- Dense autoencoder training and inference.
- LSTM autoencoder training.
- VAE training.

CPU-only components:
- KMeans, GMM, HMM, changepoint detection.

---

**Research Packaging**

Run a single experiment:

```bash
PYTHONPATH=. python3 -m src.research.run_research_packaging run-experiment \
  --config configs/config_research.yml \
  --experiment-id exp_baseline \
  --output-root experiments
```

Build comparative tables:

```bash
PYTHONPATH=. python3 -m src.research.run_research_packaging build-comparative \
  --experiments-dir experiments
```

Outputs:
- `experiments/<id>/experiment.json`
- `experiments/<id>/comparative_rows.csv`
- `experiments/comparative_results.csv`
- `experiments/comparative_results_mean_by_model.csv`

---

**Configuration Reference**

Configs live in:
- `configs/config.yml`
- `configs/config_major.yml`
- `configs/config_research.yml`

Important fields:
- `data.input_path`: dataset path.
- `data.timestamp_col`, `data.station_col`.
- `data.numeric_columns`, `data.directional_columns`.
- `data.resample_rule`: pandas resample frequency (e.g., `1h`).
- `preprocess.small_gap_limit`, `preprocess.medium_gap_limit`, `preprocess.drop_large_gap_rows`.
- `preprocess.clip_quantile_low`, `preprocess.clip_quantile_high`.
- `features.window_size`, `features.step_size`.
- `features.use_multiscale`, `features.multi_window_sizes`, `features.multi_stride`.
- `features.rolling_features`: subset of stats to keep.
- `models.candidate_states`: candidate k values.
- `models.hmm_covariance_type`, `models.min_segment_length`, `models.n_super_regimes`.
- `deep.enabled`, `deep.enable_dense_ae`, `deep.enable_vae`.
- `deep.*`: latent sizes, epochs, batch size, learning rate.
- `tracking.*`: MLflow settings.

---

**Tests**

```bash
PYTHONPATH=. pytest -q
```

---

**Troubleshooting**

- HMM skipped: ensure `hmmlearn` is installed.
- Changepoints skipped: ensure `ruptures` is installed.
- Deep models disabled: set `deep.enabled: true` and flags for AE/VAE.
- Column mismatch: check `inference_config.json` and input CSV headers.

---

**Author**

Rishabh Gokhe
