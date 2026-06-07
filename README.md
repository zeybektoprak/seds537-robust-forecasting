# SEDS 537 — Robust Time-Series Forecasting
### Evaluating Forecasting Models under Noise and Anomaly Injection

**Student:** Toprak Zeybek — 323011022  
**Course:** SEDS 537 Machine Learning · İzmir Institute of Technology  
**Instructor:** Prof. Dr. Aytuğ Onan · Spring 2026  
**Dataset:** Jena Climate (2009–2016) — [download link](https://raw.githubusercontent.com/gilbutITbook/006975/master/datasets/jena_climate/jena_climate_2009_2016.csv)

---

## Overview

Time-series forecasting models are typically evaluated on clean data, yet real-world sensor streams
are often corrupted by noise and anomalies. This project asks: **how robust are modern forecasting
architectures when the test data is messy?**

Three baseline models are trained on clean data and evaluated under two synthetic corruption regimes.
A Temporal Transformer is proposed as the primary method, motivated by its global self-attention
mechanism which can "outvote" locally corrupted positions — a property that RNNs and LSTMs cannot
offer because they propagate hidden state sequentially.

---

## Models

| Role | Model | Key property |
|---|---|---|
| Baseline 1 | **ARIMA** | Classical generative model; immune to test-time input corruption |
| Baseline 2 | **Vanilla RNN** | Single recurrent layer (64 hidden units) |
| Baseline 3 | **Stacked LSTM** | 2 layers, 128 hidden, MC-Dropout uncertainty |
| **Proposed** | **Temporal Transformer** | Multi-head self-attention (4 heads, 2 layers, d_model=64) |

---

## Project Structure

```
seds537-robust-forecasting/
├── data/
│   ├── download_data.py           # one-off dataset download
│   └── raw/                       # jena_climate_2009_2016.csv
│
├── src/
│   ├── data_loader.py             # clean / normalise / window
│   ├── corruption.py              # Gaussian noise + point anomaly injection
│   ├── evaluate.py                # MAE / RMSE / MAPE
│   └── models/
│       ├── arima_model.py         # statsmodels ARIMA wrapper
│       ├── rnn_model.py           # PyTorch vanilla RNN
│       ├── lstm_model.py          # PyTorch stacked LSTM + MC-Dropout
│       └── transformer_model.py   # PyTorch Temporal Transformer (proposed)
│
├── experiments/
│   ├── run_baselines.py           # train all models on clean data
│   ├── run_corruption.py          # robustness study (core experiment)
│   ├── plot_results.py            # all publication figures
│   └── error_analysis.py         # failure case investigation
│
├── results/
│   ├── checkpoints/               # saved .pt model weights
│   ├── metrics/                   # JSON + CSV results
│   └── figures/                   # PNG figures
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1 — Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Download dataset

```bash
python data/download_data.py
```

---

## Running the Experiments

### Step 1 — Baselines + Proposed Method (clean data)

```bash
python experiments/run_baselines.py
```

Trains ARIMA, RNN, LSTM, and Transformer. Saves checkpoints and metrics.

### Step 2 — Corruption Study

```bash
python experiments/run_corruption.py
```

Tests all models under Gaussian noise (σ = 0–2) and point anomalies (ratio = 0–20%).  
Saves `corruption_results.json` and `corruption_results.csv`.

### Step 3 — Generate Figures

```bash
python experiments/plot_results.py
```

Produces 11 figures in `results/figures/`.

### Step 4 — Error Analysis

```bash
python experiments/error_analysis.py
```

Failure case investigation: error distributions, worst predictions, temporal patterns.

---

## Experimental Design

| Aspect | Choice |
|---|---|
| Input window | 120 hourly steps ≈ 5 days (subsampled from 10-min to hourly) |
| Forecast horizon | 1 step ahead |
| Normalisation | Z-score — fit on training set only (no leakage) |
| Split | 70 / 15 / 15 % chronological |
| Gaussian noise σ | 0.00, 0.25, 0.50, 1.00, 2.00 |
| Point anomaly ratio | 0.00, 0.01, 0.05, 0.10, 0.20 |
| Uncertainty method | MC-Dropout (n=50) on LSTM |

## Reproducibility

Progress is tracked via GitHub commits.  
All random seeds, hyper-parameters, and data preprocessing steps are fixed in code.  
Re-running all four experiment scripts from a fresh clone reproduces all figures and metrics.
