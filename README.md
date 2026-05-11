# seds537-robust-forecasting

A robustness study for time-series forecasting models using the
[Jena Climate dataset](https://www.bgc-jena.mpg.de/wetter/).  
Three baseline models (ARIMA, Vanilla RNN, Stacked LSTM) are trained on clean
data, then evaluated under synthetic corruptions (Gaussian noise and point
anomalies) to measure degradation in forecast accuracy.

---

## Project structure

```
seds537-robust-forecasting/
├── data/
│   ├── download_data.py      # one-off dataset download script
│   └── raw/                  # CSV lives here after download
├── src/
│   ├── data_loader.py        # loading, cleaning, normalisation, windowing
│   ├── corruption.py         # Gaussian noise & point-anomaly injection
│   ├── evaluate.py           # MAE / RMSE / MAPE metrics
│   └── models/
│       ├── arima_model.py    # statsmodels ARIMA wrapper
│       ├── rnn_model.py      # PyTorch vanilla RNN
│       └── lstm_model.py     # PyTorch stacked LSTM + MC-Dropout
├── experiments/
│   └── run_baselines.py      # main experiment entry point
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone / navigate to the project root.**

**2. (Optional) create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Download the dataset:**

```bash
python data/download_data.py
```

This saves `jena_climate_2009_2016.csv` to `data/raw/`.

---

## Running experiments

### Baseline evaluation (clean data)

```bash
python experiments/run_baselines.py
```

This will:
1. Load and preprocess the Jena Climate CSV.
2. Train ARIMA, RNN, and LSTM on the training split.
3. Evaluate each model on the held-out test split.
4. Print a results table with MAE, RMSE, and MAPE.
5. Run an MC-Dropout uncertainty pass with the LSTM.

---

## Key design decisions

| Aspect | Choice |
|---|---|
| Normalisation | Z-score, fit on training set only |
| Input window | 720 steps (~5 days at 10-min intervals) |
| Forecast horizon | 1 step ahead |
| Train / val / test split | 70 / 15 / 15 % (chronological) |
| LSTM uncertainty | Monte-Carlo Dropout (`n_samples=50`) |

---

## Extending the study

- Add corruption experiments by importing `src/corruption.py` functions and
  passing corrupted arrays to `compute_metrics`.
- Swap the target column (default `T (degC)`) by editing the `target_col`
  argument in `load_and_preprocess`.
- Tune model hyper-parameters at the top of `experiments/run_baselines.py`.
