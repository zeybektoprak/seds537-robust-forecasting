#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  SEDS 537 — Robust Forecasting  |  Full Pipeline
#  Toprak Zeybek | 323011022
#  Usage:  bash run_all.sh
# ─────────────────────────────────────────────────────────────

set -e  # herhangi bir hata olursa dur

cd "$(dirname "$0")"  # script'in olduğu klasöre geç

echo "================================================="
echo "  SEDS 537 — Robust Forecasting  |  Full Run"
echo "================================================="

# 1. Veri
echo ""
echo "[1/5] Veri indiriliyor..."
python3 data/download_data.py

# 2. Eğitim
echo ""
echo "[2/5] Modeller eğitiliyor (ARIMA, RNN, LSTM, Transformer)..."
python3 experiments/run_baselines.py

# 3. Corruption deneyleri
echo ""
echo "[3/5] Corruption sweep çalışıyor..."
python3 experiments/run_corruption.py

# 4. Grafikler
echo ""
echo "[4/5] Grafikler üretiliyor..."
python3 experiments/plot_results.py

# 5. Hata analizi + Ablation
echo ""
echo "[5/5] Hata analizi ve ablation study..."
python3 experiments/error_analysis.py
python3 experiments/run_ablation.py

echo ""
echo "================================================="
echo "  TAMAMLANDI!"
echo "  Grafikler : results/figures/"
echo "  Metrikler : results/metrics/"
echo "  Modeller  : results/checkpoints/"
echo "================================================="
