#GridGuard-AI ⚡
### Energy Theft Detection Using LSTM Autoencoder & SHAP Explainability

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Power distribution companies lose crores of rupees annually to electricity theft, but detection is still largely manual and reactive. **GridGuard-AI** is an end-to-end AI system that continuously monitors smart meter consumption data, learns what normal usage looks like, and automatically flags suspicious meters — enabling proactive intervention before revenue is lost.

---

## Key Results

| Metric | Value |
|---|---|
| Model | LSTM Autoencoder |
| Dataset | SGCC (42,000+ customers, 450 days) |
| Threshold | 25th percentile of train reconstruction error |
| Precision | 68.8% |
| Recall | 80.8% |
| F1 Score | 74.3% |
| Normal mean reconstruction error | 0.000142 |
| Theft mean reconstruction error | 0.009146 |
| Separation ratio | 64x |

---

## Architecture

```
Raw SGCC Data
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  → Missing value imputation, MinMaxScaler normalization
│  preprocess.py  │  → Synthetic theft augmentation (4 attack types)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │  → 22 features: statistical, temporal, neighborhood
│  features.py    │  → KMeans neighborhood grouping (500 clusters)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LSTM Autoencoder│  → Trained on normal customers only
│   model.py      │  → Reconstruction error = anomaly score
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    SHAP         │  → Surrogate RandomForest classifier
│  explain.py     │  → Per attack type feature importance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Streamlit Dashboard│ → 4 tabs: Overview, Insights, Customer Lookup,
│   app.py         │          Live Simulation
└─────────────────┘
```

---

## Theft Attack Types Simulated

| Type | Attack | Detection Signal |
|---|---|---|
| 1 | Scale Down — multiply consumption by 0.1–0.4x | min_consumption, coefficient_of_variation |
| 2 | Flatline — replace 30–60 day window with near-zero | std_consumption, coefficient_of_variation |
| 3 | Average — replace readings with mean value | skewness, kurtosis, peak_to_mean_ratio |
| 4 | Random Noise — reduce consumption by 20–50% randomly | trend_slope, coefficient_of_variation |

**Universal theft indicators:** `coefficient_of_variation` and `min_consumption` appear as top signals across 3 of 4 attack types.

---

## Project Structure

```
energy-theft-detection/
├── data/
│   ├── raw/                    # Raw SGCC CSV files
│   └── processed/              # Preprocessed .npy and .csv files
├── notebooks/
│   └── exploration.ipynb       # EDA and experimentation
├── src/
│   ├── preprocess.py           # Data cleaning + theft augmentation
│   ├── features.py             # 22-feature time-series extraction
│   ├── model.py                # LSTM Autoencoder training + scoring
│   ├── detect.py               # Threshold-based detection
│   └── explain.py              # SHAP explainability + plots
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   ├── shap_summary.png        # SHAP beeswarm plot
│   └── shap_bar.png            # SHAP bar plot
├── models/
│   ├── autoencoder.pt          # Trained LSTM Autoencoder weights
│   ├── surrogate_clf.pkl       # Surrogate RandomForest classifier
│   ├── scaler.pkl              # Fitted MinMaxScaler
│   └── loss_history.npy        # Training loss curve
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GridGuard-AI.git
cd GridGuard-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset

**SGCC (State Grid Corporation of China) Dataset**
- 42,372 electricity customers
- 1,035 days of daily consumption readings (we use 450 days)
- Binary labels: 0 = normal, 1 = theft
- Available on [Kaggle](https://www.kaggle.com/datasets/sgcc-electricity-theft)

Place the raw CSV in `data/raw/` before running the pipeline.

---

## Running the Pipeline

Run each step in order:

```bash
# Step 1 — Preprocess data + augment theft patterns
python src/preprocess.py

# Step 2 — Extract 22 time-series features
python src/features.py

# Step 3 — Train LSTM Autoencoder (recommended on GPU/Colab)
python src/model.py

# Step 4 — SHAP explainability + plots
python src/explain.py

# Step 5 — Launch Streamlit dashboard
streamlit run dashboard/app.py
```

---

## Dashboard

The Streamlit dashboard has 4 tabs:

- **Overview** — Total customers monitored, flagged meters, precision/recall metrics, theft distribution charts
- **Model Insights** — SHAP feature importance plots with plain-language explanations
- **Customer Lookup** — Per-customer reconstruction error, theft verdict, and top 5 SHAP features driving the prediction
- **Live Simulation** — Adjust key features via sliders and see theft probability update in real time

---

## Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch |
| Classical ML | scikit-learn |
| Explainability | SHAP |
| Feature Engineering | pandas, numpy |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit, Folium |
| Training Environment | Google Colab (CUDA GPU) |

---

## Future Improvements

- Replace KMeans neighborhood simulation with real meter network graph (GNN approach)
- Add real-time streaming data ingestion via MQTT or Kafka
- Fine-tune on more recent smart meter datasets beyond 2014–2016
- Deploy dashboard to Streamlit Cloud or Hugging Face Spaces

---

## License

MIT License — free to use for academic and research purposes.
