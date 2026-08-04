# 🌍 Urban Air Quality Prediction with Satellite & Ground Sensor Fusion

**MSc Data Science Dissertation | University of Roehampton**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shifa19sayed/air-quality-prediction/blob/main/MSc_Project_AirQuality.ipynb)

---

## Overview

This project investigates whether ground sensor and meteorological data materially improve urban
PM2.5 Air Quality Index (AQI) classification, and quantifies the performance degradation as ground
sensor availability is progressively reduced — evaluated on **real OpenAQ ground sensor data** from
verified UK monitoring stations, fused with ERA5 meteorological reanalysis. A subsequent data
augmentation phase addresses class imbalance identified during evaluation.

**Base Paper:** Bai, X., Zhang, N., Cao, X. and Chen, W. (2024) *Prediction of PM2.5 concentration based
on a CNN-LSTM neural network algorithm*, PeerJ, 12, e17811.
DOI: [10.7717/peerj.17811](https://doi.org/10.7717/peerj.17811)

---

## Key Results (Real 2017 UK Sensor Data)

### Sensor Sparsity Ablation Study — Core Novel Contribution

| Metric | 100% Sensors | 75% | 50% | 25% | 0% (No Sensors) |
|---|---|---|---|---|---|
| Macro F1 | 0.4933 | 0.4700 | 0.4516 | 0.3757 | 0.4142 |
| Cohen's κ | **0.4806** | 0.4091 | 0.3582 | 0.1825 | **0.0053** |

Cohen's Kappa — the primary degradation metric — falls from **moderate–substantial agreement**
at full sensor availability to **chance-level agreement** once ground sensors are entirely removed,
directly answering the project's research question: ground sensor meteorological data materially
improves AQI classification, concentrated specifically in distinguishing Moderate-band pollution days.

### Classification Results

| Model | Macro F1 | Cohen's κ |
|---|---|---|
| CNN-LSTM AQI Classifier | 0.5876 | 0.1839 |
| XGBoost Baseline | 0.4933 | 0.4806 |

### Data Augmentation — Class Balancing (Gaussian Jitter)

Following review of the confusion matrices showing zero-scoring minority classes, a fourth phase
applied Gaussian jitter augmentation to the CNN-LSTM's training sequences (SMOTE was judged
unsuitable for sequential data since its interpolation distorts temporal structure).

| Model | Macro F1 | Cohen's κ |
|---|---|---|
| CNN-LSTM (original, imbalanced) | 0.5876 | 0.1839 |
| CNN-LSTM (augmented, balanced) | **0.6006** | **0.2039** |

Augmentation improved overall classification quality but confirmed an important boundary: classes
entirely absent from the test set (Unhealthy_Sensitive, Unhealthy, Very_Unhealthy) remained at
F1 = 0.0000 even after augmentation, since training-side rebalancing cannot manufacture test
examples of a class that does not occur in the evaluated period.

---

## Research Question

> Does ground sensor meteorological data materially improve urban PM2.5 AQI classification relative to
> temporal features alone, and how does classification quality degrade as ground sensor availability is
> progressively reduced?

---

## Project Structure

```
air-quality-prediction/
├── MSc_Project_AirQuality.ipynb   # Main Colab notebook (executed, real data + augmentation)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/
│   ├── raw/                       # OpenAQ + ERA5 raw downloads (gitignored)
│   └── processed/                 # Processed daily PM2.5 CSVs (gitignored)
├── models/                        # Trained model checkpoints (gitignored)
└── outputs/
    ├── figures/                   # Real-data EDA, training curves, ablation, augmentation plots
    └── results/                   # CSV results tables
```

---

## Four-Phase Pipeline

| Phase | Description | Real-Data Result |
|-------|-------------|-------------------|
| **Phase 1 — Replication** | CNN-LSTM regression matching Bai et al. (2024) | R² = -3.31, RMSE = 11.68 µg/m³ |
| **Phase 2 — Extension** | AQI band classification (CNN-LSTM + XGBoost + SHAP) | Macro F1 up to 0.59, κ up to 0.48 |
| **Phase 3 — Novel Contribution** | Sensor sparsity ablation: 100% → 75% → 50% → 25% → 0% ground sensors | κ: 0.48 → 0.01 (genuine degradation) |
| **Phase 4 — Class Balancing** | Gaussian jitter augmentation for CNN-LSTM training data | Macro F1 +0.013, κ +0.020 |

---

## Data Sources

| Dataset | Source | Coverage | Access |
|---------|--------|----------|--------|
| OpenAQ (real) | [openaq.org](https://openaq.org) | 14 verified UK sensors, 2017 | Free API key → [register](https://explore.openaq.org/register) |
| ERA5 Reanalysis | [ECMWF CDS](https://cds.climate.copernicus.eu) | Temperature, wind, pressure, humidity, 2017 | Free → [register](https://cds.climate.copernicus.eu/profile) |

### Important OpenAQ v3 API notes (discovered during development)

1. **The `city=` location filter is unreliable** — it can silently return stations from other countries.
   This project uses an explicit geographic bounding box per city, combined with verification of each
   result's `country.code == 'GB'`.
2. **The `/v3/sensors/{sensor_id}/measurements` endpoint does not enforce `date_from`/`date_to`** — it
   returns each sensor's complete historical record regardless of the requested range. This project
   diagnoses actual per-sensor coverage by inspecting returned timestamp distributions, then restricts
   the processed dataset to a verified, fully-covered year (2017) rather than trusting the request
   parameters.

Both issues, their diagnosis, and their fixes are documented in the accompanying dissertation report.

---

## Setup

### Option 1 — Google Colab (recommended)

1. Click the **Open in Colab** badge above
2. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
3. Add secrets (🔑 icon in left sidebar):
   - `OPENAQ_API_KEY` — from [explore.openaq.org/register](https://explore.openaq.org/register)
   - `CDS_TOKEN` — Personal Access Token from [cds.climate.copernicus.eu/profile](https://cds.climate.copernicus.eu/profile)
   - Accept the ERA5 licence once at [cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) (scroll to bottom → Accept terms)
4. Run all cells (Runtime → Run all). The OpenAQ fetch takes ~15 minutes (real API calls with retry/backoff).

### Option 2 — Local Installation

```bash
git clone https://github.com/shifa19sayed/air-quality-prediction.git
cd air-quality-prediction
pip install -r requirements.txt
jupyter notebook MSc_Project_AirQuality.ipynb
```

---

## Model Architectures

### BaiCNNLSTM (Phase 1 — Replication)
```
Conv1D(64) → ReLU
Conv1D(128) → ReLU
LSTM(128, batch_first=True)
LSTM(64, batch_first=True)
Linear(64→32) → Sigmoid
Linear(32→1) → Sigmoid
Loss: MSELoss | Optimiser: Adam (lr=0.001)
```

### AQIClassifierCNNLSTM (Phase 2 — Extension)
```
Conv1D(64) → ReLU → BatchNorm1d(64)
Conv1D(128) → ReLU → BatchNorm1d(128)
LSTM(128, batch_first=True)
LSTM(64, batch_first=True)
Dropout(0.3)
Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→n_classes)
Loss: CrossEntropyLoss (class-weighted) | Optimiser: AdamW
```

---

## AQI Band Definitions (US EPA)

| Band | PM2.5 (µg/m³) | Health Implication |
|------|----------------|-------------------|
| Good | 0 – 12.0 | Satisfactory |
| Moderate | 12.1 – 35.4 | Acceptable |
| Unhealthy for Sensitive Groups | 35.5 – 55.4 | Limit prolonged outdoor exertion |
| Unhealthy | 55.5 – 150.4 | Everyone may experience effects |
| Very Unhealthy | > 150.4 | Health alert |

---

## Limitations

- Real-data sample is drawn predominantly from London (16 of 24 qualifying sensors); generalisation to
  Manchester, Birmingham, Leeds, and Bristol individually is limited by sample size in this run.
- The Very_Unhealthy class is absent from the 2017 sample; Unhealthy_Sensitive appears in training but
  not in either evaluated test split. Neither can be recovered through training-side augmentation.
- MODIS satellite AOD fusion (architecturally supported via the `get_embedding()` method) was not
  exercised in the reported results; the "0% ground sensor" ablation condition reflects temporal
  features only, not genuine satellite inference.

Full discussion in the accompanying dissertation report.

---

## Citation

```bibtex
@article{bai2024pm25,
  title={Prediction of PM2.5 concentration based on a CNN-LSTM neural network algorithm},
  author={Bai, Xuesong and Zhang, Na and Cao, Xiaoyi and Chen, Wenqian},
  journal={PeerJ},
  volume={12},
  pages={e17811},
  year={2024},
  doi={10.7717/peerj.17811}
}
```

---

## Licence

MIT Licence — see [LICENSE](LICENSE) for details.

## Author

**Shifa Ali Asif Sayed** | MSc Data Science | University of Roehampton | 2024–2026
