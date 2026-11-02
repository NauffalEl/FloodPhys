# FloodPhys — Flood Vulnerability Mapping, Banyumas Regency

Machine‑learning pipeline for producing a **flood vulnerability map** of Kabupaten Banyumas (Central Java, Indonesia) from open geospatial data. Built for the CENTIVE 2026 conference paper.

---

## Overview

```
DEM tiles ──► mosaic & clip ──► slope, TWI, HAND ──► ┐
Rivers ──────► distance‑to‑river, drainage density ──► ├─► feature table ──► ML models ──► vulnerability map
WorldCover ──► land‑cover code ─────────────────────► ┘
Rainfall ────► annual mean ─────────────────────────► ┘
```

1. **Pre‑process** — mosaic 9 DEMNAS tiles, clip to county boundary, derive terrain rasters, compute river proximity, extract pixel values at flood/non‑flood sample points, spatial‑block split.
2. **Train & evaluate** — fit four classifiers (LogReg, Random Forest, XGBoost, LightGBM) with imbalance‑aware hyper‑parameters, report AUC / F1 / precision / recall / latency.
3. **Visualise** — generate confusion matrices, ROC curves, feature importance chart, and the final vulnerability map.

---

## Dataset

All raw data are hosted on Google Drive:

> **[📂 Download Dataset](https://drive.google.com/drive/folders/1dGfy3hmHNI5i4xDqvG3oe6gSCQn_vdHM)**

| Layer | Source | Resolution |
|-------|--------|-----------|
| DEM | DEMNAS — BIG Indonesia | ~8 m |
| Land cover | ESA WorldCover 2021 v200 | 10 m |
| Rivers | HydroSHEDS HydroRIVERS v1.0 | vector |
| Boundary | geoBoundaries ADM2 (BPS / WFP / OCHA) | vector |
| Rainfall | BMKG Sta. Tunggul Wulung (monthly) | point |
| Flood points | BNPB + BPBD Banyumas (35 events, 2025) | point |
| Bencana stats | Open Data Kab. Banyumas (CKAN) | tabular |

---

## Repository Structure

```
FloodPhys/
├── Dataset/
│   ├── *.tif / *.geojson / *.csv   # raw inputs
│   └── processed/                   # derived rasters & feature tables
├── Program/
│   ├── preprocess.py                # raster derivation + sampling
│   ├── Train.py                     # training & full evaluation
│   ├── Predict.py                   # predict on new features
│   ├── visualize.py                 # plots & maps
│   ├── make_workflow_fig.py         # workflow diagram generator
│   ├── Model/
│   │   ├── model.py                 # build_models() definition
│   │   └── *.joblib                 # saved model artefacts
│   └── Utils/
│       ├── features.py              # FEATURES list & LABEL constant
│       └── metrics.py               # evaluation helpers
├── Outputs/
│   ├── figures/                     # confusion matrices, ROC, importance
│   ├── maps/                        # vulnerability map (PNG + GeoTIFF)
│   └── tables/                      # metrik_model.csv
├── Paper/
│   ├── main.tex                     # LaTeX source (infotel.cls)
│   ├── main.pdf                     # compiled manuscript
│   └── figures/                     # paper figures
└── README.md
```

---

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/<user>/centive26.git && cd centive26

# 2. Create environment
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install rasterio pysheds scipy shapely scikit-learn xgboost lightgbm joblib pandas matplotlib

# 3. Place raw data
#    Download from the Drive link above into Dataset/

# 4. Run pipeline
python Program/preprocess.py      # ~10 min — produces processed/ rasters + features.csv
python Program/Train.py           # ~30 s  — trains 4 models, saves .joblib & metrics
python Program/visualize.py       # generates figures & maps
```

---

## Results

Metrics from the latest spatial‑block evaluation (24 samples):

| Model | AUC | Accuracy | F1 Flood (thr=0.5) | Precision | Recall | Latency (ms) |
|-------|-----|----------|-------------------|-----------|--------|-------------|
| Random Forest | **1.00** | **1.00** | **1.00** | 1.00 | 1.00 | 193.5 |
| XGBoost | 1.00 | 0.92 | 0.92 | 0.86 | 1.00 | 1.1 |
| LightGBM | 1.00 | 0.92 | 0.92 | 0.86 | 1.00 | 3.0 |
| Logistic Reg. | 1.00 | 0.83 | 0.86 | 0.75 | 1.00 | 0.2 |

Full table: `Outputs/tables/metrik_model.csv`

---

## Data Sources & Licences

| Data | Provider | Licence |
|------|----------|---------|
| DEMNAS | Badan Informasi Geospasial (BIG) | Open data |
| WorldCover 2021 | ESA / Zanaga et al. | CC‑BY 4.0 |
| HydroRIVERS | HydroSHEDS / WWF | Public domain |
| geoBoundaries | William & Mary / BPS | CC‑BY 3.0 IGO |
| Rainfall | BMKG Sta. Tunggul Wulung | Open data |
| Flood events | BNPB / BPBD Banyumas | Public records |
| Bencana stats | Open Data Kab. Banyumas | Open data |

---

## Paper

The accompanying manuscript (`Paper/main.pdf`) is formatted with the `infotel.cls` IEEE‑style template for the CENTIVE 2026 conference.

## Licence

Code: **MIT**. Data: see table above.
