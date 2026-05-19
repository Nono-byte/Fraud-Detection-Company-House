# Fraud Detection MVP — Companies House UK

Detects potentially fraudulent company registrations in the Companies House UK database using unsupervised anomaly detection.

## Setup

```bash
pip install requests pandas numpy scikit-learn pyod shap matplotlib seaborn tqdm xgboost lightgbm
```

Set your API key (optional — synthetic mode runs without it):
```bash
export COMPANIES_HOUSE_API_KEY=your_key_here
```

## Usage

Open `IQbusiness_Fraud_Detection_Companies_House.ipynb` and run all cells.

Toggle live vs. synthetic data at the top of Section 1:
```python
USE_LIVE_API = False  # Set to True for live Companies House API
```

## Pipeline

| Step | Description |
|---|---|
| Data Gathering | Companies House REST API or synthetic data generator |
| Cleaning & EDA | Type validation, missing values, distribution analysis |
| Feature Engineering | 20+ signals: governance risk, compliance, filing velocity, naming patterns |
| Feature Selection | Variance filter → correlation filter → Mutual Information ranking |
| Anomaly Detection | Isolation Forest, LOF, HBOS, Autoencoder (PyOD) |
| Ensemble Scoring | Weighted average → HIGH / MEDIUM / LOW risk tiers |
| Explainability | SHAP feature attribution per flagged company |

## Outputs

| File | Description |
|---|---|
| `fraud_flagged_companies.csv` | HIGH risk entities with top fraud driver |
| `companies_scored.csv` | Full dataset with ensemble fraud scores |

## API Reference

Base URL: `https://api.company-information.service.gov.uk`  
Docs: [developer.company-information.service.gov.uk](https://developer.company-information.service.gov.uk)
