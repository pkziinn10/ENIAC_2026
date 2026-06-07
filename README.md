# Revenue Forecasting for E-Commerce: A Comparative Study of Statistical, Machine Learning, and Deep Learning Approaches

## Overview

This repository contains the complete experimental pipeline for the research article submitted to **ENIAC 2026**. The study addresses the problem of daily revenue forecasting in the online retail domain by systematically comparing three classes of predictive models across multiple granularity levels derived from a real-world transactional dataset.

The core contribution lies in the rigorous, multi-perspective evaluation: rather than predicting revenue from a single aggregated time series, we decompose the problem into three distinct series — by **Product**, by **Country**, and by **Customer** — each exhibiting different temporal dynamics, volatility profiles, and data availability windows. This design enables a nuanced analysis of model behavior under varying conditions of sparsity, seasonality, and scale.

---

## Dataset

The experiments are based on the **Online Retail II** dataset, originally published by the UCI Machine Learning Repository. It comprises approximately 1,067,371 transactions from a UK-based non-store online retailer, spanning from December 2009 to December 2010.

| Property             | Value                          |
|----------------------|--------------------------------|
| Source               | UCI Machine Learning Repository|
| Period               | Dec 2009 -- Dec 2010           |
| Raw transactions     | ~1,067,371                     |
| Target variable      | `Revenue` (daily aggregate)    |
| Granularity levels   | Product, Country, Customer     |

After cleaning (removal of cancellations, null customer IDs, and negative quantities), the dataset is aggregated into three daily time series with the following characteristics:

| Series    | Start Date   | Observations | Training Days | Validation Days |
|-----------|--------------|--------------|---------------|-----------------|
| Product   | 2010-03-15   | 194          | ~155          | ~39             |
| Country   | 2009-12-01   | 268          | ~229          | ~39             |
| Customer  | 2009-12-01   | 374          | ~335          | ~39             |

---

## Feature Engineering

A dedicated notebook (`engenharia_de_atributos.ipynb`) implements the construction of predictive features guided by SHAP-based importance analysis. All features are engineered with strict temporal discipline to prevent data leakage:

- **Expanding Window Aggregations**: Rolling statistics (mean, median, IQR) are computed using `expanding().shift(1)` to ensure that only past information is available at each time step.
- **Lag Features**: Revenue lags at 7, 14, and 30 days capture short- and medium-term autocorrelation.
- **Rolling Statistics**: 7-day and 30-day rolling mean, standard deviation, and maximum provide smoothed trend signals.
- **Calendar Variables**: Year, quarter, month, week, weekday, and day-of-year encode seasonal and cyclical patterns.
- **Pre-Christmas Indicator**: A binary flag marks the high-impact commercial period (November 1 through December 24), with associated historical mean quantities shifted by one year to avoid leakage.
- **Price Dispersion Features**: Week-weekday grouped statistics (`KnownStockCodePrice_WW_mean`, `_median`, `_std`) capture pricing volatility patterns using expanding windows.

---

## Methodology

### Validation Strategy

All models are evaluated under **Time Series Cross-Validation** using `TimeSeriesSplit` with **10 folds**. This approach creates progressively larger training windows while always validating on the immediately subsequent temporal block, preserving the causal structure of the data and providing robust performance estimates.

### Model Categories

The study evaluates models from three paradigms:

#### 1. Statistical Models (baseline)
Implemented in Python (`treinamento_estatisticos.ipynb`) using the `sktime` library:
- **AutoARIMA**: Automatic order selection for ARIMA(p,d,q) models.
- **AutoETS**: Exponential smoothing with automated component selection.

#### 2. Machine Learning Models
Implemented in `treinamento_modelos_ML.ipynb`:
- **XGBoost**: Gradient-boosted decision trees with hyperparameter optimization via Optuna (10 trials per fold). Search space includes `learning_rate` (log-uniform, 1e-3 to 0.1), `max_depth` (3--7), and `subsample` (0.6--1.0).
- **CatBoost**: Gradient boosting with native categorical support. Optimized via Optuna over `learning_rate` (log-uniform, 1e-3 to 0.1) and `depth` (4--6).

Both models use an inner validation split (last 15% of each training fold) for early stopping and hyperparameter selection. Final models are retrained on the full fold before evaluation.

#### 3. Deep Learning Models
Implemented in `treinamento_modelos_DL.ipynb` using the Darts framework:
- **BlockRNN (LSTM)**: A block-input/block-output recurrent architecture supporting `past_covariates`. Configuration: `input_chunk_length=7`, `output_chunk_length=1`, `hidden_dim=20`, `n_rnn_layers=1`, `dropout=0.1`.
- **TCN (Temporal Convolutional Network)**: Causal dilated convolutions for temporal pattern extraction. Configuration: `input_chunk_length=7`, `output_chunk_length=1`, `dropout=0.1`.

Both neural architectures receive exogenous features as `past_covariates` and undergo `MinMaxScaler` normalization. The covariates for the validation period are concatenated to the training covariates to satisfy the Darts forecasting horizon requirement.


---

## Results

The table below consolidates the performance metrics (MAE and RMSE) obtained under the holdout evaluation protocol described in the article:

| Model      | Category         | Series   | MAE        | RMSE      |
|------------|------------------|----------|------------|-----------|
| AutoARIMA  | Statistical      | Product  | 97.82      | 120.80    |
| AutoARIMA  | Statistical      | Country  | 6,865.28   | 8,313.52  |
| AutoARIMA  | Statistical      | Customer | 610.62     | 672.19    |
| AutoETS    | Statistical      | Product  | 125.32     | 146.55    |
| AutoETS    | Statistical      | Country  | 6,872.95   | 8,319.82  |
| AutoETS    | Statistical      | Customer | 522.39     | 591.22    |
| XGBoost    | Machine Learning | Product  | 97.20      | 119.92    |
| XGBoost    | Machine Learning | Country  | 2,276.79   | 2,945.49  |
| XGBoost    | Machine Learning | Customer | 93.29      | 218.07    |
| CatBoost   | Machine Learning | Product  | 94.20      | 117.33    |
| CatBoost   | Machine Learning | Country  | 2,339.87   | 3,046.38  |
| CatBoost   | Machine Learning | Customer | 99.23      | 237.82    |
| TCN        | Deep Learning    | Product  | 97.29      | 122.07    |
| TCN        | Deep Learning    | Country  | 3,704.56   | 4,712.20  |
| TCN        | Deep Learning    | Customer | 475.44     | 665.59    |
| LSTM       | Deep Learning    | Product  | 95.32      | 117.96    |
| LSTM       | Deep Learning    | Country  | 4,096.80   | 5,298.72  |
| LSTM       | Deep Learning    | Customer | 394.62     | 556.97    |

### Key Findings

- **Machine Learning models** (XGBoost, CatBoost) achieved the lowest MAE across practically all granularity levels, dominating especially in the Country and Customer series. This demonstrates their capability in capturing non-linear relationships and efficiently handling the engineered tabular features.
- **Deep Learning models** (TCN, LSTM) showed highly competitive performance, closely following the tree-based models and outperforming classical statistical approaches.
- **Statistical models** (AutoARIMA, AutoETS) struggled to maintain parity, presenting the highest error metrics in the Country and Customer series, suggesting that simple univariate temporal patterns are insufficient compared to rich feature sets.

---

## Project Structure

```
revenue_estimate/
├── README.md
├── .gitignore
└── src/
    ├── limpeza_de_dados.ipynb              # Data cleaning and preprocessing
    ├── engenharia_de_atributos.ipynb       # Feature engineering (SHAP-guided, leakage-free)
    ├── analise_exploratoria.Rmd            # Exploratory Data Analysis
    ├── treinamento_naive.ipynb             # Baseline naive models
    ├── treinamento_estatisticos.ipynb      # Statistical models (AutoARIMA, AutoETS via sktime)
    ├── treinamento_modelos_ML.ipynb        # ML models (XGBoost, CatBoost + Optuna)
    ├── treinamento_modelos_DL.ipynb        # DL models (BlockRNN/LSTM, TCN via Darts)
    ├── LLMs.ipynb                          # LLMs evaluation
    ├── teste_wilcoxon.ipynb                # Wilcoxon signed-rank statistical test
    ├── analise_shap.ipynb                  # SHAP values analysis and visualizations
    ├── xlsx_to_csv.ipynb                   # Utility for data conversion
    ├── requirements.txt                    # Python dependencies
    ├── figuras/                            # Exported charts and figures
    │   ├── shap_summary_*.pdf              # SHAP summary plots in PDF format
    │   └── shap_bar_*.pdf                  # SHAP bar plots (mean impact) in PDF format
    ├── data/
    │   ├── product.csv                     # Daily series aggregated by product
    │   ├── country.csv                     # Daily series aggregated by country
    │   └── customer.csv                    # Daily series aggregated by customer
    ├── resultados/
    │   ├── resultados_naive.csv            # Metrics and forecasts for Naive models
    │   ├── resultados_estatisticos.csv     # Metrics and forecasts for Statistical models
    │   ├── resultados_ML.csv               # Metrics and forecasts for ML models
    │   ├── resultados_DL.csv               # Metrics and forecasts for DL models
    │   ├── resultados_LLMs.csv             # Metrics and forecasts for LLMs
    │   ├── resultados_medias_LLMs.csv      # Average metrics for LLMs
    │   └── resultado_wilcoxon.txt          # Statistical significance comparisons
    └── models/
        └── result_pkl/
            ├── xgb_*.pkl                   # Serialized XGBoost models
            ├── cat_*.pkl                   # Serialized CatBoost models
            ├── lstm_*.pt                   # Serialized LSTM models (PyTorch)
            └── tcn_*.pt                    # Serialized TCN models (PyTorch)
```

---

## Reproduction

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended for deep learning models)

### Setup

```bash
cd src
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Execution Order

1. `limpeza_de_dados.ipynb` -- Generates cleaned transactional data.
2. `engenharia_de_atributos.ipynb` -- Produces feature-enriched datasets.
3. `treinamento_naive.ipynb` -- Generates simple baselines.
4. `treinamento_estatisticos.ipynb` -- Trains and evaluates statistical models.
5. `treinamento_modelos_ML.ipynb` -- Trains XGBoost and CatBoost with Optuna optimization.
6. `treinamento_modelos_DL.ipynb` -- Trains BlockRNN (LSTM) and TCN via Darts.
7. `LLMs.ipynb` -- Evaluates LLMs approach.
8. `teste_wilcoxon.ipynb` -- Executes the statistical hypothesis tests between the best models.
9. `analise_shap.ipynb` -- Performs SHAP analysis to explain feature impact and exports plots as PDFs.

---

## Technology Stack

| Component              | Technology                                    |
|------------------------|-----------------------------------------------|
| Data manipulation      | Pandas, NumPy                                 |
| Statistical models     | sktime, pmdarima, statsmodels                 |
| ML models              | XGBoost, CatBoost                             |
| Hyperparameter tuning  | Optuna                                        |
| DL framework           | Darts (PyTorch Lightning backend)             |
| DL architectures       | BlockRNNModel (LSTM), TCNModel                |
| Scaling                | Scikit-learn (MinMaxScaler via Darts Scaler)  |
| Validation             | Scikit-learn (TimeSeriesSplit)                 |
| Visualization          | Matplotlib, Plotly                            |
| Explainability         | SHAP                                          |

---

## License

This repository is part of an academic research project submitted to ENIAC 2026. Please contact the authors before any commercial or derivative use.
