# ML-Based Investment Portfolio System

Course project for FinTech Lab. Implements the end-to-end pipeline from
*Empirical Asset Pricing via Machine Learning* (Gu, Kelly & Xiu, 2020) and
*The Virtue of Complexity in Return Prediction* (Kelly, Malamud & Zhou, 2024)
using firm-level accounting predictors from the
[Open Source Asset Pricing](https://www.openassetpricing.com) project
(Chen & Zimmermann).

## What it does

1. **Downloads** firm characteristics via the `openassetpricing` Python
   package. Without WRDS credentials it cannot get firm-month CRSP returns,
   so it falls back to a realistic simulated panel with the same schema
   (firm entry/exit, persistent AR(1) characteristics, sparse "true" betas,
   ~10% monthly idiosyncratic vol). Everything downstream works identically.
2. **Cleans** the panel: Fama-French 6-month accounting lag, cross-sectional
   median imputation, 1/99 winsorization, rank-normalization to [-0.5, 0.5]
   — all within month, no look-ahead.
3. **Trains** five ML models with expanding-window walk-forward refits:
   Ridge (with TimeSeriesSplit-CV-tuned alpha), Random Forest, XGBoost (or
   LightGBM / sklearn GBM as fallbacks), MLP, and PLS.
4. **Builds** monthly decile portfolios and the long-short top-minus-bottom
   spread, equal-weighted.
5. **Evaluates**: annualised return, Sharpe, max drawdown, CAPM alpha+beta
   (with t-stat), out-of-sample R² in the GKX convention. Charts for
   cumulative returns, drawdowns, decile bars, and tree feature importance.
6. **Demonstrates** the "virtue of complexity": Ridge OOS R² and Sharpe as
   a function of model size, both with raw predictors and with a random
   Fourier feature expansion that reaches the n << p regime.

## Files

| File                       | Role                                            |
|---------------------------|-------------------------------------------------|
| `data_pipeline.py`         | Build the clean firm-month panel                |
| `models.py`                | Sklearn-style wrappers + walk-forward driver    |
| `portfolio.py`             | Decile sorts + long-short construction          |
| `evaluation.py`            | Metrics + plots                                 |
| `complexity_analysis.py`   | Ridge complexity sweeps                         |
| `main.py`                  | End-to-end orchestrator                         |
| `results_walkthrough.ipynb`| Jupyter notebook with narrative + embedded outputs |

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

or, for a guided walkthrough with explanatory text and charts:

```bash
jupyter lab results_walkthrough.ipynb
```

Outputs (CSVs + PNG charts) are saved under `./outputs/`.

## Performance tuning

Wall time is dominated by the Random Forest refits and the largest random
Fourier feature widths. If `python main.py` is too slow on your hardware,
edit `main.py`:

- Set `REFIT_FREQ_MONTHS = 24` to refit every two years instead of annually.
- In `complexity_analysis.run_complexity_analysis`, drop the largest entry
  from `rff_widths` (default goes to `p=500`).

## Notes on look-ahead bias

This is the most important correctness property of the project. Three
defences are in place:

- **At the feature level**: the 6-month accounting lag ensures we only use
  information that would have been public at the formation date.
- **At the cross-section level**: winsorization and rank-normalization are
  computed *within each month*, so no future cross-sections leak in.
- **At the time-series level**: training uses only data strictly before
  the test block, and CV inside the training window uses `TimeSeriesSplit`
  so every validation fold is later than its training fold.
