"""
models.py
=========
Sklearn-style wrappers for the six ML models we train to predict next-month
stock returns:

    * Ridge (linear shrinkage; the workhorse baseline of Kelly/Malamud/Zhou)
    * Random Forest
    * Gradient Boosting (XGBoost if installed, else LightGBM, else sklearn)
    * MLP Neural Network (sklearn MLPRegressor)
    * PLS Regression (Kelly & Pruitt 2013-style supervised dimension reduction)

Plus a `run_expanding_window_predictions` orchestrator that:
    * Walks forward through time monthly
    * Retrains on the expanding window {data <= t} every `refit_freq` months
    * Picks hyperparameters via TimeSeriesSplit on the training window
    * Generates *strictly* out-of-sample predictions for the next block

Design notes (from Gu, Kelly & Xiu 2020 + Kelly, Malamud & Zhou 2024):
    * Expanding window, NOT rolling -- accounting predictors are highly
      persistent so more history helps, and KMZ specifically argue for
      maximally large training sets.
    * Refit annually (not monthly) for compute reasons; this is what GKX did
      and our results should be insensitive to it.
    * Cross-validation must respect the time axis -- never use future
      observations to validate past ones. We use sklearn's TimeSeriesSplit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# Optional faster GBMs. Fall back gracefully so the pipeline still runs.
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# --------------------------------------------------------------------------- #
# Model registry                                                              #
# --------------------------------------------------------------------------- #
def _make_ridge(alpha: float = 1.0):
    """Ridge regression with L2 shrinkage.

    The KMZ "virtue of complexity" result is most easily seen with ridge:
    even when p >> n, a small positive alpha gives a well-defined estimator
    whose out-of-sample R^2 keeps improving as p grows (up to a point).
    """
    return Ridge(alpha=alpha, fit_intercept=True, random_state=0)


def _make_random_forest():
    """Random forest -- captures non-linearities & interactions.

    GKX find tree ensembles among the top performers because firm
    characteristics interact heavily (e.g. value works mostly among small
    stocks).
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=500,   # heavy regularization for noisy financial data;
                                # also speeds up fits substantially
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )


def _make_gbm():
    """Gradient boosting: prefer XGBoost > LightGBM > sklearn.

    We pick the fastest implementation available so refits are tractable.
    Hyperparameters are mildly tuned for monthly return prediction:
    shallow trees + lots of boosting rounds + strong shrinkage.
    """
    if _HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            n_jobs=-1,
            verbosity=0,
            random_state=0,
            tree_method="hist",
        )
    if _HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            num_leaves=15,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            n_jobs=-1,
            verbosity=-1,
            random_state=0,
        )
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        random_state=0,
    )


def _make_mlp(hidden=(16, 8)):
    """Two-layer MLP, mirroring GKX's NN3 architecture (just smaller).

    Returns are mostly noise so very deep networks overfit immediately;
    GKX find 2-3 hidden layers, 32->16->8 neurons works best. We also use
    early stopping based on a held-out validation slice.
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-3,                    # L2 weight decay
        batch_size=2048,               # large batches -- finance datasets are big
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=0,
    )


def _make_pls(n_components: int = 3):
    """Partial Least Squares regression (Kelly & Pruitt 2013).

    PLS finds low-dimensional combinations of features that maximally covary
    with the target -- think of it as supervised PCA. It's a complexity
    sweet-spot between OLS and ridge: aggressive dimension reduction.
    """
    # PLS in sklearn returns a 2D array for predict -- we'll squeeze later.
    return PLSRegression(n_components=n_components, scale=False)


def get_model_registry() -> Dict[str, callable]:
    """Map a friendly name to a no-arg factory producing a fresh estimator."""
    return {
        "Ridge":         _make_ridge,
        "RandomForest":  _make_random_forest,
        "GBM":           _make_gbm,
        "MLP":           _make_mlp,
        "PLS":           _make_pls,
    }


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (Ridge only, for speed)                                #
# --------------------------------------------------------------------------- #
def tune_ridge_alpha(X: np.ndarray, y: np.ndarray,
                     alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
                     n_splits: int = 3) -> float:
    """
    Pick the ridge alpha that minimises MSE on time-ordered CV folds.
    Using TimeSeriesSplit means each validation fold is *strictly later*
    than its training fold, which preserves the no-look-ahead property
    inside the training window.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_score = alphas[0], np.inf
    for a in alphas:
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            m = Ridge(alpha=a, fit_intercept=True, random_state=0)
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[va_idx])
            scores.append(np.mean((pred - y[va_idx]) ** 2))
        score = float(np.mean(scores))
        if score < best_score:
            best_score, best_alpha = score, a
    return best_alpha


# --------------------------------------------------------------------------- #
# Expanding-window prediction engine                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ExpandingWindowConfig:
    """Configuration for the walk-forward training loop."""
    feature_cols: List[str] = field(default_factory=list)
    target_col: str = "fwd_ret"
    date_col: str = "date"
    train_end_date: str = "2000-12-31"     # initial train window
    refit_freq_months: int = 12            # re-train annually
    tune_ridge: bool = True                # CV-tune ridge alpha
    verbose: bool = True


def _safe_predict(model, X: np.ndarray) -> np.ndarray:
    """PLS predict returns a 2D column; everything else returns 1D. Normalize."""
    pred = model.predict(X)
    if pred.ndim > 1:
        pred = pred.ravel()
    return pred


def run_expanding_window_predictions(
    panel: pd.DataFrame,
    model_name: str,
    model_factory: callable,
    cfg: ExpandingWindowConfig,
) -> pd.DataFrame:
    """
    Walk forward through time. At each refit boundary:
        1. Train on everything strictly before the boundary.
        2. Predict the next `refit_freq_months` of data.
    Return a long-format dataframe of (date, permno, y_true, y_pred).

    This is the canonical setup of Gu/Kelly/Xiu (2020): every prediction is
    out-of-sample with respect to its model's training window.
    """
    panel = panel.sort_values([cfg.date_col, "permno"]).reset_index(drop=True)
    dates = pd.to_datetime(panel[cfg.date_col].unique())
    dates = pd.DatetimeIndex(sorted(dates))

    train_end = pd.Timestamp(cfg.train_end_date)
    if train_end < dates.min() or train_end > dates.max():
        raise ValueError(
            f"train_end_date {train_end.date()} outside panel range "
            f"[{dates.min().date()}, {dates.max().date()}]")

    # Build the schedule of (train_cutoff, test_start, test_end) tuples.
    test_dates = dates[dates > train_end]
    schedule = []
    i = 0
    while i < len(test_dates):
        block = test_dates[i:i + cfg.refit_freq_months]
        schedule.append((block[0], block[-1]))
        i += cfg.refit_freq_months

    predictions = []
    for k, (test_start, test_end) in enumerate(schedule):
        # CRITICAL no-look-ahead rule: the target `fwd_ret` for a training
        # row dated `t` is the return realised between t and t+1. So if we
        # included a row dated exactly one month before the test window
        # begins, its target return would fall *inside* the test window --
        # the model would learn from data the test set is about to score
        # it on. We therefore require both
        #     row.date            <  test_start   (feature is pre-test)
        # AND row.date + 1 month  <  test_start   (target is pre-test)
        # which collapses to date < test_start - 1 month.
        train_cutoff = test_start - pd.offsets.MonthEnd(1)
        train_mask = panel[cfg.date_col] < train_cutoff
        test_mask = ((panel[cfg.date_col] >= test_start) &
                     (panel[cfg.date_col] <= test_end))

        X_tr = panel.loc[train_mask, cfg.feature_cols].to_numpy(np.float32)
        y_tr = panel.loc[train_mask, cfg.target_col].to_numpy(np.float32)
        X_te = panel.loc[test_mask, cfg.feature_cols].to_numpy(np.float32)

        if len(X_tr) < 1000 or len(X_te) == 0:
            continue

        # Build a fresh model. For Ridge, optionally tune alpha first.
        if model_name == "Ridge" and cfg.tune_ridge:
            best_alpha = tune_ridge_alpha(X_tr, y_tr)
            model = _make_ridge(alpha=best_alpha)
        else:
            model = model_factory()

        try:
            model.fit(X_tr, y_tr)
            y_pred = _safe_predict(model, X_te)
        except Exception as e:
            if cfg.verbose:
                print(f"[models:{model_name}] fit failed for window "
                      f"ending {test_end.date()}: {e}")
            continue

        block = panel.loc[test_mask, [cfg.date_col, "permno",
                                      cfg.target_col]].copy()
        block["y_pred"] = y_pred
        predictions.append(block)

        if cfg.verbose:
            print(f"[models:{model_name}] window {k + 1}/{len(schedule)}: "
                  f"trained on {len(X_tr):,} obs through "
                  f"{(train_cutoff - pd.offsets.MonthEnd(1)).date()}, "
                  f"predicted {len(X_te):,} obs "
                  f"{test_start.date()}..{test_end.date()}")

    if not predictions:
        return pd.DataFrame(columns=[cfg.date_col, "permno",
                                     cfg.target_col, "y_pred"])
    out = pd.concat(predictions, ignore_index=True)
    out = out.rename(columns={cfg.target_col: "y_true"})
    return out


def run_all_models(
    panel: pd.DataFrame,
    feature_cols: List[str],
    train_end_date: str = "2000-12-31",
    refit_freq_months: int = 12,
    models_to_run: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience: run every model in the registry on the same panel and
    return a dict {model_name: predictions_dataframe}.
    """
    registry = get_model_registry()
    if models_to_run is None:
        models_to_run = list(registry.keys())

    cfg = ExpandingWindowConfig(
        feature_cols=feature_cols,
        train_end_date=train_end_date,
        refit_freq_months=refit_freq_months,
        verbose=verbose,
    )

    results: Dict[str, pd.DataFrame] = {}
    for name in models_to_run:
        if name not in registry:
            print(f"[models] unknown model '{name}', skipping.")
            continue
        if verbose:
            print(f"\n--- Training {name} ---")
        results[name] = run_expanding_window_predictions(
            panel, name, registry[name], cfg
        )
    return results
