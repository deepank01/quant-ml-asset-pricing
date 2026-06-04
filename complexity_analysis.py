"""
complexity_analysis.py
======================
Empirically demonstrate the "virtue of complexity" of Kelly, Malamud &
Zhou (2024, JF). Their key claim: when training data is fixed but the
number of parameters p grows past the number of observations n, ridge
regression's out-of-sample R^2 keeps improving (in a non-monotone way:
the "double descent" curve dips around p ~ n and then climbs again).

We replicate the spirit of that experiment with two complementary
exercises:

    Sweep A — Raw feature count.
        Run Ridge with the first k of our accounting predictors, for
        k in {1, 3, 5, 7, len(predictors)}. This is what the spec asks
        for ("features = 5, 10, 20, 50, all"). With only 9 base
        predictors we adapt the grid accordingly.

    Sweep B — Random feature expansion (the actual KMZ setup).
        Lift the X features to p random non-linear basis functions
        (random Fourier features: x -> cos(W x + b) with W ~ N(0,1)).
        Sweep p in {2, 5, 10, 20, 50, 100, 200, 500} so we span the
        n < p regime that KMZ care about. Ridge with a small but fixed
        alpha is fit at each width.

For each sweep we plot:
    * OOS R^2 vs number of parameters
    * Sharpe of the L/S portfolio vs number of parameters
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
# Same convention as evaluation.py: do not force a backend at import time.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge

from evaluation import out_of_sample_r2, sharpe_ratio
from models import ExpandingWindowConfig, _safe_predict
from portfolio import build_decile_portfolios, long_short_returns

sns.set_style("whitegrid")


# --------------------------------------------------------------------------- #
# Sweep A: raw feature count                                                  #
# --------------------------------------------------------------------------- #
def sweep_raw_features(panel: pd.DataFrame,
                       feature_cols: List[str],
                       grid: List[int],
                       train_end_date: str,
                       refit_freq_months: int = 12,
                       alpha: float = 1.0) -> pd.DataFrame:
    """
    For each k in `grid`, run an expanding-window Ridge using the first k
    of `feature_cols` and report OOS R^2 + L/S Sharpe.
    """
    rows = []
    for k in grid:
        k = min(k, len(feature_cols))
        cols = feature_cols[:k]
        preds = _run_ridge_expanding(panel, cols, train_end_date,
                                     refit_freq_months, alpha=alpha)
        if preds.empty:
            continue
        r2 = out_of_sample_r2(preds)
        deciles = build_decile_portfolios(preds)
        ls = long_short_returns(deciles)
        rows.append({
            "n_params": k,
            "oos_r2": r2,
            "ls_sharpe": sharpe_ratio(ls),
            "ls_ann_return": ls.dropna().mean() * 12,
        })
        print(f"[complexity:A] p={k:3d}  "
              f"OOS R^2={r2*100:+.3f}%  "
              f"L/S Sharpe={rows[-1]['ls_sharpe']:.2f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sweep B: random Fourier feature expansion (the KMZ setup)                   #
# --------------------------------------------------------------------------- #
def _rff_transform(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Random Fourier features: phi(x) = sqrt(2/p) * cos(W x + b).
    Rahimi & Recht (2007). Approximates a Gaussian kernel as p -> infinity.
    """
    p = W.shape[0]
    return np.sqrt(2.0 / p) * np.cos(X @ W.T + b)


def sweep_random_features(panel: pd.DataFrame,
                          feature_cols: List[str],
                          widths: List[int],
                          train_end_date: str,
                          refit_freq_months: int = 12,
                          alpha: float = 1.0,
                          bandwidth: float = 1.0,
                          seed: int = 0) -> pd.DataFrame:
    """
    Lift X to p random Fourier features and fit Ridge for each p in `widths`.

    Why this matters: with only 9 base predictors we *can't* reach the
    n << p regime by listing more variables -- but we can by non-linearly
    expanding what we have. The KMZ result is precisely about p, not about
    economic interpretability of features.
    """
    rng = np.random.default_rng(seed)
    d = len(feature_cols)
    rows = []
    # Pre-draw the maximum-width projection; smaller widths just slice it.
    p_max = max(widths)
    W_full = rng.standard_normal((p_max, d)).astype(np.float32) * bandwidth
    b_full = rng.uniform(0, 2 * np.pi, size=p_max).astype(np.float32)

    for p in widths:
        W, b = W_full[:p], b_full[:p]
        preds = _run_ridge_rff_expanding(
            panel, feature_cols, W, b,
            train_end_date, refit_freq_months, alpha
        )
        if preds.empty:
            continue
        r2 = out_of_sample_r2(preds)
        deciles = build_decile_portfolios(preds)
        ls = long_short_returns(deciles)
        rows.append({
            "n_params": p,
            "oos_r2": r2,
            "ls_sharpe": sharpe_ratio(ls),
            "ls_ann_return": ls.dropna().mean() * 12,
        })
        print(f"[complexity:B] p={p:4d}  "
              f"OOS R^2={r2*100:+.3f}%  "
              f"L/S Sharpe={rows[-1]['ls_sharpe']:.2f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Helpers: expanding-window Ridge specialised for this analysis               #
# --------------------------------------------------------------------------- #
def _run_ridge_expanding(panel, cols, train_end_date,
                         refit_freq_months, alpha):
    """Expanding-window Ridge on a given subset of *raw* columns."""
    cfg = ExpandingWindowConfig(
        feature_cols=cols, train_end_date=train_end_date,
        refit_freq_months=refit_freq_months, tune_ridge=False, verbose=False,
    )
    return _ridge_loop(panel, cfg, transform=None, alpha=alpha)


def _run_ridge_rff_expanding(panel, cols, W, b, train_end_date,
                             refit_freq_months, alpha):
    """Expanding-window Ridge on a random Fourier feature lift."""
    cfg = ExpandingWindowConfig(
        feature_cols=cols, train_end_date=train_end_date,
        refit_freq_months=refit_freq_months, tune_ridge=False, verbose=False,
    )

    def tf(X):
        return _rff_transform(X, W, b)

    return _ridge_loop(panel, cfg, transform=tf, alpha=alpha)


def _ridge_loop(panel, cfg, transform, alpha):
    """Inner walk-forward loop shared by both sweeps."""
    panel = panel.sort_values([cfg.date_col, "permno"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(panel[cfg.date_col].unique()))
    train_end = pd.Timestamp(cfg.train_end_date)
    test_dates = dates[dates > train_end]
    if len(test_dates) == 0:
        return pd.DataFrame()

    schedule = []
    i = 0
    while i < len(test_dates):
        block = test_dates[i:i + cfg.refit_freq_months]
        schedule.append((block[0], block[-1]))
        i += cfg.refit_freq_months

    preds = []
    for (test_start, test_end) in schedule:
        # Same no-look-ahead rule as models.run_expanding_window_predictions:
        # the training-row target `fwd_ret` must NOT fall inside the test
        # window, so we exclude rows dated within one month of test_start.
        train_cutoff = test_start - pd.offsets.MonthEnd(1)
        tr_mask = panel[cfg.date_col] < train_cutoff
        te_mask = ((panel[cfg.date_col] >= test_start) &
                   (panel[cfg.date_col] <= test_end))
        X_tr = panel.loc[tr_mask, cfg.feature_cols].to_numpy(np.float32)
        y_tr = panel.loc[tr_mask, cfg.target_col].to_numpy(np.float32)
        X_te = panel.loc[te_mask, cfg.feature_cols].to_numpy(np.float32)
        if len(X_tr) < 1000 or len(X_te) == 0:
            continue
        if transform is not None:
            X_tr = transform(X_tr)
            X_te = transform(X_te)
        m = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
        m.fit(X_tr, y_tr)
        block = panel.loc[te_mask, [cfg.date_col, "permno",
                                    cfg.target_col]].copy()
        block["y_pred"] = _safe_predict(m, X_te)
        preds.append(block)
    if not preds:
        return pd.DataFrame()
    out = pd.concat(preds, ignore_index=True)
    return out.rename(columns={cfg.target_col: "y_true"})


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_complexity_curves(sweep_raw: pd.DataFrame,
                           sweep_rff: pd.DataFrame,
                           save_dir: str) -> None:
    """Two-panel figure: OOS R^2 vs p, and Sharpe vs p, for both sweeps."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # OOS R^2 panel.
    ax = axes[0]
    if not sweep_raw.empty:
        ax.plot(sweep_raw["n_params"], sweep_raw["oos_r2"] * 100,
                marker="o", linewidth=2, label="Raw predictors")
    if not sweep_rff.empty:
        ax.plot(sweep_rff["n_params"], sweep_rff["oos_r2"] * 100,
                marker="s", linewidth=2, label="Random feature expansion")
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (p)")
    ax.set_ylabel("OOS R^2 (%)")
    ax.set_title("Virtue of Complexity: OOS R^2 vs Model Size")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Sharpe panel.
    ax = axes[1]
    if not sweep_raw.empty:
        ax.plot(sweep_raw["n_params"], sweep_raw["ls_sharpe"],
                marker="o", linewidth=2, label="Raw predictors")
    if not sweep_rff.empty:
        ax.plot(sweep_rff["n_params"], sweep_rff["ls_sharpe"],
                marker="s", linewidth=2, label="Random feature expansion")
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (p)")
    ax.set_ylabel("Long-Short Sharpe (ann.)")
    ax.set_title("Virtue of Complexity: Strategy Sharpe vs Model Size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "complexity_curves.png"), dpi=140)
    plt.close(fig)


def run_complexity_analysis(panel: pd.DataFrame,
                            feature_cols: List[str],
                            train_end_date: str,
                            save_dir: str,
                            refit_freq_months: int = 12) -> Dict[str, pd.DataFrame]:
    """End-to-end orchestrator called by main.py."""
    print("\n=== Complexity Analysis: Sweep A (raw predictors) ===")
    # Adapt the requested grid {5,10,20,50,all} to however many predictors
    # we actually have (we use 9 accounting variables by default).
    n = len(feature_cols)
    raw_grid = sorted({1, max(2, n // 4), max(3, n // 2), n})
    raw_grid = [g for g in raw_grid if g >= 1 and g <= n]
    sweep_a = sweep_raw_features(panel, feature_cols, raw_grid,
                                 train_end_date, refit_freq_months)

    print("\n=== Complexity Analysis: Sweep B (random feature expansion) ===")
    rff_widths = [2, 5, 10, 20, 50, 100, 200, 500]
    sweep_b = sweep_random_features(panel, feature_cols, rff_widths,
                                    train_end_date, refit_freq_months,
                                    alpha=1.0, bandwidth=1.0)

    plot_complexity_curves(sweep_a, sweep_b, save_dir)
    sweep_a.to_csv(os.path.join(save_dir, "complexity_sweep_raw.csv"),
                   index=False)
    sweep_b.to_csv(os.path.join(save_dir, "complexity_sweep_rff.csv"),
                   index=False)
    return {"raw": sweep_a, "rff": sweep_b}
