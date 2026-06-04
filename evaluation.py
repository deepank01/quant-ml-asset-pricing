"""
evaluation.py
=============
Performance diagnostics for ML-based asset-pricing strategies.

Each model produces a stock-level prediction stream; this file translates
that stream into:
    * Annualised mean return & Sharpe of the long-short portfolio
    * Maximum drawdown
    * CAPM alpha and beta (regressing the L/S on the market excess return)
    * Out-of-sample R^2 (Gu/Kelly/Xiu definition -- not demeaned)
    * Cumulative wealth chart
    * Decile bar chart (monotonicity diagnostic)
    * Feature importances for tree models

The OOS R^2 definition matters: GKX measure R^2 against zero (not the
sample mean) because in finance the "naive" forecast is r_i,t+1 = 0,
not r_i,t+1 = bar{r}. We follow that convention here.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib
# Note: we intentionally do NOT call matplotlib.use() at import time. Setting
# the backend in library code overrides whatever the caller wanted (e.g. the
# `inline` backend in a Jupyter notebook). main.py picks a backend if needed;
# the notebook uses %matplotlib inline.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

ANN_FACTOR = 12  # monthly returns -> annualised


# --------------------------------------------------------------------------- #
# Scalar performance metrics                                                  #
# --------------------------------------------------------------------------- #
def annualised_mean(ret: pd.Series) -> float:
    """Annualised mean of a monthly return series."""
    return float(ret.dropna().mean() * ANN_FACTOR)


def annualised_vol(ret: pd.Series) -> float:
    """Annualised volatility of a monthly return series."""
    return float(ret.dropna().std() * np.sqrt(ANN_FACTOR))


def sharpe_ratio(ret: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe. `rf` should be the monthly risk-free rate."""
    r = ret.dropna() - rf
    if r.std() == 0 or len(r) == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ANN_FACTOR))


def max_drawdown(ret: pd.Series) -> float:
    """
    Worst peak-to-trough drawdown of the cumulative-return path. Returned
    as a negative number, e.g. -0.25 = -25%. The conventional measure of
    pain in long-horizon evaluation.
    """
    r = ret.dropna()
    if len(r) == 0:
        return float("nan")
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1
    return float(dd.min())


def capm_alpha_beta(strategy_ret: pd.Series,
                    market_ret: pd.Series,
                    rf_rate: float = 0.0) -> Dict[str, float]:
    """
    OLS regression r_strat = alpha + beta * (r_mkt - rf) + eps.

    Returns annualised alpha (12 * monthly intercept), beta, and the
    Newey-West-ish t-stat on alpha (here just OLS for simplicity; the
    finance literature would use NW with ~6 lags).
    """
    df = pd.concat([strategy_ret.rename("r_s"),
                    market_ret.rename("r_m")], axis=1).dropna()
    if len(df) < 12:
        return {"alpha_ann": float("nan"), "beta": float("nan"),
                "alpha_t": float("nan")}

    x = df["r_m"].to_numpy() - rf_rate
    y = df["r_s"].to_numpy() - rf_rate
    X = np.column_stack([np.ones_like(x), x])
    # Normal equations -- small enough to do directly.
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_m, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    n = len(y)
    sigma2 = (resid @ resid) / max(n - 2, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_alpha = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
    t_alpha = alpha_m / se_alpha if se_alpha > 0 else float("nan")
    return {
        "alpha_ann": alpha_m * ANN_FACTOR,
        "beta": beta,
        "alpha_t": t_alpha,
    }


def out_of_sample_r2(predictions: pd.DataFrame) -> float:
    """
    OOS R^2 in the GKX (2020) convention:
        R^2_oos = 1 - sum_i (y - y_hat)^2 / sum_i y^2
    Note the denominator is sum of squared realised returns (no demeaning).
    Typical monthly stock-level OOS R^2 for the *best* model in GKX is
    around 0.40%; that's the bar.
    """
    df = predictions.dropna(subset=["y_true", "y_pred"])
    if len(df) == 0:
        return float("nan")
    num = float(((df["y_true"] - df["y_pred"]) ** 2).sum())
    den = float((df["y_true"] ** 2).sum())
    return 1.0 - num / den if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Summary tables                                                              #
# --------------------------------------------------------------------------- #
def summarize_strategy(name: str,
                       strategy_ret: pd.Series,
                       market_ret: pd.Series,
                       predictions: Optional[pd.DataFrame] = None
                       ) -> Dict[str, float]:
    """One-row performance summary for the headline results table."""
    capm = capm_alpha_beta(strategy_ret, market_ret)
    return {
        "model":       name,
        "ann_return":  annualised_mean(strategy_ret),
        "ann_vol":     annualised_vol(strategy_ret),
        "sharpe":      sharpe_ratio(strategy_ret),
        "alpha_ann":   capm["alpha_ann"],
        "beta":        capm["beta"],
        "alpha_t":     capm["alpha_t"],
        "max_dd":      max_drawdown(strategy_ret),
        "oos_r2":      (out_of_sample_r2(predictions)
                        if predictions is not None else float("nan")),
        "n_months":    int(strategy_ret.dropna().shape[0]),
    }


def format_summary_table(summary_rows) -> str:
    """Pretty-print the headline table in the style the spec asks for."""
    header = (f"{'Model':<18} | {'Ann.Ret':>8} | {'Sharpe':>6} | "
              f"{'Alpha':>7} | {'Max DD':>7} | {'OOS R^2':>8}")
    sep = "-" * len(header)
    lines = [header, sep]
    for row in summary_rows:
        lines.append(
            f"{row['model']:<18} | "
            f"{row['ann_return']*100:>7.2f}% | "
            f"{row['sharpe']:>6.2f} | "
            f"{row['alpha_ann']*100:>6.2f}% | "
            f"{row['max_dd']*100:>6.1f}% | "
            f"{row['oos_r2']*100:>7.3f}%"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_cumulative_returns(strategy_rets: Dict[str, pd.Series],
                            save_path: str) -> None:
    """One line per model, cumulative ($1 invested) on a log scale."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in strategy_rets.items():
        r = r.dropna()
        if r.empty:
            continue
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, label=name, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_title("Cumulative Long-Short Portfolio Returns (log scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_decile_returns(decile_panels: Dict[str, pd.DataFrame],
                        save_path: str) -> None:
    """
    Side-by-side bars: average monthly return for D1..D10 for each model.
    Monotone increasing bars = the model successfully ranks stocks.
    """
    n = len(decile_panels)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             squeeze=False)
    for ax_idx, (name, dec) in enumerate(decile_panels.items()):
        ax = axes[ax_idx // cols][ax_idx % cols]
        means = dec.mean() * 100  # to %
        ax.bar(range(1, len(means) + 1), means.values,
               color=sns.color_palette("viridis", len(means)))
        ax.set_title(f"{name} — Decile mean monthly return")
        ax.set_xlabel("Decile (1=low pred, 10=high pred)")
        ax.set_ylabel("Mean monthly return (%)")
        ax.axhline(0, color="black", linewidth=0.8)
    # Hide unused axes.
    for k in range(len(decile_panels), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path: str,
                            title: str = "Feature Importance") -> None:
    """
    Bar plot of feature importances for tree models. Silently skips for
    models that don't expose .feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        return
    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(feature_names) + 1.5))
    ax.barh([feature_names[i] for i in order][::-1],
            importances[order][::-1],
            color=sns.color_palette("crest", len(feature_names))[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_drawdown(strategy_rets: Dict[str, pd.Series],
                  save_path: str) -> None:
    """Drawdown curves over time. Useful for seeing crisis-period pain."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in strategy_rets.items():
        r = r.dropna()
        if r.empty:
            continue
        cum = (1 + r).cumprod()
        dd = cum / cum.cummax() - 1
        ax.plot(dd.index, dd.values * 100, label=name, linewidth=1.4)
    ax.set_title("Drawdowns of Long-Short Portfolios")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)
    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
