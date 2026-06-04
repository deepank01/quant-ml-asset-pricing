"""
portfolio.py
============
Translate stock-level predictions into tradable portfolio returns.

Each month we:
    1. Rank all stocks by predicted return (high -> low)
    2. Sort into 10 equal-sized buckets (deciles)
    3. Equal-weight within each decile
    4. Form the long-short = (top decile) - (bottom decile)

This is the textbook decile-sort test of cross-sectional return prediction
(Fama & French 1992, Gu/Kelly/Xiu 2020). If a model has real predictive
power, the decile portfolio returns should be roughly monotone in
predicted rank, and the L/S spread should earn a positive average return.

Conventions:
    * Each row in the predictions dataframe carries the formation date `t`,
      the prediction `y_pred` made using information available at `t`, and
      `y_true = fwd_ret(t)` = the realised return earned between t and t+1.
    * Deciles are formed cross-sectionally per formation month using
      `y_pred`. The implied trade is to buy/sell at the close of month `t`.
    * The realised return of the resulting decile portfolio is then earned
      in month t+1. `build_decile_portfolios` shifts the output index by
      +1 month so it reads as the realised-return calendar month, which
      lets CAPM regressions align cleanly with `market_return_series`
      (also indexed by the realised-return month).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _assign_deciles(group: pd.DataFrame, n_bins: int = 10) -> pd.Series:
    """
    Cross-sectional decile assignment. Uses `qcut` which handles duplicate
    edges by ranking first -- important when predictions cluster (e.g. ridge
    predictions can be nearly identical for many stocks).
    """
    ranks = group["y_pred"].rank(method="first")  # break ties by row order
    # `qcut` with `duplicates='drop'` would silently drop bins; rank-first
    # guarantees we always get exactly n_bins.
    return pd.qcut(ranks, q=n_bins, labels=False) + 1  # 1..n_bins


def build_decile_portfolios(predictions: pd.DataFrame,
                            n_bins: int = 10,
                            min_stocks_per_month: int = 30) -> pd.DataFrame:
    """
    Build monthly decile portfolio returns from stock-level predictions.

    Parameters
    ----------
    predictions : DataFrame with columns [date, permno, y_true, y_pred]
                  where `date` is the FORMATION date (end of month t) and
                  `y_true` is the realised return earned over month t -> t+1.
    n_bins      : number of buckets (default 10 -> deciles)
    min_stocks_per_month : drop months too sparse to sort

    Returns
    -------
    DataFrame indexed by the *realised-return* month (i.e. formation date
    + 1 month), columns 1..n_bins each holding the equal-weighted realised
    return of that decile in that month.

    The index shift matters: it lines up the strategy return series with
    a contemporaneous market-return series in calendar time, so CAPM
    alphas and betas are not off by one month. Without the shift, the
    portfolio "labeled month t" would be the return earned in month t+1,
    while a market series indexed by month t would be the return in month
    t -- creating a one-month misalignment in the regression.
    """
    df = predictions.copy()

    # Throw out months that are too small to make decile sorts meaningful.
    counts = df.groupby("date").size()
    good_months = counts[counts >= min_stocks_per_month].index
    df = df[df["date"].isin(good_months)].copy()

    # Assign deciles cross-sectionally per month.
    df["decile"] = (df.groupby("date", group_keys=False)
                      .apply(lambda g: _assign_deciles(g, n_bins)))

    # Equal-weighted realised return per decile per FORMATION month.
    pivot = (df.groupby(["date", "decile"])["y_true"]
               .mean()
               .unstack("decile")
               .sort_index())

    # Shift the index from formation date to realised-return month.
    pivot.index = pivot.index + pd.offsets.MonthEnd(1)
    pivot.index.name = "date"

    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot.reindex(columns=range(1, n_bins + 1))
    return pivot


def long_short_returns(deciles: pd.DataFrame,
                       n_bins: Optional[int] = None) -> pd.Series:
    """
    Long-short spread = top decile minus bottom decile.

    A positive average L/S return is the cleanest piece of evidence that the
    model is picking up real cross-sectional return predictability. By
    netting top and bottom we strip out most of the market factor (the L/S
    is roughly dollar-neutral by construction).
    """
    if n_bins is None:
        n_bins = int(deciles.columns.max())
    ls = deciles[n_bins] - deciles[1]
    ls.name = "long_short"
    return ls


def decile_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised mean return and Sharpe for each decile + L/S spread.

    Useful for the "monotonicity" check: deciles 1..10 should show a
    roughly monotonic average return pattern if the model has signal.
    """
    rows = []
    for col in deciles.columns:
        r = deciles[col].dropna()
        rows.append({
            "bucket": f"D{col}",
            "mean_monthly": r.mean(),
            "ann_return":   r.mean() * 12,
            "ann_vol":      r.std() * np.sqrt(12),
            "sharpe":       (r.mean() / r.std() * np.sqrt(12)
                             if r.std() > 0 else np.nan),
            "n_months":     len(r),
        })
    ls = long_short_returns(deciles).dropna()
    rows.append({
        "bucket": "LongShort",
        "mean_monthly": ls.mean(),
        "ann_return":   ls.mean() * 12,
        "ann_vol":      ls.std() * np.sqrt(12),
        "sharpe":       (ls.mean() / ls.std() * np.sqrt(12)
                         if ls.std() > 0 else np.nan),
        "n_months":     len(ls),
    })
    return pd.DataFrame(rows)


def market_return_series(panel: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted "market" return from the same panel of stocks. We use
    this as the market proxy for CAPM regressions. Real research code would
    use the value-weighted CRSP market or Fama-French Mkt-RF -- equal-weighted
    is a defensible proxy when market caps aren't available.

    Alignment: `ret` on a row dated month t is the realised return earned
    in month t (it is NOT a forward return; that's `fwd_ret`). So this
    series is indexed by the realised-return month, matching the convention
    that `build_decile_portfolios` enforces. CAPM regressions therefore
    align strategy and market returns in the same calendar month.
    """
    mkt = panel.groupby("date")["ret"].mean()
    mkt.name = "mkt"
    return mkt
