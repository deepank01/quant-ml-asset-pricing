"""
data_pipeline.py
================
Build a clean panel of (permno, date, accounting characteristics, fwd_ret) for
ML training. Uses the openassetpricing package (Chen & Zimmermann's Open Source
Asset Pricing dataset) when WRDS credentials are available; otherwise falls back
to a realistic simulated panel with the same schema so the rest of the pipeline
still runs end-to-end.

Finance conventions enforced here:
* Fama-French 6-month accounting lag (e.g. FY ending Dec 2010 is only assumed
  known from July 2011 onward). This prevents look-ahead from delayed filings.
* Cross-sectional median imputation each month (firm characteristics are
  cross-sectional signals -- a missing BM should be replaced with this month's
  median BM, not last month's BM for this firm).
* Winsorization at 1/99 pct cross-sectionally per month, then rank-normalize
  to [-0.5, 0.5] cross-sectionally per month. Rank-normalization is the
  standard pre-processing in Gu/Kelly/Xiu (2020); it removes the influence of
  outliers and puts every characteristic on the same scale.
* The target is `fwd_ret`: the firm's return in month t+1 (lined up so that
  features at time t predict the next month's realised return).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Load WRDS_USERNAME, WRDS_PASSWORD from .env

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# The nine accounting predictors the spec asks us to prioritise. These are the
# OSAP "Acronym" column names, which is how dl_signal expects to receive them.
ACCOUNTING_PREDICTORS: List[str] = [
    "BM",              # Book-to-market (Rosenberg, Reid, Lanstein 1985)
    "AssetGrowth",     # Cooper, Gulen & Schill (2008)
    "GP",              # Gross profitability -- Novy-Marx (2013); OSAP acronym is "GP"
    "Accruals",        # Sloan (1996)
    "roaq",            # Return on assets (quarterly) -- proxy for ROE if RoE not available
    "InvestPPEInv",    # Investment-to-assets (Lyandres, Sun, Zhang 2008)
    "EntMult",         # Enterprise multiple, Loughran & Wellman (2011)
    "ChTax",           # Change in taxes, Thomas & Zhang (2002)
    "OperProf",        # Operating profitability (Fama & French 2015)
]


@dataclass
class PanelConfig:
    """Bundle of all knobs the pipeline accepts. Defaults follow the spec."""
    start_year: int = 1970
    end_year: int = 2023
    predictors: Tuple[str, ...] = tuple(ACCOUNTING_PREDICTORS)
    accounting_lag_months: int = 6      # Fama-French convention
    winsor_pct: float = 0.01            # 1% / 99% trim
    min_firms_per_month: int = 50       # drop sparse early months


# --------------------------------------------------------------------------- #
# WRDS Connection and CRSP Returns                                            #
# --------------------------------------------------------------------------- #
def _get_wrds_returns(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch monthly returns from CRSP (crsp.msf) via WRDS.
    """
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")

    if not username or not password:
        print("[data_pipeline] WRDS credentials not found in environment.")
        return None

    try:
        import wrds
        # Connect using the provided credentials
        db = wrds.Connection(wrds_username=username)
        # Note: wrds.Connection doesn't take a password directly in the constructor
        # for security, it usually prompts. However, we can use the .pgpass approach
        # or environment variables if the library supports it, or just rely on the
        # user having set it up.
        # Actually, for non-interactive use with password, we can create a ~/.pgpass file.
        # But a simpler way if the library allows is just to use it.
        # The `wrds` library uses psycopg2.
        
        print(f"[data_pipeline] Querying CRSP returns from {start_date} to {end_date} ...")
        sql = f"""
            SELECT permno, date, ret
            FROM crsp.msf
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        """
        df_ret = db.raw_sql(sql, date_cols=["date"])
        db.close()
        
        # Ensure date is end-of-month to match OSAP signals
        df_ret["date"] = pd.to_datetime(df_ret["date"]) + pd.offsets.MonthEnd(0)
        df_ret["permno"] = df_ret["permno"].astype(int)
        
        return df_ret
    except Exception as e:
        print(f"[data_pipeline] WRDS connection or query failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# Real-data path: openassetpricing                                            #
# --------------------------------------------------------------------------- #
def _try_openassetpricing(cfg: PanelConfig) -> Optional[pd.DataFrame]:
    """
    Attempt to download real firm characteristics via the openassetpricing
    package. Returns None on any failure (no internet, no WRDS creds, etc.) so
    the caller can fall back to simulation. We never raise -- the spec requires
    that the pipeline ALWAYS run.
    """
    try:
        import openassetpricing as oap
    except Exception as e:
        print(f"[data_pipeline] openassetpricing not importable ({e}); "
              "using simulated panel.")
        return None

    try:
        openap = oap.OpenAP()
        print("[data_pipeline] Downloading firm characteristics from OSAP "
              f"(predictors: {list(cfg.predictors)}) ...")
        # dl_signal returns (permno, yyyymm, <each signal as column>).
        df = openap.dl_signal("pandas", list(cfg.predictors))
    except Exception as e:
        # The most common failure is "needs WRDS account"; report and fall back.
        print(f"[data_pipeline] OSAP download failed ({e}); "
              "using simulated panel.")
        return None

    # Normalise schema -- different OSAP releases use slightly different names.
    df.columns = [c.lower() for c in df.columns]
    if "yyyymm" in df.columns:
        df["date"] = pd.to_datetime(df["yyyymm"].astype(str) + "01",
                                    format="%Y%m%d") + pd.offsets.MonthEnd(0)
        df = df.drop(columns=["yyyymm"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    else:
        print("[data_pipeline] Unexpected OSAP schema; using simulated panel.")
        return None

    # Filter to requested window BEFORE fetching returns to save time/bandwidth
    df = df[(df["date"].dt.year >= cfg.start_year) & 
            (df["date"].dt.year <= cfg.end_year)].copy()

    if df.empty:
        print("[data_pipeline] No OSAP data in requested range.")
        return None

    # Now fetch returns from WRDS to complete the panel
    start_str = df["date"].min().strftime("%Y-%m-%d")
    end_str = df["date"].max().strftime("%Y-%m-%d")
    
    df_ret = _get_wrds_returns(start_str, end_str)
    
    if df_ret is None:
        print("[data_pipeline] Could not fetch real returns; falling back to simulation.")
        return None

    # Merge signals and returns
    df["permno"] = df["permno"].astype(int)
    final_df = pd.merge(df, df_ret, on=["permno", "date"], how="inner")
    
    if final_df.empty:
        print("[data_pipeline] Merge of signals and returns resulted in empty panel.")
        return None

    print(f"[data_pipeline] Successfully built real panel: {len(final_df):,} rows.")
    return final_df


# --------------------------------------------------------------------------- #
# Simulation fallback                                                          #
# --------------------------------------------------------------------------- #
def _simulate_panel(cfg: PanelConfig, n_firms: int = 800,
                    seed: int = 42) -> pd.DataFrame:
    """
    Build a realistic synthetic panel that mimics CRSP/Compustat firm-month
    data. Each "firm" gets persistent characteristics that slowly drift, and
    returns are generated from a sparse linear model on (lagged) features
    plus heavy idiosyncratic noise. This guarantees ML models *can* find
    signal but won't dominate -- the same R^2 ballpark as the GKX paper
    (~0.4% monthly OOS R^2 for the best models).
    """
    rng = np.random.default_rng(seed)
    n_predictors = len(cfg.predictors)

    # Month index, end-of-month.
    months = pd.date_range(
        start=f"{cfg.start_year}-01-31",
        end=f"{cfg.end_year}-12-31",
        freq="ME",
    )
    n_months = len(months)
    print(f"[data_pipeline] Simulating panel: {n_firms} firms x "
          f"{n_months} months ({n_predictors} accounting predictors).")

    # Generate AR(1) characteristic processes per firm. Accounting ratios are
    # highly persistent month-to-month (book value changes slowly), so we set
    # phi ~ 0.97. This is close to estimates in Lewellen (2015) and others.
    phi = 0.97
    sigma_innov = np.sqrt(1 - phi ** 2)  # keeps unconditional var ~ 1

    # Pre-allocate. Shape: (n_months, n_firms, n_predictors)
    X = np.empty((n_months, n_firms, n_predictors), dtype=np.float32)
    X[0] = rng.standard_normal((n_firms, n_predictors)).astype(np.float32)
    for t in range(1, n_months):
        innov = rng.standard_normal((n_firms, n_predictors)) * sigma_innov
        X[t] = phi * X[t - 1] + innov.astype(np.float32)

    # "True" sparse betas: only some predictors really matter, and their
    # effect on returns is small (~10-20 bps/month per std move of a
    # characteristic). This matches Kelly/Pruitt/Su (2019) empirical magnitudes.
    true_beta = np.zeros(n_predictors, dtype=np.float32)
    # Pick ~half of predictors as "true" signals with mixed signs.
    n_true = max(1, n_predictors // 2)
    signal_idx = rng.choice(n_predictors, size=n_true, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_true).astype(np.float32)
    # Effect sizes drawn so a one-std-move in a signal moves expected return
    # by 10-25 bps per month.
    magnitudes = rng.uniform(0.001, 0.0025, size=n_true).astype(np.float32)
    true_beta[signal_idx] = signs * magnitudes

    # Realized return = signal + idio. Idio std ~10%/month (typical for US equities).
    idio = rng.standard_normal((n_months, n_firms)).astype(np.float32) * 0.10
    expected_ret = X @ true_beta                      # (n_months, n_firms)
    realised_ret = expected_ret + idio

    # Build long-format dataframe.
    permnos = np.arange(10000, 10000 + n_firms)
    records = []
    for t, dt in enumerate(months):
        block = pd.DataFrame(X[t], columns=list(cfg.predictors))
        block.insert(0, "date", dt)
        block.insert(1, "permno", permnos)
        block["ret"] = realised_ret[t]
        records.append(block)
    df = pd.concat(records, ignore_index=True)

    # Inject realistic missingness: ~10% of (firm-month, predictor) cells
    # missing-at-random, simulating Compustat coverage holes.
    for col in cfg.predictors:
        mask = rng.random(len(df)) < 0.10
        df.loc[mask, col] = np.nan

    # Simulate firm entry/exit so the panel is unbalanced (firms IPO and
    # delist). Each firm has a uniform start month and end month.
    firm_start = rng.integers(0, n_months // 3, size=n_firms)
    firm_end = rng.integers(2 * n_months // 3, n_months, size=n_firms)
    firm_alive = pd.DataFrame({
        "permno": permnos,
        "first_month": months[firm_start],
        "last_month": months[firm_end],
    })
    df = df.merge(firm_alive, on="permno", how="left")
    df = df[(df["date"] >= df["first_month"]) &
            (df["date"] <= df["last_month"])].copy()
    df = df.drop(columns=["first_month", "last_month"])

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pre-processing: lag, winsorize, rank-normalize                              #
# --------------------------------------------------------------------------- #
def _apply_accounting_lag(df: pd.DataFrame, predictors: List[str],
                          lag_months: int) -> pd.DataFrame:
    """
    Push each firm's characteristics forward by `lag_months` so that, at the
    portfolio formation date, we are only using information already public.

    Fama-French convention: an accounting variable from fiscal year ending in
    calendar year y is used for portfolio formation starting July of y+1 (so
    a 6-month lag if you treat year-end as January).
    """
    out = df.sort_values(["permno", "date"]).copy()
    out[predictors] = (out.groupby("permno", sort=False)[predictors]
                          .shift(lag_months))
    return out


def _cross_sectional_median_impute(df: pd.DataFrame,
                                   predictors: List[str]) -> pd.DataFrame:
    """
    Fill missing characteristics with this month's cross-sectional median.
    This is the standard GKX/JKP approach: a firm with no BM gets the typical
    BM of firms trading this month, not a stale value of its own.
    """
    out = df.copy()
    for col in predictors:
        med = out.groupby("date")[col].transform("median")
        out[col] = out[col].fillna(med)
    # Edge case: a month where every firm is missing a predictor -- fill 0.
    out[predictors] = out[predictors].fillna(0.0)
    return out


def _winsorize_cross_section(df: pd.DataFrame, predictors: List[str],
                             pct: float) -> pd.DataFrame:
    """
    Cap each predictor at the [pct, 1-pct] percentiles within each month.
    Done cross-sectionally so no information from other months leaks in --
    this keeps it look-ahead-safe.
    """
    out = df.copy()
    for col in predictors:
        lo = out.groupby("date")[col].transform(lambda x: x.quantile(pct))
        hi = out.groupby("date")[col].transform(lambda x: x.quantile(1 - pct))
        out[col] = np.clip(out[col], lo, hi)
    return out


def _rank_normalize_cross_section(df: pd.DataFrame,
                                  predictors: List[str]) -> pd.DataFrame:
    """
    Map each predictor to its cross-sectional rank in [-0.5, +0.5] each month.
    This is the GKX pre-processing: rank within month, divide by N+1, subtract
    0.5. Puts every characteristic on the same scale and makes ML training
    well-conditioned without leaking future info.
    """
    out = df.copy()
    for col in predictors:
        out[col] = (out.groupby("date")[col]
                       .rank(method="average", pct=True)) - 0.5
    return out


def _attach_forward_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each firm, line up the *next* month's return as the prediction target.
    Features at time t are used to predict ret at t+1. This is the standard
    asset-pricing setup and is what prevents look-ahead bias in the target.

    Implementation note: we do this by an explicit *date-aware* self-merge,
    NOT a positional `groupby.shift(-1)`. Positional shift is unsafe when
    the panel has gaps (e.g. a stock temporarily delists and re-lists, or
    skips months because of missing CRSP coverage): it would attach a
    return from several months in the future as if it were next month's.
    """
    out = df.sort_values(["permno", "date"]).copy()
    next_date = out["date"] + pd.offsets.MonthEnd(1)
    # Build a lookup of (permno, date) -> ret, then look up (permno, next_date).
    ret_lookup = out.set_index(["permno", "date"])["ret"]
    keys = list(zip(out["permno"].to_numpy(), next_date.to_numpy()))
    out["fwd_ret"] = ret_lookup.reindex(keys).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def build_panel(cfg: Optional[PanelConfig] = None,
                save_path: Optional[str] = None,
                n_firms: int = 800) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the final ML-ready panel.

    Returns a dataframe with columns:
        permno, date, <predictors...>, ret, fwd_ret

    Pre-processing pipeline:
        1. Load raw panel (OSAP or simulated)
        2. Apply 6-month accounting lag
        3. Median impute missing predictors cross-sectionally
        4. Winsorize at 1/99 pct cross-sectionally
        5. Rank-normalize to [-0.5, 0.5] cross-sectionally
        6. Attach fwd_ret = next month's ret
        7. Drop rows with missing fwd_ret (e.g. last month for each firm)
        8. Drop sparse months
    """
    if cfg is None:
        cfg = PanelConfig()
    
    raw = _try_openassetpricing(cfg)
    if raw is None:
        raw = _simulate_panel(cfg, n_firms=n_firms)

    # Restrict to the requested window.
    raw = raw[(raw["date"].dt.year >= cfg.start_year) &
              (raw["date"].dt.year <= cfg.end_year)].copy()

    # Determine the actual predictor column names (OSAP often returns lowercase)
    actual_cols = []
    for p in cfg.predictors:
        if p in raw.columns:
            actual_cols.append(p)
        elif p.lower() in raw.columns:
            actual_cols.append(p.lower())
        else:
            # Fallback to the original if not found, it will error later if still missing
            actual_cols.append(p)
    predictors = actual_cols

    print("[data_pipeline] Applying Fama-French accounting lag "
          f"({cfg.accounting_lag_months} months).")
    df = _apply_accounting_lag(raw, predictors, cfg.accounting_lag_months)

    print("[data_pipeline] Cross-sectional median imputation.")
    df = _cross_sectional_median_impute(df, predictors)

    print(f"[data_pipeline] Winsorizing at {cfg.winsor_pct:.0%} / "
          f"{1 - cfg.winsor_pct:.0%}.")
    df = _winsorize_cross_section(df, predictors, cfg.winsor_pct)

    print("[data_pipeline] Rank-normalizing cross-sectionally to [-0.5, 0.5].")
    df = _rank_normalize_cross_section(df, predictors)

    print("[data_pipeline] Attaching forward (t+1) returns.")
    df = _attach_forward_return(df)

    # Drop rows we can't train on.
    before = len(df)
    df = df.dropna(subset=["fwd_ret"])
    df = df.dropna(subset=predictors)  # post-lag NaNs at firm start
    print(f"[data_pipeline] Dropped {before - len(df):,} rows missing target "
          f"or features after lag.")

    # Drop months that are too sparse to compute cross-section statistics.
    counts = df.groupby("date").size()
    keep_months = counts[counts >= cfg.min_firms_per_month].index
    df = df[df["date"].isin(keep_months)].copy()

    df = df.sort_values(["date", "permno"]).reset_index(drop=True)
    print(f"[data_pipeline] Final panel: {len(df):,} firm-months, "
          f"{df['permno'].nunique():,} firms, "
          f"{df['date'].nunique():,} months "
          f"({df['date'].min().date()} -> {df['date'].max().date()}).")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_parquet(save_path) if save_path.endswith(".parquet") \
            else df.to_csv(save_path, index=False)
        print(f"[data_pipeline] Saved panel to {save_path}.")

    return df, predictors


if __name__ == "__main__":
    panel = build_panel()
    print(panel.head())
    print(panel.describe().T)
