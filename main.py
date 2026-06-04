"""
main.py
=======
End-to-end runner for the ML asset-pricing project.

Usage:
    python main.py

Pipeline:
    1. Build the firm-month panel (data_pipeline)
    2. Train every model with walk-forward expanding windows (models)
    3. Form decile portfolios and the long-short spread (portfolio)
    4. Evaluate Sharpe, alpha, drawdown, OOS R^2 (evaluation)
    5. Run the "virtue of complexity" sweeps (complexity_analysis)
    6. Save all charts + CSVs under ./outputs/
"""

from __future__ import annotations

import os
from typing import Dict

# Pick a non-interactive backend before any matplotlib import in our libraries.
# Safe for `python main.py` headless runs; the notebook uses %matplotlib inline.
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from complexity_analysis import run_complexity_analysis
from data_pipeline import ACCOUNTING_PREDICTORS, PanelConfig, build_panel
from evaluation import (format_summary_table, plot_cumulative_returns,
                        plot_decile_returns, plot_drawdown,
                        plot_feature_importance, summarize_strategy)
from models import (ExpandingWindowConfig, _safe_predict, get_model_registry,
                    run_all_models)
from portfolio import (build_decile_portfolios, decile_summary,
                       long_short_returns, market_return_series)


OUTPUT_DIR = "outputs"
TRAIN_END_DATE = "2000-12-31"   # train pre-2001, test post-2001
REFIT_FREQ_MONTHS = 12           # re-train annually


def _fit_full_model_for_importance(panel: pd.DataFrame,
                                   feature_cols: list,
                                   model_name: str):
    """
    Re-fit a chosen model on the entire training window so we can read off
    feature importances.

    No-look-ahead rule: a training row dated `t` has `fwd_ret` equal to the
    return realised between t and t+1. The first test month is
    TRAIN_END_DATE + 1 month, so any training row dated *within* one month
    of TRAIN_END_DATE would have a target that lands inside the test window.
    We therefore require date < TRAIN_END_DATE (strictly), and additionally
    drop the final training month so the latest training-row target is the
    return realised over the month ending at TRAIN_END_DATE itself --
    safely *before* the test window opens.
    """
    registry = get_model_registry()
    if model_name not in registry:
        return None
    cutoff = pd.Timestamp(TRAIN_END_DATE) - pd.offsets.MonthEnd(1)
    train_panel = panel[panel["date"] < cutoff]
    X = train_panel[feature_cols].to_numpy()
    y = train_panel["fwd_ret"].to_numpy()
    model = registry[model_name]()
    try:
        model.fit(X, y)
    except Exception as e:
        print(f"[main] full-fit for {model_name} failed: {e}")
        return None
    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1. Build panel ----------
    print("\n" + "=" * 60)
    print("STEP 1: Build firm-month panel")
    print("=" * 60)
    cfg = PanelConfig(
        start_year=1970,
        end_year=2023,
        predictors=tuple(ACCOUNTING_PREDICTORS),
    )
    panel, feature_cols = build_panel(cfg=cfg,
                                      save_path=os.path.join(OUTPUT_DIR, "panel.parquet"))

    # ---------- 2. Train all ML models ----------
    print("\n" + "=" * 60)
    print("STEP 2: Train ML models (expanding window, refit annually)")
    print("=" * 60)
    predictions: Dict[str, pd.DataFrame] = run_all_models(
        panel,
        feature_cols=feature_cols,
        train_end_date=TRAIN_END_DATE,
        refit_freq_months=REFIT_FREQ_MONTHS,
        verbose=True,
    )

    # ---------- 3. Portfolios + 4. Evaluation ----------
    print("\n" + "=" * 60)
    print("STEP 3+4: Portfolio construction and evaluation")
    print("=" * 60)

    mkt = market_return_series(panel)
    summary_rows = []
    strategy_returns: Dict[str, pd.Series] = {}
    decile_panels: Dict[str, pd.DataFrame] = {}

    for model_name, preds in predictions.items():
        if preds.empty:
            print(f"[main] {model_name}: no predictions produced, skipping.")
            continue
        deciles = build_decile_portfolios(preds)
        ls = long_short_returns(deciles)
        decile_panels[model_name] = deciles
        strategy_returns[model_name] = ls

        row = summarize_strategy(model_name, ls, mkt, predictions=preds)
        summary_rows.append(row)

        # Per-model decile table on disk for the appendix.
        decile_summary(deciles).to_csv(
            os.path.join(OUTPUT_DIR, f"deciles_{model_name}.csv"),
            index=False,
        )

    # Headline results table.
    if summary_rows:
        print("\n=== ML Asset Pricing Results ===")
        print(format_summary_table(summary_rows))
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUTPUT_DIR, "summary_table.csv"), index=False
        )

    # Charts.
    plot_cumulative_returns(
        strategy_returns, os.path.join(OUTPUT_DIR, "cumulative_returns.png")
    )
    plot_drawdown(
        strategy_returns, os.path.join(OUTPUT_DIR, "drawdowns.png")
    )
    plot_decile_returns(
        decile_panels, os.path.join(OUTPUT_DIR, "decile_bars.png")
    )

    # Feature importance for the tree-based models. We re-fit on the full
    # training window once each, just to read off .feature_importances_.
    for tree_model in ("RandomForest", "GBM"):
        m = _fit_full_model_for_importance(panel, feature_cols, tree_model)
        if m is not None:
            plot_feature_importance(
                m, feature_cols,
                os.path.join(OUTPUT_DIR, f"feature_importance_{tree_model}.png"),
                title=f"{tree_model} feature importances "
                      f"(trained <= {TRAIN_END_DATE})",
            )

    # ---------- 5. Virtue of complexity ----------
    print("\n" + "=" * 60)
    print("STEP 5: Virtue-of-complexity analysis")
    print("=" * 60)
    sweeps = run_complexity_analysis(
        panel, feature_cols=feature_cols,
        train_end_date=TRAIN_END_DATE,
        save_dir=OUTPUT_DIR,
        refit_freq_months=REFIT_FREQ_MONTHS,
    )

    # Comparison: Ridge with the *fewest* vs the *most* features.
    if not sweeps["raw"].empty and len(sweeps["raw"]) >= 2:
        simple = sweeps["raw"].iloc[0]
        complex_ = sweeps["raw"].iloc[-1]
        print(f"\nRidge simple (p={int(simple['n_params'])}): "
              f"Sharpe={simple['ls_sharpe']:.2f}, "
              f"OOS R^2={simple['oos_r2']*100:+.3f}%")
        print(f"Ridge complex (p={int(complex_['n_params'])}): "
              f"Sharpe={complex_['ls_sharpe']:.2f}, "
              f"OOS R^2={complex_['oos_r2']*100:+.3f}%")
    if not sweeps["rff"].empty:
        best_rff = sweeps["rff"].loc[sweeps["rff"]["ls_sharpe"].idxmax()]
        print(f"Best RFF Ridge (p={int(best_rff['n_params'])}): "
              f"Sharpe={best_rff['ls_sharpe']:.2f}, "
              f"OOS R^2={best_rff['oos_r2']*100:+.3f}%")

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
