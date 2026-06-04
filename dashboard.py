"""
dashboard.py
============
Streamlit dashboard for the ML Asset Pricing pipeline.

Run with:
    streamlit run dashboard.py

The full pipeline (data → models → portfolios → evaluation → complexity)
runs on first load and is cached so subsequent interactions are instant.
Change any sidebar setting and the relevant stages re-run automatically.
"""

from __future__ import annotations

import sys
import os
import time
import traceback
from typing import Dict, List, Optional

# ── Streamlit must be imported before matplotlib so it can set its own
#    backend before our modules touch matplotlib.
import streamlit as st

# Now safe to set Agg so our library code never accidentally triggers a GUI.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local pipeline modules.
sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import ACCOUNTING_PREDICTORS, PanelConfig, build_panel
from models import get_model_registry, run_all_models
from portfolio import (
    build_decile_portfolios,
    decile_summary,
    long_short_returns,
    market_return_series,
)
from evaluation import (
    capm_alpha_beta,
    max_drawdown,
    out_of_sample_r2,
    sharpe_ratio,
    summarize_strategy,
)
from complexity_analysis import run_complexity_analysis

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Asset Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent across all charts)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Ridge":        "#4C72B0",
    "RandomForest": "#DD8452",
    "GBM":          "#55A868",
    "MLP":          "#C44E52",
    "PLS":          "#8172B2",
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — pipeline configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Data window")
    start_year = st.slider("Start year", 1975, 2010, 1990)
    end_year   = st.slider("End year",   2005, 2023, 2015)
    train_end  = st.slider(
        "Train / test split",
        start_year + 5, end_year - 2, min(2000, end_year - 3),
        help="Models are trained on data strictly before this year; "
             "portfolio performance is evaluated on data after it.",
    )
    train_end_date = f"{train_end}-12-31"

    st.subheader("Models")
    all_model_names = list(get_model_registry().keys())
    selected_models = st.multiselect(
        "Models to run",
        all_model_names,
        default=all_model_names,
    )

    st.subheader("Training")
    refit_freq = st.select_slider(
        "Refit frequency (months)",
        options=[6, 12, 24, 36],
        value=24,
        help="How often to retrain each model on the expanding window. "
             "Smaller = more accurate but slower.",
    )

    n_firms = st.select_slider(
        "Simulated firms",
        options=[200, 400, 600, 800],
        value=400,
        help="Panel size for the simulated dataset. Larger = noisier but "
             "more realistic. Only affects the simulation fallback.",
    )

    st.subheader("Complexity analysis")
    run_complexity = st.checkbox("Run complexity sweep", value=True)

    st.markdown("---")
    run_btn = st.button("▶ Run pipeline", type="primary", use_container_width=True)
    st.caption(
        "Results are cached — re-running with the same settings is instant. "
        "Change any setting to trigger a fresh run."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Cached pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_build_panel(start_year, end_year, n_firms):
    cfg = PanelConfig(
        start_year=start_year,
        end_year=end_year,
        predictors=tuple(ACCOUNTING_PREDICTORS),
    )
    # Using the public entry point which now handles real/simulated logic
    df, feature_cols = build_panel(cfg, n_firms=n_firms)
    return df, feature_cols


@st.cache_data(show_spinner=False)
def cached_run_models(panel_hash, train_end_date, refit_freq, selected_models_tuple):
    # panel is passed by hash (st.cache_data hashes dataframe content).
    panel, feature_cols = _panel_store["panel"], _panel_store["feature_cols"]
    return run_all_models(
        panel,
        feature_cols=feature_cols,
        train_end_date=train_end_date,
        refit_freq_months=refit_freq,
        models_to_run=list(selected_models_tuple),
        verbose=False,
    )


@st.cache_data(show_spinner=False)
def cached_complexity(panel_hash, train_end_date, refit_freq):
    panel, feature_cols = _panel_store["panel"], _panel_store["feature_cols"]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sweeps = run_complexity_analysis(
            panel, feature_cols,
            train_end_date=train_end_date,
            save_dir=tmp,
            refit_freq_months=refit_freq,
        )
    return sweeps


# Module-level store so cached functions can retrieve the panel without
# passing the full dataframe as a cache key (which would be slow to hash
# on every call). We store it here and pass a lightweight hash string instead.
_panel_store: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────────────────────────────────────

LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#ddd", borderwidth=1),
)


def fig_cumulative(strategies: Dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in strategies.items():
        r = r.dropna()
        cum = (1 + r).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name=name,
            line=dict(color=MODEL_COLORS.get(name), width=2),
            hovertemplate="%{x|%b %Y}<br>$%{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Cumulative growth of $1 — long-short portfolio",
        xaxis_title="Date", yaxis_title="Portfolio value ($)",
        yaxis_type="log", height=420,
    )
    return fig


def fig_drawdown(strategies: Dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in strategies.items():
        r = r.dropna()
        cum = (1 + r).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        hex_color = MODEL_COLORS.get(name, "#888888")
        # Plotly 6 dropped support for 8-digit hex (#RRGGBBAA); use rgba() instead.
        r_int = int(hex_color[1:3], 16)
        g_int = int(hex_color[3:5], 16)
        b_int = int(hex_color[5:7], 16)
        fill_rgba = f"rgba({r_int},{g_int},{b_int},0.13)"
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name=name,
            line=dict(color=hex_color, width=1.8),
            fill="tozeroy",
            fillcolor=fill_rgba,
            hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Drawdowns",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=360,
    )
    return fig


def fig_decile_bars(deciles: Dict[str, pd.DataFrame]) -> go.Figure:
    n = len(deciles)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    model_names = list(deciles.keys())
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=model_names,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )
    for idx, (name, dec) in enumerate(deciles.items()):
        r, c = divmod(idx, cols)
        means = dec.mean() * 100
        n_bins = len(means)
        colors = px.colors.sample_colorscale("Viridis", [i / (n_bins - 1) for i in range(n_bins)])
        fig.add_trace(
            go.Bar(
                x=list(range(1, n_bins + 1)), y=means.values,
                marker_color=colors,
                showlegend=False,
                name=name,
                hovertemplate="D%{x}<br>%{y:.3f}%<extra>" + name + "</extra>",
            ),
            row=r + 1, col=c + 1,
        )
        fig.add_hline(y=0, line_width=0.8, line_color="black", row=r + 1, col=c + 1)
    fig.update_layout(
        **LAYOUT_BASE,
        title="Decile mean monthly returns (D1 = low predicted, D10 = high predicted)",
        height=280 * rows,
    )
    fig.update_xaxes(title_text="Decile")
    fig.update_yaxes(title_text="Mean monthly ret (%)")
    return fig


def fig_summary_bar(summary_rows: list, metric: str, label: str) -> go.Figure:
    df = pd.DataFrame(summary_rows).sort_values(metric, ascending=False)
    fig = go.Figure(go.Bar(
        x=df["model"], y=df[metric] * 100,
        marker_color=[MODEL_COLORS.get(m, "#888") for m in df["model"]],
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=0.8, line_color="black")
    fig.update_layout(**LAYOUT_BASE, title=label, yaxis_title=label, height=320)
    return fig


def fig_complexity(sweeps: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["OOS R² vs model size", "Long-short Sharpe vs model size"],
        horizontal_spacing=0.12,
    )
    styles = {
        "raw": dict(mode="lines+markers", marker_symbol="circle",
                    line=dict(width=2.5), name="Raw predictors"),
        "rff": dict(mode="lines+markers", marker_symbol="square",
                    line=dict(width=2.5, dash="dash"), name="RFF expansion"),
    }
    colors = {"raw": "#4C72B0", "rff": "#DD8452"}
    for key, df in sweeps.items():
        if df.empty:
            continue
        s = styles[key]
        fig.add_trace(
            go.Scatter(x=df["n_params"], y=df["oos_r2"] * 100,
                       line_color=colors[key], **s,
                       hovertemplate="p=%{x}<br>OOS R²=%{y:.4f}%<extra>" + s["name"] + "</extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["n_params"], y=df["ls_sharpe"],
                       line_color=colors[key], showlegend=False,
                       mode=s["mode"], marker_symbol=s["marker_symbol"],
                       line=dict(width=s["line"]["width"],
                                 dash=s["line"].get("dash", "solid")),
                       hovertemplate="p=%{x}<br>Sharpe=%{y:.3f}<extra>" + s["name"] + "</extra>"),
            row=1, col=2,
        )
    fig.add_hline(y=0, line_width=0.8, line_color="black", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Number of parameters (p)")
    fig.update_yaxes(title_text="OOS R² (%)", row=1, col=1)
    fig.update_yaxes(title_text="Annualised Sharpe", row=1, col=2)
    fig.update_layout(**LAYOUT_BASE, title="The Virtue of Complexity (Kelly, Malamud & Zhou 2024)", height=440)
    return fig


def fig_feature_importance(model, feature_names: list, title: str) -> Optional[go.Figure]:
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index,
        orientation="h",
        marker_color=px.colors.sample_colorscale(
            "Teal", [i / max(len(imp) - 1, 1) for i in range(len(imp))]
        ),
        hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title=title, height=320,
                      xaxis_title="Importance")
    return fig


def fig_return_heatmap(strategies: Dict[str, pd.Series]) -> go.Figure:
    """Monthly return heatmap (year × month) for the best-Sharpe strategy."""
    if not strategies:
        return go.Figure()
    best = max(strategies, key=lambda k: sharpe_ratio(strategies[k]))
    r = strategies[best].dropna()
    df = r.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret") * 100
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale="RdYlGn", zmid=0,
        hoverongaps=False,
        hovertemplate="%{y} %{x}<br>%{z:.2f}%<extra></extra>",
        colorbar=dict(title="Ret (%)"),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Monthly returns heatmap — {best} L/S portfolio",
        height=max(320, 25 * pivot.shape[0] + 120),
    )
    return fig


def fig_rolling_sharpe(strategies: Dict[str, pd.Series], window: int = 24) -> go.Figure:
    fig = go.Figure()
    for name, r in strategies.items():
        r = r.dropna()
        roll_sharpe = r.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan
        )
        fig.add_trace(go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            mode="lines", name=name,
            line=dict(color=MODEL_COLORS.get(name), width=1.8),
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=0, line_width=0.8, line_color="black")
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Rolling {window}-month Sharpe ratio",
        xaxis_title="Date", yaxis_title="Sharpe (annualised)",
        height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 ML Asset Pricing Dashboard")
st.caption(
    "Implements Gu, Kelly & Xiu (2020) — *Empirical Asset Pricing via Machine "
    "Learning* — and Kelly, Malamud & Zhou (2024) — *The Virtue of Complexity "
    "in Return Prediction*."
)

# Gate behind the Run button on first load; auto-run on setting change.
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False

if run_btn:
    st.session_state.pipeline_run = True
    # Clear relevant caches when user explicitly clicks Run.
    cached_run_models.clear()
    cached_complexity.clear()

if not st.session_state.pipeline_run:
    st.info("👈 Configure the pipeline in the sidebar, then click **▶ Run pipeline**.")
    st.stop()

# ── Step 1: Panel ──────────────────────────────────────────────────────────
with st.status("Building firm-month panel…", expanded=False) as status:
    t0 = time.time()
    try:
        panel, feature_cols = cached_build_panel(start_year, end_year, n_firms)
        panel_hash = f"{start_year}_{end_year}_{n_firms}"
        _panel_store["panel"]       = panel
        _panel_store["feature_cols"] = feature_cols
        elapsed = time.time() - t0
        status.update(
            label=f"✅ Panel ready — {len(panel):,} firm-months, "
                  f"{panel['permno'].nunique()} firms, "
                  f"{panel['date'].nunique()} months "
                  f"({panel['date'].min().date()} → {panel['date'].max().date()})  "
                  f"[{elapsed:.1f}s]",
            state="complete",
        )
    except Exception:
        status.update(label="❌ Panel build failed", state="error")
        st.exception(traceback.format_exc())
        st.stop()

# ── Step 2: Models ─────────────────────────────────────────────────────────
with st.status("Training ML models (expanding window)…", expanded=False) as status:
    t0 = time.time()
    try:
        predictions = cached_run_models(
            panel_hash, train_end_date, refit_freq, tuple(selected_models)
        )
        elapsed = time.time() - t0
        n_preds = sum(len(p) for p in predictions.values() if not p.empty)
        status.update(
            label=f"✅ Models trained — {n_preds:,} total predictions across "
                  f"{len(predictions)} models  [{elapsed:.1f}s]",
            state="complete",
        )
    except Exception:
        status.update(label="❌ Model training failed", state="error")
        st.exception(traceback.format_exc())
        st.stop()

# ── Step 3: Portfolios ─────────────────────────────────────────────────────
with st.status("Building portfolios and evaluating…", expanded=False) as status:
    try:
        mkt = market_return_series(panel)
        deciles, strategies, summary_rows = {}, {}, []
        feature_importances = {}

        for name, preds in predictions.items():
            if preds.empty:
                continue
            dec = build_decile_portfolios(preds)
            ls  = long_short_returns(dec)
            deciles[name]    = dec
            strategies[name] = ls
            summary_rows.append(
                summarize_strategy(name, ls, mkt, predictions=preds)
            )

        # Feature importances — refit trees on train window only.
        reg = get_model_registry()
        train_panel = panel[panel["date"] <= train_end_date]
        X_tr = train_panel[feature_cols].values
        y_tr = train_panel["fwd_ret"].values
        for tree in [n for n in selected_models if n in ("RandomForest", "GBM")]:
            try:
                m = reg[tree]()
                m.fit(X_tr, y_tr)
                if hasattr(m, "feature_importances_"):
                    feature_importances[tree] = dict(
                        zip(feature_cols, m.feature_importances_)
                    )
            except Exception:
                pass

        status.update(label="✅ Portfolios and evaluation complete", state="complete")
    except Exception:
        status.update(label="❌ Portfolio evaluation failed", state="error")
        st.exception(traceback.format_exc())
        st.stop()

# ── Step 4: Complexity ─────────────────────────────────────────────────────
sweeps = {"raw": pd.DataFrame(), "rff": pd.DataFrame()}
if run_complexity:
    with st.status("Running complexity sweep…", expanded=False) as status:
        t0 = time.time()
        try:
            sweeps = cached_complexity(panel_hash, train_end_date, refit_freq)
            elapsed = time.time() - t0
            status.update(
                label=f"✅ Complexity sweep done  [{elapsed:.1f}s]",
                state="complete",
            )
        except Exception:
            status.update(label="⚠️ Complexity sweep failed (results still shown)", state="error")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_portfolios, tab_complexity, tab_data = st.tabs([
    "📊 Overview", "📂 Portfolios", "🔬 Complexity", "🗃️ Data"
])

# ── Overview ──────────────────────────────────────────────────────────────
with tab_overview:
    if not summary_rows:
        st.warning("No model results to display.")
    else:
        # KPI metric strip.
        best_sharpe_row = max(summary_rows, key=lambda r: r["sharpe"] or -99)
        best_alpha_row  = max(summary_rows, key=lambda r: r["alpha_ann"] or -99)
        best_r2_row     = max(summary_rows, key=lambda r: r["oos_r2"] or -99)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Sharpe",
                  f"{best_sharpe_row['sharpe']:.2f}",
                  best_sharpe_row["model"])
        c2.metric("Best Ann. Return",
                  f"{max(r['ann_return'] for r in summary_rows)*100:.1f}%")
        c3.metric("Best Alpha",
                  f"{best_alpha_row['alpha_ann']*100:.2f}%",
                  best_alpha_row["model"])
        c4.metric("Best OOS R²",
                  f"{best_r2_row['oos_r2']*100:.3f}%",
                  best_r2_row["model"])

        st.divider()

        # Summary table.
        st.subheader("Performance summary")
        df_sum = pd.DataFrame(summary_rows)
        display_cols = {
            "model":      "Model",
            "ann_return": "Ann. Return",
            "sharpe":     "Sharpe",
            "alpha_ann":  "Alpha (ann.)",
            "beta":       "Beta",
            "alpha_t":    "Alpha t-stat",
            "max_dd":     "Max Drawdown",
            "oos_r2":     "OOS R²",
            "n_months":   "Months",
        }
        df_display = df_sum[list(display_cols)].rename(columns=display_cols)
        pct_cols = ["Ann. Return", "Alpha (ann.)", "Max Drawdown", "OOS R²"]
        for col in pct_cols:
            df_display[col] = df_display[col].apply(lambda v: f"{v*100:.2f}%")
        df_display["Sharpe"]      = df_display["Sharpe"].apply(lambda v: f"{v:.2f}")
        df_display["Beta"]        = df_display["Beta"].apply(lambda v: f"{v:.2f}")
        df_display["Alpha t-stat"]= df_display["Alpha t-stat"].apply(lambda v: f"{v:.2f}")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.divider()
        col_l, col_r = st.columns(2)
        with col_l:
            metric_choice = st.selectbox(
                "Metric to compare",
                ["sharpe", "ann_return", "alpha_ann", "max_dd", "oos_r2"],
                format_func=lambda x: {
                    "sharpe": "Sharpe ratio",
                    "ann_return": "Ann. return",
                    "alpha_ann": "CAPM alpha",
                    "max_dd": "Max drawdown",
                    "oos_r2": "OOS R²",
                }[x],
            )
            st.plotly_chart(
                fig_summary_bar(summary_rows, metric_choice,
                                metric_choice.replace("_", " ").title() + " (×100 = %)"),
                use_container_width=True,
            )
        with col_r:
            st.plotly_chart(fig_cumulative(strategies), use_container_width=True)

        st.plotly_chart(fig_rolling_sharpe(strategies), use_container_width=True)
        st.plotly_chart(fig_drawdown(strategies), use_container_width=True)
        st.plotly_chart(fig_return_heatmap(strategies), use_container_width=True)


# ── Portfolios ─────────────────────────────────────────────────────────────
with tab_portfolios:
    if not deciles:
        st.warning("No portfolio data available.")
    else:
        st.subheader("Decile portfolio returns")
        st.caption(
            "Stocks are ranked by predicted return each month into 10 buckets. "
            "If a model has genuine predictive power the bars should be "
            "monotone increasing from D1 (low predicted) to D10 (high predicted)."
        )
        st.plotly_chart(fig_decile_bars(deciles), use_container_width=True)

        st.subheader("Decile breakdown — detailed table")
        model_choice = st.selectbox("Select model", list(deciles.keys()), key="port_model")
        if model_choice:
            df_dec = decile_summary(deciles[model_choice])
            df_dec["ann_return"] = df_dec["ann_return"].apply(lambda v: f"{v*100:.2f}%")
            df_dec["ann_vol"]    = df_dec["ann_vol"].apply(lambda v: f"{v*100:.2f}%")
            df_dec["sharpe"]     = df_dec["sharpe"].apply(lambda v: f"{v:.2f}")
            df_dec["mean_monthly"] = df_dec["mean_monthly"].apply(lambda v: f"{v*100:.3f}%")
            st.dataframe(df_dec, use_container_width=True, hide_index=True)

        st.subheader("Feature importance (tree models)")
        if feature_importances:
            imp_model = st.selectbox("Tree model", list(feature_importances.keys()),
                                     key="imp_model")
            imp_series = pd.Series(feature_importances[imp_model]).sort_values()
            imp_fig = go.Figure(go.Bar(
                x=imp_series.values, y=imp_series.index,
                orientation="h",
                marker_color=px.colors.sample_colorscale(
                    "Teal", [i / max(len(imp_series) - 1, 1)
                             for i in range(len(imp_series))]),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            imp_fig.update_layout(
                **LAYOUT_BASE,
                title=f"{imp_model} — feature importance (trained on data ≤ {train_end})",
                xaxis_title="Importance", height=340,
            )
            st.plotly_chart(imp_fig, use_container_width=True)
        else:
            st.info("Run RandomForest or GBM to see feature importances.")


# ── Complexity ─────────────────────────────────────────────────────────────
with tab_complexity:
    st.subheader("The Virtue of Complexity")
    st.markdown(
        """
Kelly, Malamud & Zhou (2024) show that ridge regression's out-of-sample R²
**keeps improving** as the number of parameters p grows past the sample
size n — contradicting the classical bias-variance intuition. We replicate
this with two sweeps:

* **Sweep A — raw predictors**: Ridge fitted on the first k of our accounting
  variables (k = 1 … 9). Expect both R² and Sharpe to rise monotonically.
* **Sweep B — random Fourier features (RFF)**: the 9 base features are lifted
  into a p-dimensional random cosine basis (Rahimi & Recht 2007) that
  approximates a Gaussian kernel. Sweeping p from 2 → 500 spans the n ≪ p
  regime where the KMZ "double descent" shape is visible.
        """
    )

    if sweeps["raw"].empty and sweeps["rff"].empty:
        if not run_complexity:
            st.info("Enable the **Run complexity sweep** checkbox in the sidebar.")
        else:
            st.warning("Complexity sweep produced no results.")
    else:
        st.plotly_chart(fig_complexity(sweeps), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if not sweeps["raw"].empty:
                st.markdown("**Sweep A — raw predictors**")
                df_a = sweeps["raw"].copy()
                df_a["oos_r2"]  = df_a["oos_r2"].apply(lambda v: f"{v*100:.4f}%")
                df_a["ls_sharpe"] = df_a["ls_sharpe"].round(3)
                df_a["ls_ann_return"] = df_a["ls_ann_return"].apply(
                    lambda v: f"{v*100:.2f}%")
                df_a.columns = ["# Params", "OOS R²", "L/S Sharpe", "L/S Ann. Ret"]
                st.dataframe(df_a, use_container_width=True, hide_index=True)
        with col_b:
            if not sweeps["rff"].empty:
                st.markdown("**Sweep B — RFF expansion**")
                df_b = sweeps["rff"].copy()
                df_b["oos_r2"]  = df_b["oos_r2"].apply(lambda v: f"{v*100:.4f}%")
                df_b["ls_sharpe"] = df_b["ls_sharpe"].round(3)
                df_b["ls_ann_return"] = df_b["ls_ann_return"].apply(
                    lambda v: f"{v*100:.2f}%")
                df_b.columns = ["# Params", "OOS R²", "L/S Sharpe", "L/S Ann. Ret"]
                st.dataframe(df_b, use_container_width=True, hide_index=True)


# ── Data ───────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Panel statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total firm-months", f"{len(panel):,}")
    c2.metric("Unique firms",      f"{panel['permno'].nunique():,}")
    c3.metric("Months",            f"{panel['date'].nunique():,}")

    st.markdown("**Feature distributions** (post-rank-normalization; should all be ≈ Uniform[-0.5, 0.5])")
    feat_stats = panel[feature_cols].describe().T[["mean","std","min","25%","50%","75%","max"]].round(4)
    st.dataframe(feat_stats, use_container_width=True)

    st.markdown("**Forward return distribution**")
    ret_desc = panel["fwd_ret"].describe().to_frame().T.round(4)
    st.dataframe(ret_desc, use_container_width=True, hide_index=True)

    # Return distribution histogram.
    fig_hist = px.histogram(
        panel["fwd_ret"].clip(-0.5, 0.5),
        nbins=80,
        labels={"value": "Forward return", "count": "Frequency"},
        title="Forward return distribution (clipped at ±50% for display)",
        template="plotly_white",
        color_discrete_sequence=["#4C72B0"],
    )
    fig_hist.update_layout(**LAYOUT_BASE, height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("**Firms per month** (coverage over time)")
    coverage = panel.groupby("date").size().reset_index(name="n_firms")
    fig_cov = px.area(
        coverage, x="date", y="n_firms",
        title="Number of firms per month",
        labels={"date": "Date", "n_firms": "# Firms"},
        template="plotly_white",
        color_discrete_sequence=["#4C72B0"],
    )
    fig_cov.update_layout(**LAYOUT_BASE, height=300)
    st.plotly_chart(fig_cov, use_container_width=True)

    with st.expander("📄 Raw panel sample (first 200 rows)"):
        st.dataframe(
            panel.head(200).style.format({c: "{:.4f}" for c in feature_cols + ["ret", "fwd_ret"]}),
            use_container_width=True,
        )

    with st.expander("📄 Download full panel as CSV"):
        csv = panel.to_csv(index=False)
        st.download_button(
            "⬇️ Download panel.csv",
            data=csv,
            file_name="panel.csv",
            mime="text/csv",
            use_container_width=True,
        )
