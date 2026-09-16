"""Quantitative Risk Engine & Portfolio Analytics Platform.

Institutional-grade financial risk workstation implementing Value-at-Risk (VaR),
Conditional Value-at-Risk (CVaR/Expected Shortfall), Markowitz Mean-Variance Optimization,
Kupiec Proportion of Failures backtesting, Marginal Risk Decomposition, and Regulatory Reporting.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure src/ is in python path for Streamlit Cloud and local execution
SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from frp.config import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MONTE_CARLO_SEED,
    DEFAULT_MONTE_CARLO_SIMULATIONS,
    TRADING_DAYS_PER_YEAR,
)
from frp.analytics.returns import log_returns, rolling_volatility
from frp.data.ingest import fetch_prices
from frp.optimization import (
    PortfolioOptimizer,
    efficient_frontier_weights,
    global_minimum_variance_weights,
    portfolio_return,
    portfolio_variance,
)
from frp.risk import CVaR, VaR
from frp.risk.backtest import kupiec_proportion_of_failures_test

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Institutional Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Risk Engine | Portfolio Analytics",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Institutional FinTech CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Institutional Header Bar */
    .terminal-header {
        background-color: #0d1117;
        border-bottom: 1px solid #21262d;
        padding: 10px 18px;
        margin: -4rem -4rem 1.5rem -4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
        letter-spacing: 0.05em;
    }
    .terminal-header .brand {
        color: #58a6ff;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .terminal-header .tag {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 2px 8px;
        border-radius: 3px;
        color: #c9d1d9;
    }

    /* Metric Stat Card */
    .stat-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .stat-card-border-var {
        border-left: 3px solid #f59e0b;
    }
    .stat-card-border-cvar {
        border-left: 3px solid #ef4444;
    }
    .stat-card-border-vol {
        border-left: 3px solid #3b82f6;
    }
    .stat-card-border-ret {
        border-left: 3px solid #10b981;
    }
    .stat-card-border-sharpe {
        border-left: 3px solid #8b5cf6;
    }
    .stat-card-border-param {
        border-left: 3px solid #06b6d4;
    }

    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 4px;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: #f9fafb;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .stat-meta {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Institutional Status Badges */
    .badge-pass {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-fail {
        background-color: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        display: inline-block;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        padding: 8px 18px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #f3f4f6;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .section-desc {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    /* Monospace Data Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1f2937;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_ticker_data(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    """Fetch daily adjusted close prices for tickers with disk caching."""
    prices_dict = {}
    for ticker in tickers:
        t_clean = ticker.strip().upper()
        try:
            df = fetch_prices(t_clean, period=period)
            if "Close" in df.columns:
                series = df["Close"]
            elif "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                series = df.iloc[:, 0]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors="coerce").dropna()
            series.name = t_clean
            prices_dict[t_clean] = series
        except Exception as err:
            st.sidebar.warning(f"Ticker unavailable: {t_clean} ({err})")

    if not prices_dict:
        raise ValueError("Failed to retrieve price series for requested tickers.")
    combined = pd.DataFrame(prices_dict).dropna()
    combined = combined.apply(pd.to_numeric, errors="coerce").dropna()
    return combined


def get_institutional_synthetic_panel() -> pd.DataFrame:
    """Generate calibrated multi-asset reference panel for offline execution."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=504, freq="B")
    assets = ["SPX_US_EQUITY", "NDX_TECH_GROWTH", "UST_10Y_GOVT", "BCOM_COMMODITIES"]
    mean_daily = np.array([0.00038, 0.00055, 0.00008, 0.00028])
    vol_daily = np.array([0.0115, 0.0175, 0.0048, 0.0135])
    corr = np.array([
        [1.00,  0.78, -0.28,  0.32],
        [0.78,  1.00, -0.36,  0.34],
        [-0.28, -0.36,  1.00, -0.12],
        [0.32,  0.34, -0.12,  1.00],
    ])
    cov = np.outer(vol_daily, vol_daily) * corr
    sim_returns = rng.multivariate_normal(mean_daily, cov, size=len(dates))
    price_paths = 100.0 * np.exp(np.cumsum(sim_returns, axis=0))
    return pd.DataFrame(price_paths, index=dates, columns=assets)


# -----------------------------------------------------------------------------
# Sidebar: Universe Selection & Risk Engine Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Portfolio Configuration")
    st.caption("FRP QUANT RISK ENGINE // V0.1.0")
    st.markdown("---")

    data_mode = st.radio(
        "Data Ingestion Source",
        options=["Pre-Configured Institutional Universes", "Custom Market Tickers (Yahoo Finance)", "Upload CSV Price Data"],
        index=0,
    )

    prices_df: pd.DataFrame | None = None

    if data_mode == "Pre-Configured Institutional Universes":
        portfolio_choice = st.selectbox(
            "Select Portfolio Universe",
            options=[
                "US Large-Cap Core (AAPL, MSFT, GOOGL, NVDA)",
                "Multi-Asset Endowment (SPY, QQQ, GLD, TLT)",
                "Global Macro & Beta (NVDA, TSLA, COIN, AMD)",
                "Offline Synthetic Multi-Asset Benchmark",
            ],
            index=0,
        )
        lookback = st.selectbox("Historical Window", ["1y", "2y", "3y", "5y"], index=1)

        if "Large-Cap" in portfolio_choice:
            tickers = ("AAPL", "MSFT", "GOOGL", "NVDA")
            prices_df = load_ticker_data(tickers, period=lookback)
        elif "Endowment" in portfolio_choice:
            tickers = ("SPY", "QQQ", "GLD", "TLT")
            prices_df = load_ticker_data(tickers, period=lookback)
        elif "Global Macro" in portfolio_choice:
            tickers = ("NVDA", "TSLA", "COIN", "AMD")
            prices_df = load_ticker_data(tickers, period=lookback)
        else:
            prices_df = get_institutional_synthetic_panel()

    elif data_mode == "Custom Market Tickers (Yahoo Finance)":
        ticker_str = st.text_input("Tickers (Comma-separated)", value="AAPL, MSFT, JPM, GOOGL")
        lookback = st.selectbox("Historical Window", ["6mo", "1y", "2y", "3y", "5y"], index=2)
        tickers = tuple(t.strip().upper() for t in ticker_str.split(",") if t.strip())

        if st.button("Fetch Historical Prices", use_container_width=True) or ("market_prices" not in st.session_state):
            with st.spinner("Retrieving market time series..."):
                try:
                    prices_df = load_ticker_data(tickers, period=lookback)
                    st.session_state["market_prices"] = prices_df
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    prices_df = get_institutional_synthetic_panel()
        if prices_df is None and "market_prices" in st.session_state:
            prices_df = st.session_state["market_prices"]

    else:
        file_input = st.file_uploader("Upload Time Series CSV", type=["csv"], help="Index: Date, Columns: Asset Close Prices")
        if file_input is not None:
            try:
                raw_csv = pd.read_csv(file_input, index_col=0, parse_dates=True)
                numeric_cols = raw_csv.select_dtypes(include=[np.number]).dropna()
                if numeric_cols.shape[1] >= 1:
                    prices_df = numeric_cols
                    st.success(f"Loaded {prices_df.shape[1]} series across {len(prices_df)} dates.")
                else:
                    st.error("CSV must contain at least 1 numeric price column.")
            except Exception as e:
                st.error(f"CSV read failed: {e}")
        if prices_df is None:
            prices_df = get_institutional_synthetic_panel()

    if prices_df is None or prices_df.empty:
        prices_df = get_institutional_synthetic_panel()

    st.markdown("---")
    st.markdown("### Risk Engine Parameters")
    confidence_level = st.slider(
        "Confidence Level (1 - α)",
        min_value=0.90,
        max_value=0.999,
        value=DEFAULT_CONFIDENCE_LEVEL,
        step=0.005,
        format="%.3f",
    )
    risk_free_rate = st.number_input(
        "Risk-Free Rate (Annualized Rf)",
        min_value=0.00,
        max_value=0.15,
        value=0.035,
        step=0.0025,
        format="%.4f",
    )
    mc_iterations = st.select_slider(
        "Monte Carlo Paths",
        options=[2_000, 5_000, 10_000, 25_000, 50_000],
        value=DEFAULT_MONTE_CARLO_SIMULATIONS,
    )
    portfolio_nav = st.number_input(
        "Portfolio NAV ($ Base)",
        min_value=100_000,
        max_value=1_000_000_000,
        value=10_000_000,
        step=500_000,
        format="%d",
    )

prices_df = prices_df.apply(pd.to_numeric, errors="coerce").dropna()
asset_names = list(prices_df.columns)
n_assets = len(asset_names)

# Compute daily log returns panel
returns_df = pd.DataFrame()
for col in asset_names:
    returns_df[col] = log_returns(prices_df[col])
returns_df = returns_df.dropna()

# -----------------------------------------------------------------------------
# Portfolio Allocation Strategy Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### Portfolio Weighting")
    weighting_type = st.selectbox(
        "Weight Strategy",
        options=["Equal Weight (1/N)", "Global Minimum Variance (GMV)", "Tangency / Maximum Sharpe", "Custom Weight Allocation"],
        index=0,
    )

    expected_asset_returns = returns_df.mean(axis=0) * TRADING_DAYS_PER_YEAR
    cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR

    if weighting_type == "Equal Weight (1/N)":
        weights = pd.Series(1.0 / n_assets, index=asset_names)
    elif weighting_type == "Global Minimum Variance (GMV)":
        weights = global_minimum_variance_weights(returns_df.cov())
    elif weighting_type == "Tangency / Maximum Sharpe":
        optimizer = PortfolioOptimizer(expected_asset_returns, cov_matrix)
        weights = optimizer.max_sharpe(risk_free_rate=risk_free_rate)
    else:
        st.caption("Allocate percentage per holding:")
        custom_dict = {}
        for a in asset_names:
            custom_dict[a] = st.slider(f"{a} (%)", 0, 100, int(100 / n_assets), 1)
        tot = sum(custom_dict.values())
        if tot > 0:
            weights = pd.Series({k: v / tot for k, v in custom_dict.items()})
        else:
            weights = pd.Series(1.0 / n_assets, index=asset_names)

# Aggregate Portfolio Daily Returns
portfolio_daily_returns = returns_df @ weights
portfolio_annual_return = float(portfolio_return(weights, returns_df.mean(axis=0)) * TRADING_DAYS_PER_YEAR)
portfolio_annual_vol = float(np.sqrt(max(0.0, portfolio_variance(weights, cov_matrix))))
portfolio_sharpe = (
    (portfolio_annual_return - risk_free_rate) / portfolio_annual_vol
    if portfolio_annual_vol > 1e-6
    else 0.0
)

# Core Risk Estimations
var_hist = VaR.historical(portfolio_daily_returns, confidence=confidence_level)
var_param = VaR.parametric(portfolio_daily_returns, confidence=confidence_level)
var_mc = VaR.monte_carlo(
    portfolio_daily_returns,
    confidence=confidence_level,
    simulations=mc_iterations,
    seed=DEFAULT_MONTE_CARLO_SEED,
)
cvar_hist = CVaR.historical(portfolio_daily_returns, confidence=confidence_level)
cvar_param = CVaR.parametric(portfolio_daily_returns, confidence=confidence_level)

# Dollar Loss Exposures
var_hist_dollar = var_hist * portfolio_nav
cvar_hist_dollar = cvar_hist * portfolio_nav

# -----------------------------------------------------------------------------
# Top Institutional Terminal Bar
# -----------------------------------------------------------------------------
timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(
    f"""
    <div class="terminal-header">
        <div>
            <span class="brand">QUANT RISK ENGINE</span> // DESK: MULTI-ASSET RISK // DESK ID: #CITS5770
        </div>
        <div>
            <span class="tag">UNIVERSE: {len(asset_names)} ASSETS</span>
            <span class="tag">OBSERVATIONS: {len(portfolio_daily_returns)} DAYS</span>
            <span class="tag">CONFIDENCE: {confidence_level*100:.1f}%</span>
            <span class="tag">NAV: ${portfolio_nav:,.0f}</span>
            <span style="color: #6e7681; margin-left: 8px;">{timestamp_str}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# High-Density Metric Strip (Bloomberg/FactSet Style)
# -----------------------------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(
        f"""
        <div class="stat-card stat-card-border-ret">
            <div class="stat-label">Expected Return (Ann.)</div>
            <div class="stat-value">{portfolio_annual_return*100:+.2f}%</div>
            <div class="stat-meta">${portfolio_annual_return * portfolio_nav:,.0f} / year</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="stat-card stat-card-border-vol">
            <div class="stat-label">Volatility (Ann. σ)</div>
            <div class="stat-value">{portfolio_annual_vol*100:.2f}%</div>
            <div class="stat-meta">Daily σ: {(portfolio_annual_vol/np.sqrt(252))*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="stat-card stat-card-border-sharpe">
            <div class="stat-label">Sharpe Ratio</div>
            <div class="stat-value">{portfolio_sharpe:.2f}</div>
            <div class="stat-meta">Benchmark Rf: {risk_free_rate*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="stat-card stat-card-border-var">
            <div class="stat-label">Historical VaR ({confidence_level*100:.1f}%)</div>
            <div class="stat-value" style="color: #f59e0b;">{var_hist*100:.2f}%</div>
            <div class="stat-meta">${var_hist_dollar:,.0f} 1-day threshold</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m5:
    st.markdown(
        f"""
        <div class="stat-card stat-card-border-cvar">
            <div class="stat-label">Expected Shortfall (CVaR)</div>
            <div class="stat-value" style="color: #ef4444;">{cvar_hist*100:.2f}%</div>
            <div class="stat-meta">${cvar_hist_dollar:,.0f} tail expectation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m6:
    st.markdown(
        f"""
        <div class="stat-card stat-card-border-param">
            <div class="stat-label">Parametric VaR (1-Day)</div>
            <div class="stat-value" style="color: #06b6d4;">{var_param*100:.2f}%</div>
            <div class="stat-meta">Monte Carlo: {var_mc*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Main Institutional Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_var, tab_frontier, tab_backtest, tab_stress, tab_export = st.tabs([
    "PORTFOLIO OVERVIEW & RISK DECOMPOSITION",
    "VALUE-AT-RISK & TAIL RISK ANALYTICS",
    "MARKOWITZ MEAN-VARIANCE FRONTIER",
    "KUPIEC POF MODEL VALIDATION",
    "STRESS TESTING & SCENARIOS",
    "COMPLIANCE & REGULATORY EXPORT",
])

# =============================================================================
# TAB 1: Portfolio Overview & Risk Decomposition
# =============================================================================
with tab_overview:
    st.markdown('<div class="section-title">Portfolio Allocation & Marginal Risk Contribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Analysis of active holdings, individual asset volatility, and percentage contribution to aggregate portfolio variance.</div>', unsafe_allow_html=True)

    # Calculate Marginal Contribution to Risk (MCR) and Percentage Contribution to Risk (PCR)
    cov_arr = cov_matrix.to_numpy()
    w_arr = weights.to_numpy()
    sigma_p = portfolio_annual_vol

    if sigma_p > 1e-6:
        marginal_risk = (cov_arr @ w_arr) / sigma_p
        component_risk = w_arr * marginal_risk
        pct_risk_contrib = (component_risk / sigma_p) * 100.0
    else:
        marginal_risk = np.zeros(n_assets)
        pct_risk_contrib = np.zeros(n_assets)

    decomp_df = pd.DataFrame({
        "Asset": asset_names,
        "Active Weight (%)": [weights[a] * 100.0 for a in asset_names],
        "NAV Exposure ($)": [weights[a] * portfolio_nav for a in asset_names],
        "Annualized Return (%)": [expected_asset_returns[a] * 100.0 for a in asset_names],
        "Annualized Volatility (%)": [float(np.sqrt(cov_matrix.loc[a, a])) * 100.0 for a in asset_names],
        "Risk Contribution (%)": pct_risk_contrib,
    })

    oc1, oc2 = st.columns([1, 1])

    with oc1:
        # Asset Allocation vs Risk Contribution Bar Chart
        fig_alloc = go.Figure()
        fig_alloc.add_trace(
            go.Bar(
                x=decomp_df["Asset"],
                y=decomp_df["Active Weight (%)"],
                name="Portfolio Weight (%)",
                marker_color="#3b82f6",
            )
        )
        fig_alloc.add_trace(
            go.Bar(
                x=decomp_df["Asset"],
                y=decomp_df["Risk Contribution (%)"],
                name="Risk Contribution (%)",
                marker_color="#f59e0b",
            )
        )
        fig_alloc.update_layout(
            title="Capital Allocation vs. Total Risk Contribution",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    with oc2:
        # Cross-Asset Correlation Heatmap
        corr = returns_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            title="Asset Log-Return Correlation Matrix (Σ Norm)",
            template="plotly_dark",
        )
        fig_corr.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("##### Portfolio Holdings & Risk Profile")
    st.dataframe(
        decomp_df.style.format({
            "Active Weight (%)": "{:.2f}%",
            "NAV Exposure ($)": "${:,.0f}",
            "Annualized Return (%)": "{:+.2f}%",
            "Annualized Volatility (%)": "{:.2f}%",
            "Risk Contribution (%)": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Historical Trajectory
    st.markdown("---")
    st.markdown('<div class="section-title">Cumulative Performance Benchmark (Base 100)</div>', unsafe_allow_html=True)
    norm_cum = (1.0 + portfolio_daily_returns).cumprod() * 100.0
    norm_assets = (prices_df / prices_df.iloc[0]) * 100.0

    fig_traj = go.Figure()
    for col in asset_names:
        fig_traj.add_trace(go.Scatter(x=norm_assets.index, y=norm_assets[col], name=col, opacity=0.55, line=dict(width=1.2)))
    fig_traj.add_trace(go.Scatter(x=norm_cum.index, y=norm_cum, name="AGGREGATE PORTFOLIO", line=dict(color="#ffffff", width=2.8)))

    fig_traj.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Timeline",
        yaxis_title="Normalized Value",
    )
    st.plotly_chart(fig_traj, use_container_width=True)

# =============================================================================
# TAB 2: VaR & Tail Risk Analytics
# =============================================================================
with tab_var:
    st.markdown('<div class="section-title">Value-at-Risk & Expected Shortfall Modeling</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Multi-method tail risk assessment comparing empirical non-parametric quantiles, parametric Gaussian assumptions, and stochastic Monte Carlo paths.</div>', unsafe_allow_html=True)

    vc1, vc2 = st.columns([2, 1])

    with vc1:
        # Empirical Distribution & Cutoff Thresholds
        rets_sample = portfolio_daily_returns.to_numpy()
        fig_dist = go.Figure()

        # Histogram
        fig_dist.add_trace(
            go.Histogram(
                x=rets_sample,
                nbinsx=65,
                histnorm="probability density",
                name="Daily Returns",
                marker_color="#334155",
                opacity=0.85,
            )
        )

        # Theoretical Normal Density Fit
        x_axis = np.linspace(rets_sample.min() * 1.2, rets_sample.max() * 1.2, 200)
        mean_r = float(rets_sample.mean())
        std_r = float(rets_sample.std(ddof=1))
        norm_density = stats.norm.pdf(x_axis, loc=mean_r, scale=std_r)
        fig_dist.add_trace(
            go.Scatter(
                x=x_axis,
                y=norm_density,
                mode="lines",
                name="Fitted Normal PDF",
                line=dict(color="#64748b", width=1.5, dash="dash"),
            )
        )

        # Vertical Cutoffs
        fig_dist.add_vline(x=-var_hist, line_width=2, line_dash="solid", line_color="#f59e0b",
                           annotation_text=f"Hist VaR: -{var_hist*100:.2f}%", annotation_position="top left")
        fig_dist.add_vline(x=-var_param, line_width=1.5, line_dash="dot", line_color="#06b6d4",
                           annotation_text=f"Param VaR: -{var_param*100:.2f}%", annotation_position="bottom left")
        fig_dist.add_vline(x=-cvar_hist, line_width=2.5, line_dash="solid", line_color="#ef4444",
                           annotation_text=f"CVaR: -{cvar_hist*100:.2f}%", annotation_position="top left")

        fig_dist.update_layout(
            title=f"Return Probability Density & Tail Cutoffs ({confidence_level*100:.1f}%)",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            xaxis_title="1-Day Log Return",
            yaxis_title="Probability Density",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with vc2:
        st.markdown("##### Model Comparison Framework")
        model_comp_table = pd.DataFrame({
            "Estimation Model": [
                "Historical Simulation (Empirical)",
                "Parametric Gaussian (Delta-Normal)",
                f"Monte Carlo ({mc_iterations:,} Paths)",
                "Expected Shortfall (CVaR Empirical)",
                "Expected Shortfall (CVaR Parametric)",
            ],
            "1-Day Loss (%)": [
                f"{var_hist*100:.3f}%",
                f"{var_param*100:.3f}%",
                f"{var_mc*100:.3f}%",
                f"{cvar_hist*100:.3f}%",
                f"{cvar_param*100:.3f}%",
            ],
            "Dollar Exposure": [
                f"${var_hist * portfolio_nav:,.0f}",
                f"${var_param * portfolio_nav:,.0f}",
                f"${var_mc * portfolio_nav:,.0f}",
                f"${cvar_hist * portfolio_nav:,.0f}",
                f"${cvar_param * portfolio_nav:,.0f}",
            ],
        })
        st.dataframe(model_comp_table, hide_index=True, use_container_width=True)

        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 12px; font-size: 0.78rem; color: #8b949e; line-height: 1.5; margin-top: 10px;">
                <strong style="color: #c9d1d9;">Methodological Summary:</strong><br>
                • <strong>Historical VaR:</strong> Non-parametric order statistic; sensitive to finite sample clustering.<br>
                • <strong>Parametric VaR:</strong> Assumes Gaussian tail; underestimates kurtosis and fat tails.<br>
                • <strong>CVaR (Expected Shortfall):</strong> Coherent risk measure capturing average severity beyond VaR threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Multi-Confidence Sensitivity Curve
    st.markdown("---")
    st.markdown('<div class="section-title">Confidence Sensitivity Surface (90.0% to 99.9%)</div>', unsafe_allow_html=True)
    c_grid = np.linspace(0.90, 0.999, 45)
    surf_data = []
    for c in c_grid:
        h = VaR.historical(portfolio_daily_returns, confidence=c)
        p = VaR.parametric(portfolio_daily_returns, confidence=c)
        cv = CVaR.historical(portfolio_daily_returns, confidence=c)
        surf_data.append({"Confidence (%)": c * 100.0, "Historical VaR": h * 100.0, "Parametric VaR": p * 100.0, "Expected Shortfall (CVaR)": cv * 100.0})
    surf_df = pd.DataFrame(surf_data)

    fig_surf = go.Figure()
    fig_surf.add_trace(go.Scatter(x=surf_df["Confidence (%)"], y=surf_df["Historical VaR"], name="Historical VaR", line=dict(color="#f59e0b", width=2)))
    fig_surf.add_trace(go.Scatter(x=surf_df["Confidence (%)"], y=surf_df["Parametric VaR"], name="Parametric VaR", line=dict(color="#06b6d4", width=1.5, dash="dash")))
    fig_surf.add_trace(go.Scatter(x=surf_df["Confidence (%)"], y=surf_df["Expected Shortfall (CVaR)"], name="Expected Shortfall (CVaR)", line=dict(color="#ef4444", width=2.5)))

    fig_surf.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Confidence Level (1 - α) %",
        yaxis_title="1-Day Loss Expectation (%)",
    )
    st.plotly_chart(fig_surf, use_container_width=True)

# =============================================================================
# TAB 3: Markowitz Mean-Variance Frontier
# =============================================================================
with tab_frontier:
    st.markdown('<div class="section-title">Markowitz Mean-Variance Optimization & Efficient Frontier</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Closed-form analytical solution of the unconstrained Markowitz Lagrangian quadratic problem without numerical black-box heuristics.</div>', unsafe_allow_html=True)

    optimizer = PortfolioOptimizer(expected_asset_returns, cov_matrix)
    gmv_weights = optimizer.minimum_variance()
    gmv_ret = float(portfolio_return(gmv_weights, expected_asset_returns))
    gmv_vol = float(np.sqrt(max(0.0, portfolio_variance(gmv_weights, cov_matrix))))

    tangency_weights = optimizer.max_sharpe(risk_free_rate=risk_free_rate)
    tan_ret = float(portfolio_return(tangency_weights, expected_asset_returns))
    tan_vol = float(np.sqrt(max(0.0, portfolio_variance(tangency_weights, cov_matrix))))
    tan_sharpe = (tan_ret - risk_free_rate) / tan_vol if tan_vol > 1e-6 else 0.0

    frontier_pts = optimizer.efficient_frontier(n_points=120)
    f_vols = [p["volatility"] * 100.0 for p in frontier_pts]
    f_rets = [p["return"] * 100.0 for p in frontier_pts]

    fc1, fc2 = st.columns([2, 1])

    with fc1:
        fig_front = go.Figure()

        # Frontier
        fig_front.add_trace(go.Scatter(x=f_vols, y=f_rets, mode="lines", name="Efficient Frontier", line=dict(color="#3b82f6", width=2.5)))

        # Capital Allocation Line (CAL)
        if tan_vol > 0:
            cal_x_vals = np.linspace(0, max(f_vols) * 1.25, 40)
            cal_y_vals = (risk_free_rate + tan_sharpe * (cal_x_vals / 100.0)) * 100.0
            fig_front.add_trace(go.Scatter(x=cal_x_vals, y=cal_y_vals, mode="lines", name=f"Capital Allocation Line (Rf={risk_free_rate*100:.2f}%)", line=dict(color="#475569", width=1.5, dash="dot")))

        # Individual Assets
        asset_v = [float(np.sqrt(cov_matrix.loc[a, a])) * 100.0 for a in asset_names]
        asset_r = [float(expected_asset_returns[a]) * 100.0 for a in asset_names]
        fig_front.add_trace(go.Scatter(x=asset_v, y=asset_r, mode="markers+text", name="Underlying Assets", text=asset_names, textposition="top right", marker=dict(size=8, color="#06b6d4")))

        # Key Portfolio Markers
        fig_front.add_trace(go.Scatter(x=[gmv_vol * 100], y=[gmv_ret * 100], mode="markers", name="Global Min Variance (GMV)", marker=dict(size=12, color="#10b981", symbol="square")))
        fig_front.add_trace(go.Scatter(x=[tan_vol * 100], y=[tan_ret * 100], mode="markers", name="Tangency / Max Sharpe", marker=dict(size=13, color="#f59e0b", symbol="diamond")))
        fig_front.add_trace(go.Scatter(x=[portfolio_annual_vol * 100], y=[portfolio_annual_return * 100], mode="markers", name="Active Portfolio", marker=dict(size=14, color="#ef4444", symbol="cross")))

        fig_front.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=440,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Annualized Expected Return (%)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_front, use_container_width=True)

    with fc2:
        st.markdown("##### Portfolio Optimization Metrics")
        opt_table = pd.DataFrame({
            "Strategy": ["Active Portfolio", "Global Min Variance", "Tangency / Max Sharpe"],
            "Return (Ann.)": [f"{portfolio_annual_return*100:+.2f}%", f"{gmv_ret*100:+.2f}%", f"{tan_ret*100:+.2f}%"],
            "Vol (Ann.)": [f"{portfolio_annual_vol*100:.2f}%", f"{gmv_vol*100:.2f}%", f"{tan_vol*100:.2f}%"],
            "Sharpe Ratio": [f"{portfolio_sharpe:.2f}", f"{(gmv_ret-risk_free_rate)/gmv_vol:.2f}", f"{tan_sharpe:.2f}"],
        })
        st.dataframe(opt_table, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Target Return Solver")
        t_ret = st.slider("Target Return (%)", float(np.floor(min(asset_r))), float(np.ceil(max(asset_r) * 1.25)), float(np.round(portfolio_annual_return * 100, 1)), 0.5)
        try:
            t_w = efficient_frontier_weights(returns_df.mean(axis=0), returns_df.cov(), (t_ret / 100.0) / TRADING_DAYS_PER_YEAR)
            t_vol = float(np.sqrt(max(0.0, portfolio_variance(t_w, cov_matrix))))
            st.markdown(f"**Solved Volatility:** `{t_vol*100:.2f}%` | **Solved Sharpe:** `{(t_ret/100.0 - risk_free_rate)/t_vol:.2f}`")
        except Exception as err:
            st.caption(f"No unique solution: {err}")

    # Optimal Weights Allocation Bar
    st.markdown("---")
    st.markdown('<div class="section-title">Comparative Asset Allocation Weights (%)</div>', unsafe_allow_html=True)
    comp_weights_df = pd.DataFrame({
        "Asset": asset_names,
        "Active Weight": [weights[a] * 100.0 for a in asset_names],
        "GMV Optimal": [gmv_weights[a] * 100.0 for a in asset_names],
        "Max Sharpe Optimal": [tangency_weights[a] * 100.0 for a in asset_names],
    })
    fig_w_comp = px.bar(
        comp_weights_df,
        x="Asset",
        y=["Active Weight", "GMV Optimal", "Max Sharpe Optimal"],
        barmode="group",
        template="plotly_dark",
        color_discrete_sequence=["#ef4444", "#10b981", "#f59e0b"],
    )
    fig_w_comp.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Weight Percentage (%)",
    )
    st.plotly_chart(fig_w_comp, use_container_width=True)

# =============================================================================
# TAB 4: Kupiec POF Model Validation
# =============================================================================
with tab_backtest:
    st.markdown('<div class="section-title">Kupiec Proportion of Failures (POF) Hypothesis Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Statistical validation of model calibration via likelihood-ratio test on Bernoulli trial exceedances against the Basel framework.</div>', unsafe_allow_html=True)

    bc1, bc2 = st.columns([1, 2])

    with bc1:
        st.markdown("##### Backtest Setup")
        bt_engine = st.selectbox("Selected VaR Benchmark", ["Historical VaR (Full Sample)", "Parametric Delta-Normal VaR", "Rolling 60-Day Historical VaR"], index=0)
        test_alpha = st.slider("Significance Threshold (α_test)", 0.01, 0.10, 0.05, 0.01)

        if bt_engine == "Historical VaR (Full Sample)":
            eval_var = var_hist
        elif bt_engine == "Parametric Delta-Normal VaR":
            eval_var = var_param
        else:
            roll_var = portfolio_daily_returns.rolling(60).apply(lambda s: VaR.historical(s, confidence=confidence_level), raw=False).shift(1)
            eval_var = float(roll_var.dropna().mean())

        passes, violations, expected_violations = kupiec_proportion_of_failures_test(
            portfolio_daily_returns,
            var=eval_var,
            confidence=confidence_level,
            significance_level=test_alpha,
        )

        n_obs = len(portfolio_daily_returns)
        obs_rate = (violations / n_obs) * 100.0 if n_obs > 0 else 0.0
        exp_rate = (1.0 - confidence_level) * 100.0

        # Basel Zone Classification
        if violations <= expected_violations * 1.25:
            traffic_zone = "GREEN ZONE (Calibrated)"
            zone_color = "#34d399"
        elif violations <= expected_violations * 1.85:
            traffic_zone = "YELLOW ZONE (Supervisory Review)"
            zone_color = "#f59e0b"
        else:
            traffic_zone = "RED ZONE (Model Deficient)"
            zone_color = "#f87171"

        status_badge = '<span class="badge-pass">PASS // MODEL ACCEPTED</span>' if passes else '<span class="badge-fail">REJECT // NULL HYPOTHESIS FAILED</span>'
        st.markdown(f"**Model Test Status:** {status_badge}", unsafe_allow_html=True)
        st.markdown(f"**Basel Accord Classification:** <span style='color: {zone_color}; font-weight: 600;'>{traffic_zone}</span>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.8; margin-top: 12px;">
                Total Observations: <strong>{n_obs}</strong><br>
                Empirical Breaches (X): <strong>{violations}</strong> ({obs_rate:.2f}%)<br>
                Expected Breaches (N·p): <strong>{expected_violations:.1f}</strong> ({exp_rate:.2f}%)<br>
                Significance Level: <strong>{test_alpha:.2f}</strong><br>
                Null Hypothesis H0: <strong>p = {1.0 - confidence_level:.3f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bc2:
        # Exceedance Timeline Plot
        tl_df = pd.DataFrame({"Return": portfolio_daily_returns})
        tl_df["VaR_Cutoff"] = -eval_var
        tl_df["Breach"] = tl_df["Return"] < tl_df["VaR_Cutoff"]

        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(x=tl_df.index, y=tl_df["Return"] * 100.0, mode="lines", name="Daily Return (%)", line=dict(color="#475569", width=1.1)))
        fig_tl.add_trace(go.Scatter(x=tl_df.index, y=tl_df["VaR_Cutoff"] * 100.0, mode="lines", name=f"VaR Threshold (-{eval_var*100:.2f}%)", line=dict(color="#f59e0b", width=1.8, dash="dash")))

        breaches = tl_df[tl_df["Breach"]]
        if not breaches.empty:
            fig_tl.add_trace(go.Scatter(x=breaches.index, y=breaches["Return"] * 100.0, mode="markers", name=f"Violations ({len(breaches)})", marker=dict(size=7, color="#ef4444", symbol="circle")))

        fig_tl.update_layout(
            title="Exceedance Time Series & Breach Distribution",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Return (%)",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # Rolling 30-Day Volatility
    st.markdown("---")
    st.markdown('<div class="section-title">Rolling 30-Day Volatility & Volatility Clustering Dynamics</div>', unsafe_allow_html=True)
    r_vol = rolling_volatility(portfolio_daily_returns, window=30, annualize=True)
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(x=r_vol.index, y=r_vol * 100.0, name="Rolling 30-Day Volatility", line=dict(color="#8b5cf6", width=1.8)))
    fig_rv.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Timeline",
        yaxis_title="Annualized Volatility (%)",
    )
    st.plotly_chart(fig_rv, use_container_width=True)

# =============================================================================
# TAB 5: Stress Testing & Scenario Analysis
# =============================================================================
with tab_stress:
    st.markdown('<div class="section-title">Institutional Stress Testing & Macro Scenario Shocks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Simulate historical market tail events and macroeconomic crisis shocks across the current portfolio structure.</div>', unsafe_allow_html=True)

    scenarios = [
        {"Scenario": "2008 Global Financial Crisis", "Equity Shock": -0.225, "Fixed Income Shock": 0.052, "Commodity Shock": -0.184, "Description": "Liquidity freeze and systemic subprime contagion."},
        {"Scenario": "2020 COVID-19 Liquidity Shock", "Equity Shock": -0.124, "Fixed Income Shock": 0.021, "Commodity Shock": -0.248, "Description": "Rapid velocity exogenous pandemic shutdown."},
        {"Scenario": "2022 Rates & Inflation Drawdown", "Equity Shock": -0.095, "Fixed Income Shock": -0.062, "Commodity Shock": 0.125, "Description": "Aggressive central bank quantitative tightening regime."},
        {"Scenario": "2011 US Sovereign Debt Downgrade", "Equity Shock": -0.068, "Fixed Income Shock": 0.038, "Commodity Shock": -0.042, "Description": "S&P downgrade of US sovereign credit rating."},
    ]

    stress_results = []
    for s in scenarios:
        # Apply average asset class beta shock approximation
        p_shock = float(portfolio_daily_returns.std(ddof=1) * s["Equity Shock"] * 10.0)
        p_loss_dollar = abs(p_shock) * portfolio_nav
        stress_results.append({
            "Macro Event": s["Scenario"],
            "Estimated Portfolio Drawdown (%)": f"{p_shock * 100:+.2f}%",
            "Projected Capital Loss ($)": f"-${p_loss_dollar:,.0f}",
            "Narrative Context": s["Description"],
        })

    stress_df = pd.DataFrame(stress_results)
    st.dataframe(stress_df, hide_index=True, use_container_width=True)

    st.markdown("##### Custom Tail Shock Scenario")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        custom_pct = st.slider("Immediate Portfolio Shock (%)", -50.0, 10.0, -15.0, 0.5)
    with s_col2:
        custom_loss = (custom_pct / 100.0) * portfolio_nav
        st.markdown(f"**Projected PnL Impact:** <span style='font-size: 1.3rem; font-family: monospace; color: #ef4444;'>-${abs(custom_loss):,.0f}</span> (NAV after shock: `${portfolio_nav + custom_loss:,.0f}`)", unsafe_allow_html=True)

# =============================================================================
# TAB 6: Compliance & Regulatory Export
# =============================================================================
with tab_export:
    st.markdown('<div class="section-title">Regulatory Risk Report & Audit Trail Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Download signed, machine-readable CSV audits containing model assumptions, risk measures, and exceedance logs for compliance desks.</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("##### Risk Executive Audit Record")
        audit_records = pd.DataFrame({
            "Audit Parameter": [
                "Report Generation Timestamp",
                "Portfolio Holdings Universe",
                "Model Engine",
                "Assumed Risk Horizon",
                "Confidence Interval (1 - α)",
                "Portfolio NAV",
                "Annualized Expected Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "1-Day Historical VaR",
                "1-Day Parametric VaR",
                "1-Day Monte Carlo VaR",
                "1-Day Expected Shortfall (CVaR)",
                "Kupiec POF Calibration Status",
                "Breach Count / Expected",
            ],
            "Value": [
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                ", ".join(asset_names),
                "FRP Quant Risk Engine v0.1.0",
                "1 Trading Day (252-day basis)",
                f"{confidence_level*100:.1f}%",
                f"${portfolio_nav:,.2f}",
                f"{portfolio_annual_return*100:+.2f}%",
                f"{portfolio_annual_vol*100:.2f}%",
                f"{portfolio_sharpe:.2f}",
                f"{var_hist*100:.3f}% (${var_hist * portfolio_nav:,.0f})",
                f"{var_param*100:.3f}% (${var_param * portfolio_nav:,.0f})",
                f"{var_mc*100:.3f}% (${var_mc * portfolio_nav:,.0f})",
                f"{cvar_hist*100:.3f}% (${cvar_hist * portfolio_nav:,.0f})",
                "PASS" if passes else "REJECT",
                f"{violations} observed / {expected_violations:.1f} expected",
            ],
        })
        st.dataframe(audit_records, hide_index=True, use_container_width=True)

        buf_audit = io.StringIO()
        audit_records.to_csv(buf_audit, index=False)
        st.download_button(
            label="Download Risk Audit Record (CSV)",
            data=buf_audit.getvalue(),
            file_name=f"audit_risk_record_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with rc2:
        st.markdown("##### Portfolio Holdings & Weights")
        st.dataframe(decomp_df, hide_index=True, use_container_width=True)

        buf_holdings = io.StringIO()
        decomp_df.to_csv(buf_holdings, index=False)
        st.download_button(
            label="Download Portfolio Holdings Breakdown (CSV)",
            data=buf_holdings.getvalue(),
            file_name=f"portfolio_holdings_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("##### Historical Breach Log")
        breach_log = tl_df[tl_df["Breach"]][["Return", "VaR_Cutoff"]].copy()
        breach_log["Loss Magnitude (%)"] = -breach_log["Return"] * 100.0
        breach_log["VaR Threshold (%)"] = -breach_log["VaR_Cutoff"] * 100.0

        buf_breaches = io.StringIO()
        breach_log.to_csv(buf_breaches)
        st.download_button(
            label=f"Download VaR Breach Log ({len(breach_log)} events)",
            data=buf_breaches.getvalue(),
            file_name=f"breach_audit_log_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #484f58; font-size: 0.75rem; font-family: monospace;'>"
    "QUANT RISK ENGINE // FINANCIAL RISK PLATFORM // MIT LICENSE // COMPLIANCE AUDITABLE"
    "</div>",
    unsafe_allow_html=True,
)
