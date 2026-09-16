"""Quant Risk Engine - Institutional Portfolio Analytics & Risk Dashboard.

Powered by Streamlit, Plotly, and the FRP (Financial Risk Package) engine.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure src/ is in python path for Streamlit Cloud deployment
SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Risk Engine | Institutional Risk Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(20, 24, 39, 0.85) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }
    .status-badge-pass {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-fail {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Loading & Caching Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_ticker_prices(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    """Fetch close prices for multiple tickers."""
    prices_dict = {}
    for ticker in tickers:
        try:
            df = fetch_prices(ticker.strip().upper(), period=period)
            if "Close" in df.columns:
                series = df["Close"]
            elif "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                series = df.iloc[:, 0]
            # Flatten multiindex if needed
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series.name = ticker.strip().upper()
            prices_dict[ticker.strip().upper()] = series
        except Exception as err:
            st.sidebar.warning(f"Could not fetch data for {ticker}: {err}")
    if not prices_dict:
        raise ValueError("No price data retrieved for the specified tickers.")
    combined = pd.DataFrame(prices_dict).dropna()
    return combined


def generate_synthetic_portfolio() -> pd.DataFrame:
    """Generate reproducible sample multi-asset price history."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=504, freq="B")
    assets = ["US_EQUITY", "TECH_GROWTH", "GOVT_BONDS", "COMMODITIES"]
    mean_daily = np.array([0.0004, 0.0006, 0.0001, 0.0003])
    vol_daily = np.array([0.012, 0.018, 0.005, 0.014])
    corr = np.array([
        [1.00,  0.75, -0.25,  0.30],
        [0.75,  1.00, -0.35,  0.35],
        [-0.25, -0.35,  1.00, -0.10],
        [0.30,  0.35, -0.10,  1.00],
    ])
    cov = np.outer(vol_daily, vol_daily) * corr
    sim_returns = rng.multivariate_normal(mean_daily, cov, size=len(dates))
    price_paths = 100.0 * np.exp(np.cumsum(sim_returns, axis=0))
    return pd.DataFrame(price_paths, index=dates, columns=assets)


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Data Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Quant Risk Engine")
    st.markdown("Mathematical risk analytics & portfolio optimization.")
    st.markdown("---")

    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Select Data Input",
        options=["Pre-configured Portfolios", "Live Yahoo Finance Tickers", "Upload CSV File"],
        index=0,
    )

    prices_df: pd.DataFrame | None = None

    if data_source == "Pre-configured Portfolios":
        preset = st.selectbox(
            "Select Portfolio Preset",
            options=[
                "Mega-Cap Tech (AAPL, MSFT, GOOGL, NVDA)",
                "Diversified Core (SPY, QQQ, GLD, TLT)",
                "High Beta & Crypto Equities (NVDA, TSLA, COIN, AMD)",
                "Synthetic Multi-Asset Benchmark (Offline Demo)",
            ],
            index=0,
        )
        period = st.selectbox("Historical Window", ["1y", "2y", "3y", "5y"], index=1)

        if "Mega-Cap Tech" in preset:
            tickers = ("AAPL", "MSFT", "GOOGL", "NVDA")
            prices_df = load_ticker_prices(tickers, period=period)
        elif "Diversified Core" in preset:
            tickers = ("SPY", "QQQ", "GLD", "TLT")
            prices_df = load_ticker_prices(tickers, period=period)
        elif "High Beta" in preset:
            tickers = ("NVDA", "TSLA", "COIN", "AMD")
            prices_df = load_ticker_prices(tickers, period=period)
        else:
            prices_df = generate_synthetic_portfolio()

    elif data_source == "Live Yahoo Finance Tickers":
        ticker_input = st.text_input("Enter Ticker Symbols (comma separated)", value="AAPL, MSFT, AMZN, JPM")
        period = st.selectbox("Historical Window", ["6mo", "1y", "2y", "3y", "5y"], index=2)
        tickers = tuple(t.strip().upper() for t in ticker_input.split(",") if t.strip())
        if st.button("Fetch Market Data", use_container_width=True) or ("prices_df" not in st.session_state):
            with st.spinner("Downloading price histories..."):
                try:
                    prices_df = load_ticker_prices(tickers, period=period)
                    st.session_state["cached_prices"] = prices_df
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    prices_df = generate_synthetic_portfolio()
        if prices_df is None and "cached_prices" in st.session_state:
            prices_df = st.session_state["cached_prices"]

    else:
        uploaded_file = st.file_uploader("Upload CSV (Date index, Asset Price columns)", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                numeric_df = df_upload.select_dtypes(include=[np.number]).dropna()
                if numeric_df.shape[1] >= 1:
                    prices_df = numeric_df
                    st.success(f"Loaded {prices_df.shape[1]} assets, {len(prices_df)} rows.")
                else:
                    st.error("Uploaded CSV must contain at least 1 numeric asset column.")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
        else:
            st.info("Using synthetic portfolio until CSV is uploaded.")
            prices_df = generate_synthetic_portfolio()

    st.markdown("---")
    st.subheader("⚙️ Risk Parameters")
    confidence_level = st.slider(
        "Confidence Level (1 - α)",
        min_value=0.90,
        max_value=0.999,
        value=DEFAULT_CONFIDENCE_LEVEL,
        step=0.005,
        format="%.3f",
    )
    risk_free_rate = st.number_input(
        "Risk-Free Rate (Annualized)",
        min_value=0.00,
        max_value=0.15,
        value=0.03,
        step=0.005,
        format="%.3f",
    )
    mc_sims = st.select_slider(
        "Monte Carlo Iterations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=DEFAULT_MONTE_CARLO_SIMULATIONS,
    )

# Fallback check
if prices_df is None or prices_df.empty:
    st.warning("Loading default portfolio...")
    prices_df = generate_synthetic_portfolio()

asset_names = list(prices_df.columns)
n_assets = len(asset_names)

# Compute daily log returns panel
returns_df = pd.DataFrame()
for col in asset_names:
    returns_df[col] = log_returns(prices_df[col])
returns_df = returns_df.dropna()

# -----------------------------------------------------------------------------
# Portfolio Weight Allocation Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("⚖️ Portfolio Allocation")
    weight_mode = st.radio(
        "Weighting Strategy",
        options=["Equal Weighted", "Custom Allocation", "Global Min Variance", "Max Sharpe Ratio"],
        index=0,
    )

    expected_asset_returns = returns_df.mean(axis=0) * TRADING_DAYS_PER_YEAR
    cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR

    if weight_mode == "Equal Weighted":
        weights = pd.Series(1.0 / n_assets, index=asset_names)
    elif weight_mode == "Global Min Variance":
        weights = global_minimum_variance_weights(returns_df.cov())
    elif weight_mode == "Max Sharpe Ratio":
        optimizer = PortfolioOptimizer(expected_asset_returns, cov_matrix)
        weights = optimizer.max_sharpe(risk_free_rate=risk_free_rate)
    else:
        st.caption("Adjust percentage allocation per asset:")
        raw_weights = {}
        for a in asset_names:
            raw_weights[a] = st.slider(f"{a} (%)", 0, 100, int(100 / n_assets), 1)
        total_raw = sum(raw_weights.values())
        if total_raw > 0:
            weights = pd.Series({k: v / total_raw for k, v in raw_weights.items()})
        else:
            weights = pd.Series(1.0 / n_assets, index=asset_names)

# Compute aggregated portfolio daily return series
portfolio_daily_returns = returns_df @ weights
portfolio_annual_return = float(portfolio_return(weights, returns_df.mean(axis=0)) * TRADING_DAYS_PER_YEAR)
portfolio_annual_vol = float(np.sqrt(max(0.0, portfolio_variance(weights, cov_matrix))))
portfolio_sharpe = (
    (portfolio_annual_return - risk_free_rate) / portfolio_annual_vol
    if portfolio_annual_vol > 1e-6
    else 0.0
)

# Core Risk Measures
var_hist = VaR.historical(portfolio_daily_returns, confidence=confidence_level)
var_param = VaR.parametric(portfolio_daily_returns, confidence=confidence_level)
var_mc = VaR.monte_carlo(
    portfolio_daily_returns,
    confidence=confidence_level,
    simulations=mc_sims,
    seed=DEFAULT_MONTE_CARLO_SEED,
)
cvar_hist = CVaR.historical(portfolio_daily_returns, confidence=confidence_level)
cvar_param = CVaR.parametric(portfolio_daily_returns, confidence=confidence_level)

# -----------------------------------------------------------------------------
# Main Dashboard Header & Metrics Overview
# -----------------------------------------------------------------------------
st.title("📊 Quantitative Risk & Portfolio Engine")
st.caption(
    f"Active Universe: **{', '.join(asset_names)}** | Observations: **{len(portfolio_daily_returns)} trading days** | Confidence: **{confidence_level*100:.1f}%**"
)

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Annualized Return</div>
            <div class="metric-value">{portfolio_annual_return*100:+.2f}%</div>
            <div class="metric-sub">Mean log return × {TRADING_DAYS_PER_YEAR}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Annualized Volatility</div>
            <div class="metric-value">{portfolio_annual_vol*100:.2f}%</div>
            <div class="metric-sub">√Portfolio Variance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{portfolio_sharpe:.2f}</div>
            <div class="metric-sub">Rf = {risk_free_rate*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Historical VaR ({confidence_level*100:.1f}%)</div>
            <div class="metric-value" style="color: #fb923c;">{var_hist*100:.2f}%</div>
            <div class="metric-sub">Empirical 1-day tail loss</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Parametric VaR ({confidence_level*100:.1f}%)</div>
            <div class="metric-value" style="color: #38bdf8;">{var_param*100:.2f}%</div>
            <div class="metric-sub">Gaussian assumption</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi6:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CVaR / Expected Shortfall</div>
            <div class="metric-value" style="color: #f87171;">{cvar_hist*100:.2f}%</div>
            <div class="metric-sub">Average tail loss beyond VaR</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabs Layout for Core Modules
# -----------------------------------------------------------------------------
tab_var, tab_markowitz, tab_backtest, tab_composition, tab_reports = st.tabs([
    "📉 Value-at-Risk & Tails",
    "📈 Markowitz Optimization",
    "🧪 Model Backtesting",
    "💼 Portfolio & Universe",
    "📑 Report & Data Export",
])

# =============================================================================
# TAB 1: VaR & CVaR Deep Dive
# =============================================================================
with tab_var:
    st.subheader("Value-at-Risk (VaR) & Conditional Value-at-Risk (CVaR) Analysis")
    st.markdown(
        """
        **Value-at-Risk** defines the threshold loss that is only expected to be exceeded with probability $1 - \alpha$ over a 1-day horizon.
        **Expected Shortfall (CVaR)** computes the expected loss conditional upon breaching the VaR threshold:
        $$\\text{CVaR}_\\alpha = \\mathbb{E}[L \\mid L \\ge \\text{VaR}_\\alpha]$$
        """
    )

    vcol1, vcol2 = st.columns([2, 1])

    with vcol1:
        # Loss & Return Distribution Plot
        losses = -portfolio_daily_returns.to_numpy()
        returns_arr = portfolio_daily_returns.to_numpy()

        fig_dist = go.Figure()
        # Histogram of returns
        fig_dist.add_trace(
            go.Histogram(
                x=returns_arr,
                name="Daily Returns",
                nbinsx=60,
                opacity=0.65,
                marker_color="#6366f1",
                histnorm="probability density",
            )
        )

        # VaR Vertical lines (negative return threshold)
        fig_dist.add_vline(
            x=-var_hist,
            line_width=2.5,
            line_dash="dash",
            line_color="#f97316",
            annotation_text=f"Hist VaR: -{var_hist*100:.2f}%",
            annotation_position="top left",
        )
        fig_dist.add_vline(
            x=-var_param,
            line_width=2,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Param VaR: -{var_param*100:.2f}%",
            annotation_position="bottom left",
        )
        fig_dist.add_vline(
            x=-cvar_hist,
            line_width=3,
            line_dash="solid",
            line_color="#ef4444",
            annotation_text=f"CVaR: -{cvar_hist*100:.2f}%",
            annotation_position="top left",
        )

        fig_dist.update_layout(
            title=f"Empirical Return Distribution & Tail Risk Cutoffs ({confidence_level*100:.1f}%)",
            xaxis_title="Daily Log Return",
            yaxis_title="Probability Density",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            height=420,
            showlegend=True,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with vcol2:
        st.markdown("#### Risk Measures Comparison")
        comp_df = pd.DataFrame(
            {
                "Method": [
                    "Historical VaR",
                    "Parametric VaR",
                    f"Monte Carlo VaR ({mc_sims:,} sims)",
                    "Empirical CVaR (Expected Shortfall)",
                    "Parametric CVaR (Gaussian)",
                ],
                "1-Day Loss (%)": [
                    f"{var_hist*100:.3f}%",
                    f"{var_param*100:.3f}%",
                    f"{var_mc*100:.3f}%",
                    f"{cvar_hist*100:.3f}%",
                    f"{cvar_param*100:.3f}%",
                ],
                "Interpretation": [
                    "Model-free empirical tail quantile",
                    "Assumes i.i.d. Gaussian returns",
                    "Fitted normal simulation cross-check",
                    "Empirical mean of tail exceedances",
                    "Closed-form normal conditional expectation",
                ],
            }
        )
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # Bar chart comparison
        methods = ["Historical", "Parametric", "Monte Carlo", "CVaR (Hist)", "CVaR (Param)"]
        values = [var_hist * 100, var_param * 100, var_mc * 100, cvar_hist * 100, cvar_param * 100]
        colors = ["#f97316", "#38bdf8", "#818cf8", "#ef4444", "#f43f5e"]

        fig_bar = go.Figure(
            go.Bar(
                x=methods,
                y=values,
                marker_color=colors,
                text=[f"{v:.2f}%" for v in values],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            title="Loss Expectation by Method (%)",
            yaxis_title="Loss Percentage (%)",
            template="plotly_dark",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Multi-Confidence Sensitivity Curve
    st.markdown("---")
    st.subheader("Confidence Level Sensitivity (90% to 99.9%)")
    conf_range = np.linspace(0.90, 0.999, 40)
    curve_data = []
    for c in conf_range:
        h_v = VaR.historical(portfolio_daily_returns, confidence=c)
        p_v = VaR.parametric(portfolio_daily_returns, confidence=c)
        c_v = CVaR.historical(portfolio_daily_returns, confidence=c)
        curve_data.append({
            "Confidence": c * 100,
            "Historical VaR": h_v * 100,
            "Parametric VaR": p_v * 100,
            "CVaR (Expected Shortfall)": c_v * 100,
        })
    curve_df = pd.DataFrame(curve_data)

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df["Confidence"], y=curve_df["Historical VaR"], name="Historical VaR", line=dict(color="#f97316", width=2.5)))
    fig_curve.add_trace(go.Scatter(x=curve_df["Confidence"], y=curve_df["Parametric VaR"], name="Parametric VaR", line=dict(color="#38bdf8", width=2, dash="dash")))
    fig_curve.add_trace(go.Scatter(x=curve_df["Confidence"], y=curve_df["CVaR (Expected Shortfall)"], name="CVaR (Expected Shortfall)", line=dict(color="#ef4444", width=3)))

    fig_curve.update_layout(
        title="VaR & CVaR Progression Across Confidence Levels",
        xaxis_title="Confidence Level (%)",
        yaxis_title="Estimated Loss (%)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_curve, use_container_width=True)

# =============================================================================
# TAB 2: Markowitz Portfolio Optimization & Efficient Frontier
# =============================================================================
with tab_markowitz:
    st.subheader("Markowitz Mean-Variance Optimization & Efficient Frontier")
    st.markdown(
        """
        Closed-form Lagrange multiplier solution for the unconstrained mean-variance framework:
        $$\\min_w w^T \\Sigma w \\quad \\text{s.t.} \\quad w^T \\mu = r_{\\text{target}}, \\quad w^T \\mathbf{1} = 1$$
        """
    )

    optimizer = PortfolioOptimizer(expected_asset_returns, cov_matrix)
    gmv_w = optimizer.minimum_variance()
    gmv_ret = float(portfolio_return(gmv_w, expected_asset_returns))
    gmv_vol = float(np.sqrt(max(0.0, portfolio_variance(gmv_w, cov_matrix))))

    tan_w = optimizer.max_sharpe(risk_free_rate=risk_free_rate)
    tan_ret = float(portfolio_return(tan_w, expected_asset_returns))
    tan_vol = float(np.sqrt(max(0.0, portfolio_variance(tan_w, cov_matrix))))
    tan_sharpe = (tan_ret - risk_free_rate) / tan_vol if tan_vol > 1e-6 else 0.0

    # Efficient frontier curve points
    frontier_points = optimizer.efficient_frontier(n_points=120)
    frontier_vols = [p["volatility"] * 100 for p in frontier_points]
    frontier_rets = [p["return"] * 100 for p in frontier_points]

    ocol1, ocol2 = st.columns([2, 1])

    with ocol1:
        fig_frontier = go.Figure()

        # Frontier Curve
        fig_frontier.add_trace(
            go.Scatter(
                x=frontier_vols,
                y=frontier_rets,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#6366f1", width=3),
            )
        )

        # Capital Allocation Line (CAL) from Rf through Tangency Portfolio
        if tan_vol > 0:
            cal_x = np.linspace(0, max(frontier_vols) * 1.2, 50)
            cal_y = (risk_free_rate + tan_sharpe * (cal_x / 100.0)) * 100.0
            fig_frontier.add_trace(
                go.Scatter(
                    x=cal_x,
                    y=cal_y,
                    mode="lines",
                    name=f"Capital Allocation Line (CAL, Rf={risk_free_rate*100:.1f}%)",
                    line=dict(color="#e2e8f0", width=1.5, dash="dot"),
                )
            )

        # Individual Assets
        asset_vols = [float(np.sqrt(cov_matrix.loc[a, a])) * 100 for a in asset_names]
        asset_rets = [float(expected_asset_returns[a]) * 100 for a in asset_names]
        fig_frontier.add_trace(
            go.Scatter(
                x=asset_vols,
                y=asset_rets,
                mode="markers+text",
                name="Individual Assets",
                text=asset_names,
                textposition="top right",
                marker=dict(size=11, color="#38bdf8", symbol="diamond"),
            )
        )

        # Global Minimum Variance marker
        fig_frontier.add_trace(
            go.Scatter(
                x=[gmv_vol * 100],
                y=[gmv_ret * 100],
                mode="markers",
                name="Global Min Variance",
                marker=dict(size=14, color="#10b981", symbol="star"),
            )
        )

        # Maximum Sharpe Ratio marker
        fig_frontier.add_trace(
            go.Scatter(
                x=[tan_vol * 100],
                y=[tan_ret * 100],
                mode="markers",
                name="Max Sharpe (Tangency)",
                marker=dict(size=14, color="#f59e0b", symbol="hexagram"),
            )
        )

        # Current Portfolio marker
        fig_frontier.add_trace(
            go.Scatter(
                x=[portfolio_annual_vol * 100],
                y=[portfolio_annual_return * 100],
                mode="markers",
                name="Current Portfolio",
                marker=dict(size=15, color="#ec4899", symbol="cross"),
            )
        )

        fig_frontier.update_layout(
            title="Markowitz Efficient Frontier & Risk-Return Tradeoff",
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Annualized Expected Return (%)",
            template="plotly_dark",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_frontier, use_container_width=True)

    with ocol2:
        st.markdown("#### Optimal Portfolios Summary")
        opt_summary = pd.DataFrame({
            "Metric": ["Return (Ann.)", "Volatility (Ann.)", "Sharpe Ratio"],
            "Min Variance": [f"{gmv_ret*100:.2f}%", f"{gmv_vol*100:.2f}%", f"{(gmv_ret - risk_free_rate)/gmv_vol:.2f}"],
            "Max Sharpe": [f"{tan_ret*100:.2f}%", f"{tan_vol*100:.2f}%", f"{tan_sharpe:.2f}"],
            "Current Active": [f"{portfolio_annual_return*100:.2f}%", f"{portfolio_annual_vol*100:.2f}%", f"{portfolio_sharpe:.2f}"],
        })
        st.dataframe(opt_summary, hide_index=True, use_container_width=True)

        st.markdown("#### Target Return Optimizer")
        target_ret_input = st.slider(
            "Target Annual Return (%)",
            min_value=float(np.floor(min(asset_rets))),
            max_value=float(np.ceil(max(asset_rets) * 1.3)),
            value=float(np.round(portfolio_annual_return * 100, 1)),
            step=0.5,
        )
        target_ret_daily = (target_ret_input / 100.0) / TRADING_DAYS_PER_YEAR
        try:
            target_w = efficient_frontier_weights(returns_df.mean(axis=0), returns_df.cov(), target_ret_daily)
            target_vol_ann = float(np.sqrt(max(0.0, portfolio_variance(target_w, cov_matrix))))
            st.success(f"Optimized Volatility: **{target_vol_ann*100:.2f}%** | Sharpe: **{(target_ret_input/100.0 - risk_free_rate)/target_vol_ann:.2f}**")
        except Exception as e:
            st.error(f"Cannot compute target return portfolio: {e}")

    # Weights Comparison Bar Chart
    st.markdown("---")
    st.subheader("Asset Weight Allocations Across Strategies")
    weights_comp_df = pd.DataFrame({
        "Asset": asset_names,
        "Current Active": [weights[a] * 100 for a in asset_names],
        "Min Variance": [gmv_w[a] * 100 for a in asset_names],
        "Max Sharpe": [tan_w[a] * 100 for a in asset_names],
    })
    fig_weights = px.bar(
        weights_comp_df,
        x="Asset",
        y=["Current Active", "Min Variance", "Max Sharpe"],
        barmode="group",
        title="Weights Comparison (% Allocation)",
        template="plotly_dark",
        height=350,
        color_discrete_sequence=["#ec4899", "#10b981", "#f59e0b"],
    )
    fig_weights.update_layout(yaxis_title="Weight (%)", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_weights, use_container_width=True)

# =============================================================================
# TAB 3: Model Backtesting & Kupiec POF Test
# =============================================================================
with tab_backtest:
    st.subheader("Statistical Validation & Kupiec POF Backtesting")
    st.markdown(
        """
        The **Kupiec Proportion of Failures (POF)** test tests the null hypothesis that the observed failure rate
        matches the theoretical failure rate $p = 1 - \\alpha$:
        $$LR = -2 \\ln \\left( \\frac{(1-p)^{N-x} p^x}{(1-x/N)^{N-x} (x/N)^x} \\right) \\sim \\chi^2(1)$$
        """
    )

    bcol1, bcol2 = st.columns([1, 2])

    with bcol1:
        st.markdown("#### Backtest Configuration")
        backtest_model = st.selectbox(
            "VaR Model for Validation",
            options=["Historical VaR", "Parametric VaR", "Rolling 60-Day Historical VaR"],
            index=0,
        )
        test_significance = st.slider("Significance Level (α_test)", 0.01, 0.10, 0.05, 0.01)

        # Run test
        if backtest_model == "Historical VaR":
            active_var = var_hist
        elif backtest_model == "Parametric VaR":
            active_var = var_param
        else:
            # Rolling VaR
            rolling_var_series = portfolio_daily_returns.rolling(60).apply(
                lambda s: VaR.historical(s, confidence=confidence_level), raw=False
            ).shift(1)
            active_var = float(rolling_var_series.dropna().mean())

        passes, violations, expected_violations = kupiec_proportion_of_failures_test(
            portfolio_daily_returns,
            var=active_var,
            confidence=confidence_level,
            significance_level=test_significance,
        )

        n_obs = len(portfolio_daily_returns)
        actual_rate = (violations / n_obs) * 100.0 if n_obs > 0 else 0.0
        expected_rate = (1.0 - confidence_level) * 100.0

        # Traffic Light status (Basel Committee guidelines)
        if violations <= expected_violations * 1.3:
            zone = "🟢 GREEN ZONE (Reliable)"
        elif violations <= expected_violations * 2.0:
            zone = "🟡 YELLOW ZONE (Supervisory Review)"
        else:
            zone = "🔴 RED ZONE (Model Inadequate)"

        st.markdown("#### Test Diagnostics")
        status_html = (
            '<span class="status-badge-pass">✅ MODEL VALID (ACCEPTED)</span>'
            if passes
            else '<span class="status-badge-fail">❌ MODEL REJECTED</span>'
        )
        st.markdown(status_html, unsafe_allow_html=True)
        st.markdown(f"**Basel Traffic Light:** {zone}")

        st.markdown(
            f"""
            - **Total Observations:** `{n_obs}`
            - **Observed Violations:** `{violations}` ({actual_rate:.2f}%)
            - **Expected Violations:** `{expected_violations:.1f}` ({expected_rate:.2f}%)
            - **Significance Level:** `{test_significance:.2f}`
            """
        )

    with bcol2:
        # Exceedance Timeline Plot
        exceed_df = pd.DataFrame({"Return": portfolio_daily_returns})
        exceed_df["VaR_Threshold"] = -active_var
        exceed_df["Is_Breach"] = exceed_df["Return"] < exceed_df["VaR_Threshold"]

        fig_exceed = go.Figure()
        fig_exceed.add_trace(
            go.Scatter(
                x=exceed_df.index,
                y=exceed_df["Return"] * 100,
                mode="lines",
                name="Daily Returns (%)",
                line=dict(color="#94a3b8", width=1),
            )
        )
        fig_exceed.add_trace(
            go.Scatter(
                x=exceed_df.index,
                y=exceed_df["VaR_Threshold"] * 100,
                mode="lines",
                name=f"VaR Cutoff (-{active_var*100:.2f}%)",
                line=dict(color="#f97316", width=2, dash="dash"),
            )
        )
        # Breaches
        breaches_df = exceed_df[exceed_df["Is_Breach"]]
        if not breaches_df.empty:
            fig_exceed.add_trace(
                go.Scatter(
                    x=breaches_df.index,
                    y=breaches_df["Return"] * 100,
                    mode="markers",
                    name=f"Breaches ({violations})",
                    marker=dict(size=8, color="#ef4444", symbol="circle"),
                )
            )

        fig_exceed.update_layout(
            title="Historical Exceedance Timeline & VaR Violations",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_exceed, use_container_width=True)

    # Rolling Volatility & Risk Clustering
    st.markdown("---")
    st.subheader("Rolling 30-Day Annualized Volatility & Risk Clustering")
    roll_vol = rolling_volatility(portfolio_daily_returns, window=30, annualize=True)
    fig_roll = go.Figure()
    fig_roll.add_trace(
        go.Scatter(
            x=roll_vol.index,
            y=roll_vol * 100,
            name="Rolling 30d Volatility",
            line=dict(color="#a855f7", width=2),
            fill="tozeroy",
            fillcolor="rgba(168, 85, 247, 0.15)",
        )
    )
    fig_roll.update_layout(
        title="Portfolio Volatility Dynamics Over Time",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_roll, use_container_width=True)

# =============================================================================
# TAB 4: Portfolio Composition & Correlation Universe
# =============================================================================
with tab_composition:
    st.subheader("Portfolio Composition & Cross-Asset Correlation")

    ccol1, ccol2 = st.columns([1, 1])

    with ccol1:
        # Donut Chart Allocation
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=asset_names,
                    values=[max(0.0, float(weights[a])) for a in asset_names],
                    hole=0.45,
                    textinfo="label+percent",
                )
            ]
        )
        fig_donut.update_layout(
            title="Active Portfolio Weight Distribution",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with ccol2:
        # Correlation Heatmap
        corr_matrix = returns_df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Asset Log-Return Correlation Matrix",
            template="plotly_dark",
        )
        fig_corr.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Cumulative Performance Chart
    st.markdown("---")
    st.subheader("Normalized Asset Price Growth vs Portfolio (Indexed = 100)")
    norm_prices = (prices_df / prices_df.iloc[0]) * 100.0
    norm_portfolio = (1.0 + portfolio_daily_returns).cumprod() * 100.0

    fig_cum = go.Figure()
    for col in asset_names:
        fig_cum.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], name=col, opacity=0.7))
    fig_cum.add_trace(
        go.Scatter(
            x=norm_portfolio.index,
            y=norm_portfolio,
            name="PORTFOLIO (Active)",
            line=dict(color="#f8fafc", width=3.5),
        )
    )
    fig_cum.update_layout(
        title="Historical Cumulative Performance Trajectory",
        xaxis_title="Date",
        yaxis_title="Normalized Value (Base 100)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# =============================================================================
# TAB 5: Reports & Export
# =============================================================================
with tab_reports:
    st.subheader("Institutional Risk Report & Data Export")
    st.markdown("Export validated risk metrics, position weights, and backtest logs.")

    rcol1, rcol2 = st.columns(2)

    with rcol1:
        st.markdown("#### Comprehensive Metrics Summary")
        summary_export_df = pd.DataFrame({
            "Risk Metric": [
                "Analysis Date",
                "Universe",
                "Confidence Level",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "1-Day Historical VaR",
                "1-Day Parametric VaR",
                f"1-Day Monte Carlo VaR ({mc_sims:,} sims)",
                "1-Day CVaR / Expected Shortfall",
                "Kupiec POF Test Status",
                "Historical Violations Count",
                "Expected Violations Count",
            ],
            "Value": [
                str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                ", ".join(asset_names),
                f"{confidence_level*100:.1f}%",
                f"{portfolio_annual_return*100:+.2f}%",
                f"{portfolio_annual_vol*100:.2f}%",
                f"{portfolio_sharpe:.2f}",
                f"{var_hist*100:.3f}%",
                f"{var_param*100:.3f}%",
                f"{var_mc*100:.3f}%",
                f"{cvar_hist*100:.3f}%",
                "PASSED" if passes else "REJECTED",
                str(violations),
                f"{expected_violations:.1f}",
            ],
        })
        st.dataframe(summary_export_df, hide_index=True, use_container_width=True)

        # Download CSV
        csv_buffer = io.StringIO()
        summary_export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Risk Metrics Summary (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"risk_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with rcol2:
        st.markdown("#### Asset Portfolio Weights")
        weights_export_df = pd.DataFrame({
            "Asset": asset_names,
            "Active Weight (%)": [f"{weights[a]*100:.2f}%" for a in asset_names],
            "Min Variance Weight (%)": [f"{gmv_w[a]*100:.2f}%" for a in asset_names],
            "Max Sharpe Weight (%)": [f"{tan_w[a]*100:.2f}%" for a in asset_names],
            "Annualized Return": [f"{expected_asset_returns[a]*100:.2f}%" for a in asset_names],
            "Annualized Volatility": [f"{float(np.sqrt(cov_matrix.loc[a, a]))*100:.2f}%" for a in asset_names],
        })
        st.dataframe(weights_export_df, hide_index=True, use_container_width=True)

        weights_csv = io.StringIO()
        weights_export_df.to_csv(weights_csv, index=False)
        st.download_button(
            label="📥 Download Portfolio Allocations (CSV)",
            data=weights_csv.getvalue(),
            file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("#### Backtest Breaches Log")
        breaches_export_df = exceed_df[exceed_df["Is_Breach"]][["Return", "VaR_Threshold"]].copy()
        breaches_export_df["Return (%)"] = breaches_export_df["Return"] * 100
        breaches_export_df["VaR Cutoff (%)"] = breaches_export_df["VaR_Threshold"] * 100

        breaches_csv = io.StringIO()
        breaches_export_df.to_csv(breaches_csv)
        st.download_button(
            label=f"📥 Download Exceedances Log ({len(breaches_export_df)} breaches)",
            data=breaches_csv.getvalue(),
            file_name=f"var_exceedances_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Footer
st.markdown("---")
st.markdown(
    "<center><small style='color: #64748b;'>Quant Risk Engine | Built with NumPy, Pandas, SciPy, Plotly & Streamlit | MIT License</small></center>",
    unsafe_allow_html=True,
)
