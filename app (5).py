import streamlit as st
st.set_page_config(page_title="Q-Emerge", layout="wide")


import os
import time
import numpy as np
import pandas as pd
import requests
from scipy.stats import genpareto
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.express as px
import logging
import subprocess
import sys

# ── 1) Access Token via Query Param (single API) ───────────────────────────────
# This uses the stable, non‑experimental API in Streamlit ≥1.18
#token = st.query_params.get("token", [""])[0]
#ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "Brunellome")

#if token != ACCESS_TOKEN:
   # st.error("Access denied. Append ?token=Brunellome to the URL.")
   # st.stop()

# ── 2) Suppress logs ───────────────────────────────────────────────────────────
logging.getLogger("streamlit").setLevel(logging.ERROR)


# ─ Detect running on Streamlit Community Cloud ─────────────────────────────
ON_STREAMLIT_CLOUD = "STREAMLIT_SERVER_PORT" in os.environ

# ──Qiskit availability ──────────────────────────────────────────────────────
QISKIT_AVAILABLE = False

if not ON_STREAMLIT_CLOUD:
    # Only attempt Qiskit locally (e.g. in Colab or your laptop)
    try:
        import qiskit
        QISKIT_AVAILABLE = True
    except ImportError:
        st.sidebar.info("Qiskit not installed—quantum routines disabled locally.")
else:
    # On Streamlit Cloud: skip Qiskit entirely to avoid build failures
    st.sidebar.info("Running on Streamlit Cloud: quantum disabled (classical only)")

# ── Now import Qiskit primitives if available ───────────────────────────────────
if QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit
    from qiskit.primitives import Sampler
    from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem

# CCKP API endpoints
PRECIP_API_URL = (
    "https://cckpapi.worldbank.org/cckp/v1/"
    "cru-x0.5_timeseries_pr_timeseries_annual_1901-2023_mean_historical_cru_ts4.08_mean/"
    "all_countries_subnationals?_format=json"
)
TEMP_API_URL = (
    "https://cckpapi.worldbank.org/cckp/v1/"
    "cru-x0.5_timeseries_tas_timeseries_monthly_1901-2023_mean_historical_cru_ts4.08_mean/"
    "all_countries_subnationals?_format=json"
)

@st.cache_data(ttl=3600)
def fetch_cckp_multi(api_url, var_name):
    resp = requests.get(api_url)
    resp.raise_for_status()
    payload = resp.json().get("data", {})
    out = {}
    for iso, times in payload.items():
        recs = []
        for ym, v in times.items():
            if v is None: continue
            try:
                year = int(ym.split('-')[0])
                val  = float(v)
            except:
                continue
            recs.append((year, val))
        if not recs: continue
        df = (pd.DataFrame(recs, columns=["Year", var_name])
              .drop_duplicates("Year")
              .set_index("Year")
              .sort_index())
        out[iso] = df
    return out

precip_all = fetch_cckp_multi(PRECIP_API_URL, "pr_mm")
temp_all   = fetch_cckp_multi(TEMP_API_URL,   "tas_C")


st.title("Q-EMERGE: The Hybrid-Quantum Classical Prototype")

# Sidebar controls
available = sorted(set(precip_all) & set(temp_all))
country = st.sidebar.selectbox("Country", available, index=available.index("USA"))
variable = st.sidebar.selectbox("Variable", ["Precipitation", "Temperature"])
window = st.sidebar.slider("Rolling window (years)", 20, 100, 50, 5)
run_btn = st.sidebar.button("Run Analysis")

# ── Quick QPO Demo in Sidebar ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Rapid QPO Demo")

# Let the user choose how many “assets” and the total budget
num_assets = st.sidebar.slider("Number of exposures", 1, 10, 3, key="demo_num_assets")
budget     = st.sidebar.number_input("Total budget ($)", min_value=0, value=1_000_000, step=50_000, key="demo_budget")

if st.sidebar.button("Run Demo QPO", key="demo_qpo"):
    # Simple dirichlet‐based stub allocation
    weights = np.random.dirichlet(np.ones(num_assets))
    allocations = (weights * budget).round(2)
    assets = [f"Exposure {i+1}" for i in range(num_assets)]
    
    df_demo = pd.DataFrame({
        "Asset":      assets,
        "Allocation ($)": allocations
    })
    st.sidebar.table(df_demo)
    
    # Optional: let them download this stub as a CSV audit log
    csv = df_demo.to_csv(index=False).encode()
    st.sidebar.download_button(
        "Download Demo Audit Log",
        data=csv,
        file_name="demo_qpo_audit.csv",
        key="demo_audit_dl"
    )

# ── 5) Core Tail‑Risk Analysis ─────────────────────────────────────────────────
def tail_risk_analysis(series):
    data = series.values

    # 1) Adaptive threshold & exceedances
    for pct in (95, 90, 85):
        thr = np.percentile(data, pct/100)
        exc = data[data > thr] - thr
        if len(exc) >= 10:
            break
    if len(exc) < 10:
        st.warning("Too few extremes to fit GPD.")
        return None

    # 2) Fit GPD and cap ξ
    try:
        xi, loc, scale = genpareto.fit(exc)
        xi = float(np.clip(xi, -0.9, 0.9))
    except Exception as e:
        st.error(f"GPD fit failed: {e}")
        return None

    frac = len(exc) / len(data)

    # 3) Bootstrap CI on tail p
    boot_ps = []
    for _ in range(200):
        samp = np.random.choice(data, size=len(data), replace=True)
        tb   = np.percentile(samp, pct/100)
        eb   = samp[samp > tb] - tb
        if len(eb) > 0:
            boot_ps.append(len(eb) / len(samp))
    p_lo, p_hi = np.percentile(boot_ps, [2.5, 97.5])

    # 4) Classical Monte Carlo (guard ξ ≤ 0)
    t0 = time.time()
    if xi > 0:
        samples = np.random.pareto(xi, 100_000) * scale + thr
        mc_p = float((samples > thr).mean())
    else:
        mc_p = frac
    t_mc = time.time() - t0

    # 5) Assemble base result
    result = {
        "pct": pct,
        "thr": thr,
        "xi": xi,
        "scale": scale,
        "p_lo": p_lo,
        "p_hi": p_hi,
        "mc_p": mc_p,
        "t_mc": t_mc,
    }

    # 6) Quantum IAE if available
    if QISKIT_AVAILABLE:
        theta = 2 * np.arcsin(np.sqrt(frac))
        qc    = QuantumCircuit(1)
        qc.ry(theta, 0)
        prob  = EstimationProblem(state_preparation=qc, objective_qubits=[0])
        iae   = IterativeAmplitudeEstimation(0.001, 0.05, Sampler())
        t0    = time.time()
        res   = iae.estimate(prob)
        t_q   = time.time() - t0
        q_p   = float(res.estimation)
        lo_q, hi_q = res.confidence_interval
        q_err = (hi_q - lo_q) / 2

        result.update({
            "q_p":   q_p,
            "lo_q":  lo_q,
            "hi_q":  hi_q,
            "t_q":   t_q,
            "q_err": q_err,
            "speed": t_mc / t_q if t_q > 0 else None,
        })
    else:
        result.update({
            "q_p":   None,
            "lo_q":  None,
            "hi_q":  None,
            "t_q":   None,
            "q_err": None,
            "speed": None,
        })

    return result



# ── 6) Run & Display Results ──────────────────────────────────────────────────
if run_btn:
    # -- Prepare data & run analysis once --
    df_p    = precip_all[country]
    df_t    = temp_all[country]
    df      = df_p.join(df_t, how="inner")
    df      = df.loc[df.index.max()-window+1 : df.index.max()]
    col     = "pr_mm" if variable=="Precipitation" else "tas_C"
    series  = df[col]
    result  = tail_risk_analysis(series)
    # compute expected shortfall at pct
    thr      = result["thr"]
    exc_data = np.sort(series[series > thr])
    cvar     = exc_data.mean() if len(exc_data)>0 else thr

    # -- Build tabs --
    tab1, tab2, tab3, tab4 = st.tabs([
        "Analysis", "Explainability", "Capital Allocation", "Impact Dashboard"
    ])

    # ── Tab 1: Analysis ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader(f"{variable} Tail-Risk Report for {country}")
        # 1) Key stats table
        stats = {
            "Threshold (95th pct)": thr,
            "Tail Fraction": result["mc_p"],
            "Expected Shortfall (CVaR)": cvar,
            "GPD ξ": result["xi"],
            "GPD σ": result["scale"],
            "Bootstrap CI": f"[{result['p_lo']:.4f}, {result['p_hi']:.4f}]",
        }
        st.table(pd.DataFrame.from_dict(stats, orient="index", columns=["Value"]))

        # 2) Return‐period plot
        emp_rp = 1 / (1 - series.rank(pct=True))
        fitted_cdf = lambda x: 1 - (1 + result["xi"] * (x - thr) / result["scale"]) ** (-1/result["xi"])
        xs = np.linspace(series.min(), series.max(), 200)
        import plotly.express as px
        df_rp = pd.DataFrame({
            "Value": series,
            "Empirical RP": emp_rp
        })
        df_fit = pd.DataFrame({
            "Value": xs,
            "GPD RP": 1 / (1 - fitted_cdf(xs))
        })
        fig_rp = px.line(df_rp, x="Value", y="Empirical RP", title="Return‐Period Plot")
        fig_rp.add_scatter(x=df_fit["Value"], y=df_fit["GPD RP"], mode="lines", name="GPD Fit")
        st.plotly_chart(fig_rp, use_container_width=True)

        # 3) Histogram + fitted tail overlay
        fig_hist = px.histogram(series, nbins=30, title="Data Histogram with Tail Fit")
        tail_x = xs[xs>thr]
        tail_pdf = genpareto.pdf(tail_x - thr, c=result["xi"], scale=result["scale"])
        # scale tail_pdf to histogram height
        tail_pdf *= len(series) * (series.max()-series.min())/30
        fig_hist.add_scatter(x=tail_x, y=tail_pdf, mode="lines", name="GPD PDF", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Tab 2: Explainability ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Explainability Service")
        # Build a surrogate linear model on the last N years
        from sklearn.linear_model import LinearRegression
        N = min(10, len(series))
        X = np.arange(len(series))[-N:].reshape(-1,1)
        y = series.values[-N:]
        lr = LinearRegression().fit(X,y)
        # Compute Shapley‐like contributions: effect of each year's index
        shap_contribs = lr.coef_[0] * X.flatten()
        years         = series.index[-N:]
        df_shap = pd.DataFrame({
            "Year": years,
            "Contribution": shap_contribs
        }).sort_values("Contribution", key=lambda s: s.abs(), ascending=False)
        st.bar_chart(df_shap.set_index("Year"))

        # Counterfactual slider
        year_choice = st.select_slider("Pick a year to tweak", options=list(years))
        new_val     = st.number_input(f"New value for {year_choice}", value=series.loc[year_choice])
        if st.button("Run Counterfactual", key="cf"):
            cf_series = series.copy()
            cf_series.loc[year_choice] = new_val
            cf_res = tail_risk_analysis(cf_series)
            st.write(f"> New tail‐fraction: **{cf_res['mc_p']:.4f}** (was {result['mc_p']:.4f})")

    # ── Tab 3: Capital Allocation (QPO) ─────────────────────────────────────────
    with tab3:
        st.subheader("Portfolio Optimizer Stub")
        exps = st.text_input("Exposures (comma-sep)", "100,200,150", key="exp")
        budget = st.number_input("Budget", value=1_000_000, key="bud")
        if st.button("Optimize Capital", key="opt"):
            arr = np.array([float(e) for e in exps.split(",")])
            alloc = np.random.dirichlet(np.ones(len(arr))) * budget
            df_alloc = pd.DataFrame({"Exposure":arr, "Allocated":alloc})
            st.table(df_alloc)
            # stub audit log
            csv = df_alloc.to_csv(index=False).encode()
            st.download_button("Download Audit Log", data=csv,
                               file_name=f"audit_{country}.csv", key="auditdl")

    # ── Tab 4: Impact Dashboard ─────────────────────────────────────────────────
      
    with tab4:
        st.subheader("Impact Dashboard")

        # Build precise KPI table: baseline vs. pilot
        kpis = pd.DataFrame({
            "Baseline": [
                0.15,      # CI ±15%
                1.00,      # Risk capital = 100%
                4.0,       # Pricing cycle = 4 weeks
                60.0,      # Payout latency = 60 days
                15.0       # Admin cost = $15
            ],
            "Pilot": [
                abs(result["p_hi"] - result["p_lo"]) / 2,  # New CI
                0.90,      # Risk capital reduced to 90%
                2.0 / 7,   # Pricing cycle = 2 days ⇒ 0.29 weeks
                5.0 / 1440,# Payout latency = 5 minutes ⇒ 0.0035 days
                3.0        # Admin cost = $3
            ],
        }, index=[
            "CI (fraction)",
            "Risk Capital (fraction)",
            "Pricing Cycle (weeks)",
            "Payout Latency (days)",
            "Admin Cost per Policy ($)"
        ])

        # Display KPIs as a Plotly grouped bar chart
        import plotly.express as px
        fig_kpi = px.bar(
            kpis,
            barmode="group",
            title="Year 1 Pilot vs. Baseline Impact",
            labels={"value": "Metric Value", "index": "Metric"}
        )
        st.plotly_chart(fig_kpi, use_container_width=True)

        # Cumulative admin cost savings over 10 years
        years = np.arange(1, 11)
        baseline_cost = 15 * years
        pilot_cost    = 3 * years
        df_save = pd.DataFrame({
            "Baseline": baseline_cost,
            "Pilot":    pilot_cost
        }, index=years)

        fig_save = px.area(
            df_save,
            title="Cumulative Admin Costs Over 10 Years",
            labels={"index": "Year", "value": "Total Cost ($)", "variable": "Scenario"}
        )
        st.plotly_chart(fig_save, use_container_width=True)

