
import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import genpareto
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
import matplotlib.pyplot as plt
import logging

# ── 0) Simple Password Protection ──────────────────────────────────────────────
PASSWORD = "Brunellome"  

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# If not authenticated yet, show login form
if not st.session_state.authenticated:
    pwd = st.text_input("Enter password to access the app:", type="password")
    if st.button("Login"):
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password")
    st.stop()  # stop here until correct password


# ── 1) Suppress verbose warnings ───────────────────────────────────────────────
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ── 2) Qiskit Import with Fallback ─────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import Sampler
    from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
    QISKIT_AVAILABLE = True
except Exception as e:
    logging.warning(f"Qiskit imports failed: {e}")
    QISKIT_AVAILABLE = False


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

@st.cache(ttl=3600, suppress_st_warning=True)
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

st.set_page_config(page_title="Quantum‑Classical Pricer", layout="wide")
st.title("Hybrid Quantum‑Classical Tail‑Risk Pricer Prototype")

# Sidebar controls
available = sorted(set(precip_all) & set(temp_all))
country = st.sidebar.selectbox("Country", available, index=available.index("USA"))
variable = st.sidebar.selectbox("Variable", ["Precipitation", "Temperature"])
window = st.sidebar.slider("Rolling window (years)", 20, 100, 50, 5)
run_btn = st.sidebar.button("Run Analysis")

# Demo QPO button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Demo QPO"):
    st.sidebar.write("**QPO: Capital Allocation Stub**")
    alloc = np.random.dirichlet(np.ones(3)) * 1e6
    st.sidebar.write("Allocations:", np.round(alloc, 2))

# ── 5) Core Tail‑Risk Analysis ─────────────────────────────────────────────────
def tail_risk_analysis(series):
    data = series.values
    # adaptive threshold
    for pct in (95, 90, 85):
        thr = np.percentile(data, pct/100)
        exc = data[data>thr] - thr
        if len(exc) >= 10: break
    if len(exc) < 10:
        st.warning("Too few extremes to fit GPD.")
        return None

    # GPD fit & cap
    xi, loc, scale = genpareto.fit(exc)
    xi = float(np.clip(xi, -0.9, 0.9))
    frac = len(exc) / len(data)

    # bootstrap fit CI
    boot_ps = []
    for _ in range(200):
        samp = np.random.choice(data, len(data), replace=True)
        tb   = np.percentile(samp, pct/100)
        eb   = samp[samp>tb] - tb
        if len(eb)>0:
            boot_ps.append(len(eb)/len(samp))
    p_lo, p_hi = np.percentile(boot_ps, [2.5,97.5])

    # classical MC
    t0 = time.time()
    mc_p = np.mean(np.random.pareto(xi,100000)*scale + thr > thr)
    t_mc = time.time() - t0

    result = {
        "pct": pct, "thr": thr, "xi": xi, "scale": scale,
        "p_lo": p_lo, "p_hi": p_hi,
        "mc_p": mc_p, "t_mc": t_mc
    }

    # quantum IAE if available
    if QISKIT_AVAILABLE:
        theta = 2*np.arcsin(np.sqrt(frac))
        qc = QuantumCircuit(1); qc.ry(theta,0)
        prob = EstimationProblem(state_preparation=qc, objective_qubits=[0])
        iae = IterativeAmplitudeEstimation(0.001,0.05,Sampler())
        t0 = time.time()
        res = iae.estimate(prob)
        t_q = time.time() - t0
        q_p = float(res.estimation)
        lo_q, hi_q = res.confidence_interval
        q_err = (hi_q - lo_q)/2
        result.update({"q_p": q_p, "lo_q": lo_q, "hi_q": hi_q,
                       "t_q": t_q, "q_err": q_err,
                       "speed": t_mc/t_q if t_q>0 else None})
    else:
        # fallback
        result.update({"q_p": None, "lo_q": None, "hi_q": None,
                       "t_q": None, "q_err": None, "speed": None})
        st.warning("Quantum estimation unavailable; using classical only.")

    return result


# ── 6) Run & Display Results ──────────────────────────────────────────────────
if run_btn:
    df_p = precip_all[country]
    df_t = temp_all[country]
    df   = df_p.join(df_t, how="inner")
    df   = df.loc[df.index.max()-window+1:df.index.max()]
    col  = "pr_mm" if variable=="Precipitation" else "tas_C"
    series = df[col]

    st.line_chart(series)

    result = tail_risk_analysis(series)
    if result:
        st.subheader(f"{variable} Tail‑Risk Report for {country}")
        st.write(f"- Threshold ({result['pct']}th pct): {result['thr']:.2f}")
        st.write(f"- GPD (ξ,σ): {result['xi']:.3f}, {result['scale']:.3f}")
        st.write(f"- Fit‑CI: [{result['p_lo']:.4f}, {result['p_hi']:.4f}]")
        st.write(f"- MC p̂ = {result['mc_p']:.4f} (t={result['t_mc']:.2f}s)")
        if QISKIT_AVAILABLE:
            st.write(f"- QAE p̂ = {result['q_p']:.4f} ±{result['q_err']:.4f} (t={result['t_q']:.2f}s)")
            st.write(f"- Speed‑up: {result['speed']:.1f}×")
        x = np.linspace(series.min(), series.max(), 200)
        pdf = genpareto.pdf(x-result['thr'], c=result['xi'], scale=result['scale'])
        fig, ax = plt.subplots()
        ax.plot(x, pdf); ax.axvline(result['thr'], color='red', ls='--')
        st.pyplot(fig)
    else:
        st.error("Analysis failed.")
