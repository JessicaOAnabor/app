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
from fpdf import FPDF
from io import BytesIO
import plotly.graph_objs as go
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
# -------- PDF REPORT ------
def create_pdf_report(country, variable, series, result, df_shap, df_alloc, kpis, df_save):
    """
    Build a PDF in memory that contains:
     - Title page
     - Tail-risk stats
     - Explainability table (df_shap)
     - QPO allocations (df_alloc)
     - Impact KPIs & savings curve
    Returns bytes.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Title Page ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "Hybrid Quantum-Classical Pricer Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 8, f"{country} — {variable}", ln=True, align="C")
    pdf.ln(10)

    # --- Tail-Risk Summary ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "1) Tail-Risk Analysis", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.ln(2)
    for k, v in {
        "Threshold (95th pct)": result["thr"],
        "Tail Fraction (empirical)": result["mc_p"],
        "Expected Shortfall (CVaR)": series[series>result["thr"]].mean() if (series>result["thr"]).any() else result["thr"],
        "GPD ξ": result["xi"],
        "GPD σ": result["scale"],
        "Bootstrap CI": f"[{result['p_lo']:.4f}, {result['p_hi']:.4f}]",
    }.items():
        pdf.cell(60, 6, f"{k}:", ln=False)
        pdf.cell(0, 6, f"{v:.4f}" if isinstance(v, float) else str(v), ln=True)
    pdf.ln(5)

    # --- Explainability Table ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "2) Explainability (Top Contributions)", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=10)
    # Table header
    pdf.cell(40, 6, "Year", border=1)
    pdf.cell(60, 6, "Contribution", border=1, ln=True)
    # Top 10 rows
    for year, contrib in df_shap.head(10).itertuples(index=False):
        pdf.cell(40, 6, str(int(year)), border=1)
        pdf.cell(60, 6, f"{contrib:.4f}", border=1, ln=True)
    pdf.ln(5)

    # --- Capital Allocation ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "3) Capital Allocation (QPO Stub)", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(40, 6, "Exposure", border=1)
    pdf.cell(60, 6, "Allocated ($)", border=1, ln=True)
    for exp, alloc in df_alloc.itertuples(index=False):
        pdf.cell(40, 6, f"{exp:.1f}", border=1)
        pdf.cell(60, 6, f"{alloc:.2f}", border=1, ln=True)
    pdf.ln(5)

    # --- Impact Dashboard ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "4) Impact Dashboard KPIs", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=12)
    for metric, row in kpis.iterrows():
        pdf.cell(60, 6, f"{metric}:", border=0)
        pdf.cell(0, 6, f"Baseline = {row['Baseline']:.4f}, Pilot = {row['Pilot']:.4f}", ln=True)
    pdf.ln(5)

    # --- Cumulative Savings Chart Data ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "5) Cumulative Admin-Cost Savings", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(20, 6, "Year", border=1)
    pdf.cell(40, 6, "Baseline Cost", border=1)
    pdf.cell(40, 6, "Pilot Cost", border=1, ln=True)
    for yr in df_save.index:
        pdf.cell(20, 6, str(int(yr)), border=1)
        pdf.cell(40, 6, f"{df_save.loc[yr,'Baseline']:.2f}", border=1)
        pdf.cell(40, 6, f"{df_save.loc[yr,'Pilot']:.2f}", border=1, ln=True)

    # Output bytes
    pdf_out = BytesIO()
    pdf.output(pdf_out)
    return pdf_out.getvalue()




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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Analysis", "Explainability", "Capital Allocation", "Impact Dashboard", "Capital Charge Gauge", "Global Choropleth"
    ])


    # ── Tab 1: Analysis with Dynamic Percentile Slider ─────────────────────────

    with tab1:
        st.subheader(f"{variable} Tail-Risk Explorer for {country}")

    # ─── Safe slider setup ────────────────────────────────────────────────────
        try:
            pct = st.slider(
                "Tail Threshold Percentile",
                min_value=80.0,
                max_value=99.9,
                value=95.0,
                step=0.1,
                key="pct_slider"
            )
        except Exception as e:
            st.error(f"Slider failed to initialize: {e}")
            pct = 95.0

    # ─── Compute threshold & exceedances ────────────────────────────────────────
        try:
            thr = float(np.percentile(series, pct / 100))
            exc = series[series > thr] - thr
            frac = len(exc) / len(series) if len(series) > 0 else 0.0
        except Exception as e:
            st.error(f"Failed computing threshold at {pct}%: {e}")
            thr, exc, frac = 0.0, pd.Series(dtype=float), 0.0

    # ─── GPD fit ────────────────────────────────────────────────────────────────
        if len(exc) >= 10:
            try:
                xi, loc, scale = genpareto.fit(exc)
                xi = float(np.clip(xi, -0.9, 0.9))
            except Exception as e:
                st.error(f"GPD fit error: {e}")
                xi = scale = np.nan
        else:
            xi = scale = np.nan
            st.warning("Not enough exceedances to fit GPD.")

    # ─── Display stats ─────────────────────────────────────────────────────────
        es = float(exc.mean()) if len(exc) > 0 else 0.0
        stats = {
            "Percentile":         f"{pct:.1f}%",
            "Threshold":          f"{thr:.2f}",
            "Tail Fraction":      f"{frac:.4f}",
            "Expected Shortfall": f"{es:.2f}",
            "GPD ξ":              f"{xi:.3f}" if not np.isnan(xi) else "n/a",
            "GPD σ":              f"{scale:.3f}" if not np.isnan(scale) else "n/a",
        }
        st.table(pd.DataFrame.from_dict(stats, orient="index", columns=["Value"]))

    # ─── Visualization ─────────────────────────────────────────────────────────
        try:
        # Histogram + tail fit
            import plotly.graph_objects as go
            hist = go.Histogram(x=series, nbinsx=30, name="Data", marker_color="lightblue")
            traces = [hist]

            if not np.isnan(xi):
                xs = np.linspace(thr, series.max(), 200)
                pdf_tail = genpareto.pdf(xs - thr, c=xi, scale=scale)
                scale_factor = len(series) * (series.max() - series.min()) / 30
                tail_line = go.Scatter(
                    x=xs,
                    y=pdf_tail * scale_factor,
                    mode="lines",
                    name="GPD Tail Fit",
                    line=dict(color="crimson"),
                    hovertemplate="<b>GPD PDF</b><br>Value=%{x:.2f}<br>Count≈%{y:.0f}<extra></extra>"
                )
                traces.append(tail_line)

            fig = go.Figure(
                data=traces,
                layout=go.Layout(
                    title=f"Histogram & Tail Fit @ {pct:.1f}th pct (thr={thr:.2f})",
                    xaxis_title=f"{variable} value",
                    yaxis_title="Count",
                    bargap=0.1
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Visualization failed: {e}")




    # ── Tab 2: Explainability ───────────────────────────────────────────────────
       
    with tab2:
        st.subheader("Explainability Service")

        # 1) Prepare surrogate Shapley DataFrame
        N = min(10, len(series))
        # extract the last N points
        tail_idx = series.index[-N:]
        # convert index to simple ints if possible
        try:
            years_int = [int(y) for y in tail_idx]
        except (TypeError, ValueError):
            # fallback to positional 0..N-1
            years_int = list(range(N))
        values = series.values[-N:]
        # linear surrogate
        from sklearn.linear_model import LinearRegression
        X = np.arange(N).reshape(-1,1)
        lr = LinearRegression().fit(X, values)
        contribs = lr.coef_[0] * X.flatten()

        df_shap = (
            pd.DataFrame({"Year": years_int, "Contribution": contribs})
              .sort_values("Contribution", key=lambda s: s.abs(), ascending=False)
        )

        # 2) Plot interactive bar chart
        fig_shap = px.bar(
            df_shap, x="Year", y="Contribution",
            title="Yearly Contributions to Tail-Risk (Surrogate)"
        )
        fig_shap.update_traces(
            hovertemplate="<b>Year</b>: %{x}<br><b>Contribution</b>: %{y:.3f}<extra></extra>"
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # 3) Counterfactual UI
        st.markdown("---")
        try:
            cf_year = st.select_slider(
                "Pick a year to tweak",
                options=years_int,
                key="cf_year"
            )
            # seed default from series if possible
            default_val = series.loc[tail_idx[years_int.index(cf_year)]] \
                          if cf_year in years_int else float(series.mean())
            cf_val = st.number_input(
                f"New value for year {cf_year}",
                value=float(default_val),
                key="cf_val"
            )
        except Exception as e:
            st.error(f"Slider setup failed: {e}")
            cf_year, cf_val = None, None

        if cf_year is not None and st.button("Run Counterfactual", key="cf_run"):
            try:
                # rebuild a small series with integer index for analysis
                ser_int = pd.Series(values, index=years_int)
                ser_int.loc[cf_year] = cf_val
                cf_res = tail_risk_analysis(ser_int)
                st.write(f"**New tail-fraction:** {cf_res['mc_p']:.4f} (was {result['mc_p']:.4f})")
            except Exception as e:
                st.error(f"Counterfactual failed: {e}")


        # ── Tab 3: QPO-Mimic — Multi-start Random Search Stub ────────────────────────
    with tab3:
        st.subheader("QPO-Demo: Quantum Capital Allocator")

        # 1) User inputs
        exps_input = st.text_input(
            "Enter exposures (comma-separated)", 
            value="100,200,150,300", 
            key="exp_input"
        )
        budget = st.number_input(
            "Total budget ($)", 
            value=1_000_000, 
            min_value=0.0, 
            step=10_000.0, 
            key="budget"
        )
        samples = st.slider(
            "Number of quantum samples (M)", 
            min_value=5, max_value=50, value=20, step=5, 
            key="m_samples"
        )

        st.markdown("---")
        # 2) Run button
        if st.button("Run Demo-QPO", key="demo_qpo"):
            try:
                # Parse exposures safely
                exps = [float(x.strip()) for x in exps_input.split(",") if x.strip()]
                if not exps:
                    raise ValueError("Please supply at least one exposure.")
                if budget <= 0:
                    raise ValueError("Budget must be positive.")
                
                exps = np.array(exps)
                
                # Objective: maximize sum(sqrt(e_i * a_i))
                def objective(a):
                    return np.sum(np.sqrt(exps * a))
                
                best_obj   = -np.inf
                best_alloc = None
                
                # Multi-start sampling
                for _ in range(samples):
                    cand = np.random.dirichlet(np.ones(len(exps))) * budget
                    score = objective(cand)
                    if score > best_obj:
                        best_obj   = score
                        best_alloc = cand
                
                # Display results
                df_alloc = pd.DataFrame({
                    "Exposure":      exps,
                    "Allocated ($)": best_alloc.round(2)
                })
                st.table(df_alloc)
                
                st.write(f"Simulated quantum samples: **{samples}**")  
                st.write(f"Best-of-{samples} objective = **{best_obj:.2f}**")
                
                # Audit-log download
                csv = df_alloc.to_csv(index=False).encode()
                st.download_button(
                    "Download QPO Audit Log (CSV)",
                    data=csv,
                    file_name=f"demo_qpo_audit_{country}.csv",
                    key="demo_qpo_audit_dl"
                )
            except Exception as e:
                st.error(f"Demo QPO failed: {e}")



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


        # PDF download
        st.markdown("---")
        try:
            if st.button("Download Full PDF Report", key="download_pdf"):
                # If df_alloc wasn’t defined (e.g. user hasn’t run Tab 3), create an empty frame
                alloc_df = locals().get("df_alloc", pd.DataFrame(columns=["Exposure", "Allocated ($)"]))
                # df_shap must exist from Tab 2
                shap_df = locals().get("df_shap", pd.DataFrame(columns=["Year", "Contribution"]))

                pdf_bytes = create_pdf_report(
                    country=country,
                    variable=variable,
                    series=series,
                    result=result,
                    df_shap=shap_df,
                    df_alloc=alloc_df,
                    kpis=kpis,
                    df_save=df_save
                )
                st.download_button(
                    label="Download Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{country}_{variable}_pricer_report.pdf",
                    mime="application/pdf",
                    key="pdf_dl"
                )
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")


      # ── Tab 5: Capital Charge Gauge ──────────────────────────────────────────────
        
    with tab5:
        st.subheader("Capital Charge (VaR Gauge)")

        # 1) Safe slider initialization
        try:
            conf = st.slider(
                "VaR Confidence Level (%)",
                min_value=90.0,
                max_value=99.9,
                value=99.5,
                step=0.1,
                key="var_conf"
            )
        except Exception as e:
            st.error(f"Slider error: {e}")
            conf = 99.5

        # 2) Compute empirical and (optionally) GPD‐based VaR
        try:
            thr_emp = np.percentile(series, conf/100)
        except Exception as e:
            st.error(f"Empirical VaR calculation failed: {e}")
            thr_emp = 0.0

        var_gpd = thr_emp
        if result.get("xi") is not None and result["xi"] > 0:
            try:
                p = 1 - conf/100
                xi    = result["xi"]
                scale = result["scale"]
                thr0  = result["thr"]
                var_gpd = thr0 + scale * ((p**(-xi) - 1) / xi)
            except Exception as e:
                st.warning(f"GPD VaR fallback to empirical: {e}")
                var_gpd = thr_emp

        # 3) Render the gauge
        try:
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=var_gpd,
                title={'text': f"{conf:.1f}% VaR (Capital Required)"},
                gauge={
                    'axis': {'range': [0, var_gpd * 1.5 or thr_emp * 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, var_gpd * 0.8 or thr_emp * 0.8], 'color': "lightgray"},
                        {'range': [var_gpd * 0.8 or thr_emp * 0.8, var_gpd or thr_emp], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': var_gpd
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception as e:
            st.error(f"Gauge rendering failed: {e}")


  # ── Tab 6: Global Choropleth ─────────────────────────────────────────────────
       
    with tab6:
        st.subheader("Global Tail-Risk Heatmap")

        # 1) Pull current percentile (or default to 95)
        pct_global = st.session_state.get("pct_slider", 95.0)

        # 2) Build per‐country tail fraction dict
        frac_map = {}
        for iso, df_iso in precip_all.items():
            try:
                vals = df_iso["pr_mm"].dropna().values
                if len(vals) < 2:
                    continue
                thr_i = np.percentile(vals, pct_global/100)
                exc_i = vals[vals > thr_i] - thr_i
                frac = len(exc_i) / len(vals)
                frac_map[iso] = frac
            except Exception:
                # skip any problematic country
                continue

        # 3) Assemble DataFrame
        df_map = (
            pd.DataFrame.from_dict(
                frac_map, orient="index", columns=["tail_fraction"]
            )
            .reset_index()
            .rename(columns={"index": "iso_alpha"})
        )

        # 4) Plot choropleth
        fig_map = px.choropleth(
            df_map,
            locations="iso_alpha",
            color="tail_fraction",
            color_continuous_scale="OrRd",
            range_color=(0, df_map["tail_fraction"].max()),
            labels={"tail_fraction": "Tail Fraction"},
            title=f"Global Tail Fraction @ {pct_global:.1f}th percentile"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_map, use_container_width=True)



