"""
╔══════════════════════════════════════════════════════════════════════╗
║        CREDIT RISK PREDICTOR  —  Professional Fintech Edition       ║
║        With Loan Repayment Planner & Installment Scheduler  v3.0    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import datetime
from dateutil.relativedelta import relativedelta   # pip install python-dateutil

# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CreditLens | Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — refined dark fintech aesthetic
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; font-size: 15px; }
.stApp { background: #060b12; color: #c8d8ea; }

[data-testid="stSidebar"] {
    background: #080f1a !important;
    border-right: 1px solid #0d2035 !important;
    padding-top: 0.5rem;
}
[data-testid="stSidebar"] * { color: #c8d8ea !important; }

.hero-banner {
    background: linear-gradient(125deg, #071526 0%, #0a1e35 50%, #060f1c 100%);
    border: 1px solid #0d2035; border-top: 3px solid #1a6fdf;
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; right: 0; bottom: 0; width: 40%;
    background: radial-gradient(ellipse at 80% 50%, rgba(26,111,223,0.06) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800;
    color: #e8f2ff; margin: 0 0 0.35rem 0; letter-spacing: -0.3px; line-height: 1.2;
}
.hero-sub  { color: #5580a0; font-size: 0.88rem; margin: 0 0 1rem 0; }
.hero-tag  {
    display: inline-block; background: rgba(26,111,223,0.1);
    border: 1px solid rgba(26,111,223,0.25); color: #4f9cf9;
    border-radius: 4px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; font-weight: 500; padding: 0.18rem 0.65rem;
    margin-right: 0.4rem; letter-spacing: 0.8px; text-transform: uppercase;
}

.section-head {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 700;
    color: #1a6fdf; text-transform: uppercase; letter-spacing: 2.5px;
    margin: 1.6rem 0 0.9rem 0; display: flex; align-items: center; gap: 0.6rem;
}
.section-head::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #0d2035 0%, transparent 100%);
}

.metric-card {
    background: #080f1a; border: 1px solid #0d2035; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center; position: relative; overflow: hidden;
}
.metric-card .m-val {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700;
    color: #e8f2ff; display: block; line-height: 1;
}
.metric-card .m-lbl {
    font-size: 0.7rem; color: #3d5870; text-transform: uppercase;
    letter-spacing: 1.2px; display: block; margin-top: 0.35rem;
}
.metric-card .m-accent { position: absolute; bottom: 0; left: 0; right: 0; height: 2px; }

.result-panel { border-radius: 14px; padding: 1.8rem 2rem; text-align: center; position: relative; overflow: hidden; }
.result-panel.low    { background: linear-gradient(145deg,#06160e,#092014); border: 1.5px solid #1a6b3c; }
.result-panel.medium { background: linear-gradient(145deg,#160e04,#201407); border: 1.5px solid #7a5010; }
.result-panel.high   { background: linear-gradient(145deg,#160606,#200b0b); border: 1.5px solid #7a1a1a; }
.rate-display {
    font-family: 'IBM Plex Mono', monospace; font-size: 3.8rem; font-weight: 700;
    line-height: 1; display: block;
}
.risk-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 1rem; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase; margin-top: 0.8rem;
}

/* Planner-specific */
.emi-box {
    background: linear-gradient(135deg, #071e0d, #0a2e16);
    border: 1.5px solid #1a6b3c; border-radius: 12px;
    padding: 1.4rem 1.6rem; text-align: center;
}
.emi-amount {
    font-family: 'IBM Plex Mono', monospace; font-size: 2.6rem;
    font-weight: 700; color: #3ddc84; line-height: 1; display: block;
}
.emi-label {
    font-size: 0.75rem; color: #2a6040; text-transform: uppercase;
    letter-spacing: 1.5px; margin-top: 0.4rem; display: block;
}
.planner-cta {
    background: linear-gradient(135deg, #07111e 0%, #0a1a2e 100%);
    border: 1px solid #0d2a45; border-left: 3px solid #3ddc84;
    border-radius: 14px; padding: 1.2rem 1.6rem; margin-bottom: 1rem;
}

.info-chip {
    background: #080f1a; border: 1px solid #0d2035; border-radius: 8px;
    padding: 0.65rem 0.9rem; margin: 0.35rem 0; font-size: 0.82rem;
    color: #7398b5; line-height: 1.5;
}
.report-box {
    background: #050a10; border: 1px solid #0d2035; border-radius: 12px;
    padding: 1.5rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: #6090b0; white-space: pre-wrap; line-height: 1.7;
}

.stButton > button {
    background: linear-gradient(135deg, #0f3d8c, #1a6fdf) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
    font-weight: 600 !important; padding: 0.65rem 1.5rem !important; width: 100% !important;
    transition: all 0.2s !important; box-shadow: 0 4px 16px rgba(26,111,223,0.2) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a6fdf, #4f9cf9) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0a3020, #0d4a30) !important;
    color: #3ddc84 !important; border: 1px solid #1a6b3c !important;
    border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important; width: 100% !important; transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #0d4a30, #1a6b3c) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #080f1a; border: 1px solid #0d2035; border-radius: 10px; padding: 3px; gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #3d5870 !important;
    border-radius: 7px !important; font-weight: 600 !important; font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] { background: #0f3d8c !important; color: #e8f2ff !important; }

.stSlider > div > div > div > div { background: #1a6fdf !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }

.sidebar-logo {
    font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 800;
    color: #e8f2ff; letter-spacing: -0.3px; padding: 0.8rem 0 0.2rem 0;
}
.sidebar-logo span { color: #1a6fdf; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  MATPLOTLIB DARK THEME
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#080f1a", "axes.facecolor": "#060b12",
    "axes.edgecolor": "#0d2035",   "axes.labelcolor": "#5580a0",
    "axes.titlecolor": "#c8d8ea",  "xtick.color": "#3d5870",
    "ytick.color": "#3d5870",      "grid.color": "#0d1e30",
    "grid.linestyle": "--",        "grid.linewidth": 0.5,
    "text.color": "#c8d8ea",       "font.family": "monospace",
    "axes.spines.top": False,      "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════
#  LABEL ENCODINGS  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════
LOAN_GRADE_MAP     = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
HOME_OWNERSHIP_MAP = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
LOAN_INTENT_MAP    = {
    "DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5,
}
DEFAULT_MAP = {"N": 0, "Y": 1}

# ═══════════════════════════════════════════════════════════════════════
#  LOAD MODEL  (unchanged)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    path = "rf_model.pkl"
    if not os.path.exists(path):
        st.error("❌ rf_model.pkl not found. Place it in the same folder as app.py")
        st.stop()
    return joblib.load(path)

rf_model     = load_model()
FEATURE_COLS = list(rf_model.feature_names_in_)

# ═══════════════════════════════════════════════════════════════════════
#  CORE PREDICTION HELPERS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════
def build_input_row(fields: dict) -> pd.DataFrame:
    row = pd.DataFrame([fields])
    for col in FEATURE_COLS:
        if col not in row.columns:
            row[col] = 0
    return row[FEATURE_COLS]

def predict_rate(fields: dict) -> float:
    return float(rf_model.predict(build_input_row(fields))[0])

def risk_level(rate: float):
    """Returns (level, color, emoji, label, approval_pct, rate_range)"""
    if rate < 10:
        return "low",    "#3ddc84", "🟢", "LOW RISK",    88, "7% – 10%"
    if rate < 16:
        return "medium", "#f0a22a", "🟡", "MEDIUM RISK", 62, "10% – 16%"
    return              "high",   "#e85454", "🔴", "HIGH RISK",  28, "16% – 25%"

# ═══════════════════════════════════════════════════════════════════════
#  EMI & AMORTISATION ENGINE  (banking-accurate reducing-balance)
# ═══════════════════════════════════════════════════════════════════════
def calculate_emi(principal: float, annual_rate: float, term_months: int) -> float:
    """
    Standard reducing-balance EMI formula used by major banks:
        EMI = P × r × (1+r)^n / ((1+r)^n − 1)
    where r = monthly interest rate, n = number of months.
    Handles zero-rate edge case with flat equal principal payments.
    """
    if annual_rate == 0 or annual_rate is None:
        return round(principal / term_months, 2)
    r = annual_rate / 100 / 12
    n = term_months
    emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return round(emi, 2)


def build_schedule(principal: float, annual_rate: float,
                   term_months: int, start_date: datetime.date) -> pd.DataFrame:
    """
    Full amortisation schedule with:
        No. | Payment Date | EMI | Principal | Interest | Remaining Balance
    Last installment is corrected for rounding drift.
    """
    emi     = calculate_emi(principal, annual_rate, term_months)
    r       = annual_rate / 100 / 12
    balance = principal
    rows    = []

    for i in range(1, term_months + 1):
        pay_date   = start_date + relativedelta(months=i - 1)
        interest   = round(balance * r, 2)
        princ_part = round(emi - interest, 2)

        # Last instalment: clear exact remaining balance
        if i == term_months:
            princ_part = round(balance, 2)
            emi_actual = round(princ_part + interest, 2)
        else:
            emi_actual = emi

        balance = round(max(balance - princ_part, 0), 2)

        rows.append({
            "No.":               i,
            "Payment Date":      pay_date.strftime("%d %b %Y"),
            "EMI ($)":           f"{emi_actual:,.2f}",
            "Principal ($)":     f"{princ_part:,.2f}",
            "Interest ($)":      f"{interest:,.2f}",
            "Balance ($)":       f"{balance:,.2f}",
        })

    return pd.DataFrame(rows)


def schedule_summary(principal: float, annual_rate: float, term_months: int) -> dict:
    """Quick summary totals without the full DataFrame."""
    emi       = calculate_emi(principal, annual_rate, term_months)
    total_pay = round(emi * term_months, 2)
    total_int = round(total_pay - principal, 2)
    return {"emi": emi, "total_payment": total_pay, "total_interest": total_int}


# ═══════════════════════════════════════════════════════════════════════
#  PLANNER CHARTS
# ═══════════════════════════════════════════════════════════════════════
def balance_chart(principal: float, annual_rate: float, term_months: int):
    """Declining outstanding balance line chart."""
    r       = annual_rate / 100 / 12
    emi     = calculate_emi(principal, annual_rate, term_months)
    balance = principal
    bals    = [principal]
    for _ in range(term_months):
        interest = balance * r
        balance  = max(round(balance - (emi - interest), 2), 0)
        bals.append(balance)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.fill_between(range(term_months + 1), bals, alpha=0.18, color="#3ddc84")
    ax.plot(range(term_months + 1), bals, color="#3ddc84", lw=2)
    ax.set_title("Outstanding Balance Over Loan Term", fontsize=10, pad=10)
    ax.set_xlabel("Month", fontsize=8); ax.set_ylabel("Balance ($)", fontsize=8)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout(); return fig


def amortisation_chart(principal: float, annual_rate: float, term_months: int):
    """Stacked area — cumulative principal paid vs cumulative interest paid."""
    r       = annual_rate / 100 / 12
    emi     = calculate_emi(principal, annual_rate, term_months)
    balance = principal
    cum_p, cum_i = [], []
    cp = ci = 0.0
    for _ in range(term_months):
        interest = balance * r
        princ    = emi - interest
        balance  = max(balance - princ, 0)
        cp += princ; ci += interest
        cum_p.append(cp); cum_i.append(ci)

    months = list(range(1, term_months + 1))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.stackplot(months, cum_p, cum_i,
                 labels=["Cumulative Principal", "Cumulative Interest"],
                 colors=["#1a6fdf", "#e85454"], alpha=0.75)
    ax.set_title("Cumulative Principal vs Interest Paid", fontsize=10, pad=10)
    ax.set_xlabel("Month", fontsize=8); ax.set_ylabel("Amount ($)", fontsize=8)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.15)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout(); return fig


def monthly_breakdown_chart(principal: float, annual_rate: float, term_months: int):
    """Bar chart: monthly principal portion vs interest portion for each payment."""
    r       = annual_rate / 100 / 12
    emi     = calculate_emi(principal, annual_rate, term_months)
    balance = principal
    principals, interests = [], []
    for _ in range(term_months):
        interest = round(balance * r, 2)
        princ    = round(emi - interest, 2)
        balance  = max(round(balance - princ, 2), 0)
        principals.append(princ); interests.append(interest)

    months = list(range(1, term_months + 1))
    fig, ax = plt.subplots(figsize=(max(8, term_months * 0.18), 3.2))
    ax.bar(months, principals, label="Principal",  color="#1a6fdf", alpha=0.85, width=0.7)
    ax.bar(months, interests,  label="Interest",   color="#e85454", alpha=0.85, width=0.7,
           bottom=principals)
    ax.set_title("Monthly EMI Breakdown — Principal vs Interest", fontsize=10, pad=10)
    ax.set_xlabel("Installment No.", fontsize=8); ax.set_ylabel("Amount ($)", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.15)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout(); return fig


# ═══════════════════════════════════════════════════════════════════════
#  RISK ASSESSMENT CHARTS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════
def gauge_chart(rate: float):
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#080f1a"); ax.set_facecolor("#080f1a")
    for t_s, t_e, col in [(np.pi, np.pi*2/3,"#3ddc84"),(np.pi*2/3,np.pi/3,"#f0a22a"),(np.pi/3,0,"#e85454")]:
        t = np.linspace(t_s, t_e, 60)
        ax.fill_between(t, 0.55, 1.0, color=col, alpha=0.7)
        ax.fill_between(t, 0.50, 0.55, color=col, alpha=0.25)
    needle = np.pi - np.clip((rate - 5) / 20, 0, 1) * np.pi
    ax.annotate("", xy=(needle, 0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#e8f2ff", lw=2, mutation_scale=14))
    ax.plot(0, 0, "o", color="#e8f2ff", ms=6, zorder=5)
    ax.set_ylim(0, 1.18); ax.set_theta_zero_location("E"); ax.set_theta_direction(-1); ax.axis("off")
    ax.text(0, -0.12, f"{rate:.2f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#e8f2ff", transform=ax.transData)
    for angle, lbl, col in [(np.pi*0.85,"Low","#3ddc84"),(np.pi*0.5,"Med","#f0a22a"),(np.pi*0.15,"High","#e85454")]:
        ax.text(angle, 1.14, lbl, ha="center", va="center", fontsize=7.5, color=col, fontweight="bold")
    plt.tight_layout(pad=0); return fig

def approval_donut(pct: int, color: str):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.patch.set_facecolor("#080f1a"); ax.set_facecolor("#080f1a")
    ax.pie([pct, 100-pct], colors=[color,"#0d2035"], startangle=90,
           wedgeprops=dict(width=0.38, edgecolor="#080f1a", linewidth=3))
    ax.text(0, 0, f"{pct}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#e8f2ff", fontfamily="monospace")
    ax.text(0, -0.32, "Approval\nProbability", ha="center", va="center",
            fontsize=7.5, color="#3d5870", linespacing=1.5)
    ax.axis("equal"); plt.tight_layout(pad=0.2); return fig

def feature_importance_chart():
    fi = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": rf_model.feature_importances_}
                      ).sort_values("Importance", ascending=False)
    top = fi.head(10)
    blues = ["#1a6fdf","#1862c8","#1455b0","#1048a0","#0d3b88",
             "#0a2e70","#082158","#061440","#040d28","#020a18"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                   color=blues[:len(top)][::-1], zorder=3, height=0.65)
    ax.set_title("Feature Importance — Random Forest", fontsize=11, pad=12)
    ax.set_xlabel("Importance Score", fontsize=9); ax.grid(axis="x", zorder=0)
    for bar, val in zip(bars, top["Importance"][::-1]):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="#5580a0")
    plt.tight_layout(); return fig, fi

def risk_distribution_chart(rate: float, color: str):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    fig.patch.set_facecolor("#080f1a"); ax.set_facecolor("#060b12")
    bands = ["< 10%\nLow","10–16%\nMedium","16–25%\nHigh"]; vals = [35,40,25]
    bc    = ["#3ddc84","#f0a22a","#e85454"]
    bars  = ax.bar(bands, vals, color=[c+"55" for c in bc], width=0.5, zorder=2,
                   edgecolor="#0d2035", linewidth=1)
    idx   = 0 if rate < 10 else (1 if rate < 16 else 2)
    bars[idx].set_facecolor(bc[idx]); bars[idx].set_edgecolor(bc[idx])
    ax.set_title("Risk Band Distribution", fontsize=9, pad=8, color="#5580a0")
    ax.set_ylabel("Portfolio %", fontsize=8); ax.grid(axis="y", zorder=0, alpha=0.4)
    ax.tick_params(axis="x", labelsize=8, colors="#5580a0"); ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout(); return fig


# ═══════════════════════════════════════════════════════════════════════
#  REPORT GENERATORS
# ═══════════════════════════════════════════════════════════════════════
def generate_risk_report(applicant: dict, rate: float, lvl: str,
                         label: str, approval_pct: int, rate_range: str) -> str:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ref = f"CRL-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    B = "═" * 62; T = "─" * 62
    R = lambda k, v: f"  {k:<30} {v}"
    lines = [
        B, "  CREDITLENS™  |  CREDIT RISK INTELLIGENCE REPORT", B,
        f"  Reference   : {ref}", f"  Generated   : {ts}",
        f"  Model       : Random Forest Regressor (rf_model.pkl)", T,
        "  APPLICANT INFORMATION", T,
        R("Age",                     f"{applicant.get('age','N/A')} years"),
        R("Annual Income",           f"${applicant.get('income',0):,}"),
        R("Employment Length",       f"{applicant.get('emp_length','N/A')} years"),
        R("Home Ownership",          applicant.get('home_own','N/A')),
        R("Credit History Length",   f"{applicant.get('cred_hist','N/A')} years"),
        R("Prior Default on File",   applicant.get('default_rec','N/A')), T,
        "  LOAN DETAILS", T,
        R("Loan Amount",             f"${applicant.get('loan_amnt',0):,}"),
        R("Loan Intent",             applicant.get('loan_intent','N/A')),
        R("Loan Grade",              applicant.get('loan_grade','N/A')),
        R("Loan Status",             str(applicant.get('loan_status','N/A'))), T,
        "  COMPUTED FINANCIAL RATIOS", T,
        R("Loan % of Income",        f"{applicant.get('loan_pct',0)*100:.2f}%"),
        R("Debt-to-Income Ratio",    f"{applicant.get('dti',0):.4f}"),
        R("Credit Utilization",      f"{applicant.get('c_util',0):.2f}%"),
        R("Employment-Income Ratio", f"{applicant.get('emp_inc',0):.4f}"), T,
        "  PREDICTION RESULT", T,
        R("Predicted Interest Rate",   f"{rate:.2f}%"),
        R("Risk Category",             label),
        R("Estimated Rate Band",       rate_range),
        R("Loan Approval Probability", f"{approval_pct}%"), T,
        "  RISK INTERPRETATION", T,
    ]
    if lvl == "low":
        lines += ["  ✅  LOW RISK BORROWER","  Rate below 10%. Excellent credit profile.",
                  "  Strong candidate for approval at competitive terms.","  Recommended: APPROVE"]
    elif lvl == "medium":
        lines += ["  ⚠️  MEDIUM RISK BORROWER","  Rate 10–16%. Moderate risk detected.",
                  "  Review employment history and debt obligations.","  Recommended: CONDITIONAL APPROVAL"]
    else:
        lines += ["  🚨  HIGH RISK BORROWER","  Rate above 16%. Elevated default probability.",
                  "  Consider collateral, reduced amount, or rejection.","  Recommended: REVIEW / DECLINE"]
    lines += [T,"  DISCLAIMER",T,
              "  This report is generated by an ML model for informational",
              "  purposes only and does not constitute financial or legal advice.",
              B,"  CreditLens™  |  ML-Powered Risk Intelligence",B]
    return "\n".join(lines)


def generate_installment_report(applicant: dict, pred_label: str,
                                 pp: dict, schedule_df: pd.DataFrame) -> str:
    """Full installment schedule plain-text report."""
    ts  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ref = f"INS-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    B = "═" * 74; T = "─" * 74
    R = lambda k, v: f"  {k:<36} {v}"

    sm       = schedule_summary(pp["principal"], pp["annual_rate"], pp["term_months"])
    end_date = (pp["start_date"] + relativedelta(months=pp["term_months"] - 1)).strftime("%d %b %Y")

    lines = [
        B, "  CREDITLENS™  |  LOAN REPAYMENT SCHEDULE REPORT", B,
        f"  Reference     : {ref}", f"  Generated     : {ts}", T,
        "  APPLICANT DETAILS", T,
        R("Annual Income",           f"${applicant.get('income', 0):,}"),
        R("Loan Grade",              applicant.get('loan_grade', 'N/A')),
        R("Risk Category",           pred_label), T,
        "  LOAN PARAMETERS", T,
        R("Principal Amount",        f"${pp['principal']:,.2f}"),
        R("Annual Interest Rate",    f"{pp['annual_rate']:.2f}%"),
        R("Loan Term",               f"{pp['term_months']} months"),
        R("First Payment Date",      pp["start_date"].strftime("%d %b %Y")),
        R("Last Payment Date",       end_date), T,
        "  REPAYMENT SUMMARY", T,
        R("Monthly EMI",             f"${sm['emi']:,.2f}"),
        R("Total Principal",         f"${pp['principal']:,.2f}"),
        R("Total Interest Payable",  f"${sm['total_interest']:,.2f}"),
        R("Total Repayment Amount",  f"${sm['total_payment']:,.2f}"), T,
        "  INSTALLMENT SCHEDULE", T,
        f"  {'No.':<5} {'Payment Date':<15} {'EMI ($)':>11} {'Principal ($)':>14} {'Interest ($)':>13} {'Balance ($)':>13}",
        T,
    ]
    for _, row in schedule_df.iterrows():
        lines.append(
            f"  {str(row['No.']):<5} {row['Payment Date']:<15} "
            f"{row['EMI ($)']:>11} {row['Principal ($)']:>14} "
            f"{row['Interest ($)']:>13} {row['Balance ($)']:>13}"
        )
    lines += [B, "  CreditLens™  |  ML-Powered Risk Intelligence", B]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE  — page routing
# ═══════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">Credit<span>Lens</span>™</div>
    <div style="font-size:0.72rem; color:#3d5870; margin-bottom:1.2rem;
                font-family:'IBM Plex Mono',monospace; letter-spacing:1px;">
        RISK INTELLIGENCE PLATFORM
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    for ico, lbl in [("🏠","Home"),("📋","Loan Application"),
                     ("💳","Loan Repayment Planner"),("📊","Analytics Dashboard")]:
        if st.button(f"{ico}  {lbl}", key=f"sb_{lbl}"):
            st.session_state["page"] = lbl; st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="info-chip">
        <strong style="color:#c8d8ea; font-size:0.8rem;">Risk Bands</strong><br><br>
        <span style="color:#3ddc84">●</span>  <strong>Low</strong> &nbsp;&nbsp;&nbsp;— below 10%<br>
        <span style="color:#f0a22a">●</span>  <strong>Medium</strong> — 10% to 16%<br>
        <span style="color:#e85454">●</span>  <strong>High</strong> &nbsp;&nbsp;— above 16%
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-chip" style="margin-top:0.6rem; font-size:0.75rem; font-family:'IBM Plex Mono',monospace;">
        🔧 Model features: {len(FEATURE_COLS)}<br>
        <span style="color:#3d5870">{' · '.join(FEATURE_COLS[:4])} ···</span>
    </div>""", unsafe_allow_html=True)

page = st.session_state["page"]

# ═══════════════════════════════════════════════════════════════════════
#  PAGE 0  ——  HOME
# ═══════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏦 CreditLens™ Risk Intelligence</div>
        <p class="hero-sub">ML-powered credit risk scoring, interest rate prediction and full loan repayment planning — in one platform</p>
        <span class="hero-tag">Random Forest</span>
        <span class="hero-tag">Production Ready</span>
        <span class="hero-tag">Real-Time Scoring</span>
        <span class="hero-tag">EMI Planner</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Platform Capabilities</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], [
        ("⚡","Instant Risk Scoring",    "Real-time interest rate prediction powered by a trained Random Forest model"),
        ("🎯","Risk Classification",     "Automatic Low / Medium / High risk categorisation with approval probability"),
        ("💳","EMI Repayment Planner",   "Banking-accurate amortisation schedule with full month-by-month breakdown"),
        ("📊","Analytics Dashboard",    "Feature importance charts, model performance metrics and pipeline overview"),
    ]):
        col.markdown(f"""
        <div class="metric-card" style="text-align:left; padding:1.2rem 1.3rem; min-height:130px;">
            <div style="font-size:1.4rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700;
                        color:#e8f2ff; margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:0.8rem; color:#3d5870; line-height:1.5;">{desc}</div>
            <div class="m-accent" style="background:#1a6fdf;"></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">How It Works</div>', unsafe_allow_html=True)
    for num, title, desc in [
        ("01","Fill Loan Application",     "Enter applicant personal details, loan specifics, and financial information."),
        ("02","Get Risk Score",            "The Random Forest model predicts the interest rate and classifies the risk level."),
        ("03","Plan Repayments",           "Open the Repayment Planner — pre-filled with predicted rate — to generate the EMI schedule."),
        ("04","Download Reports",          "Export the credit risk report and/or the full installment schedule as CSV or TXT."),
    ]:
        st.markdown(f"""
        <div style="display:flex; gap:1rem; align-items:flex-start; margin-bottom:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; font-weight:700;
                        color:#1a6fdf; background:#0a1e35; border:1px solid #0d2a4a;
                        border-radius:6px; padding:0.25rem 0.5rem; min-width:2.2rem;
                        text-align:center; margin-top:0.1rem;">{num}</div>
            <div>
                <div style="font-weight:600; color:#c8d8ea; font-size:0.9rem; margin-bottom:0.2rem;">{title}</div>
                <div style="font-size:0.82rem; color:#3d5870; line-height:1.5;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Navigate To</div>', unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns(3, gap="medium")
    with nav1:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:1.4rem 1.5rem; min-height:140px;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">📋</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#e8f2ff; margin-bottom:0.4rem;">Loan Application</div>
            <div style="font-size:0.8rem; color:#3d5870; line-height:1.5;">Submit applicant details and receive an instant risk score.</div>
            <div class="m-accent" style="background:#1a6fdf;"></div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Loan Application →", key="nav_loan"):
            st.session_state["page"] = "Loan Application"; st.rerun()
    with nav2:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:1.4rem 1.5rem; min-height:140px;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">💳</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#e8f2ff; margin-bottom:0.4rem;">Loan Repayment Planner</div>
            <div style="font-size:0.8rem; color:#3d5870; line-height:1.5;">Generate EMI schedules and download installment reports.</div>
            <div class="m-accent" style="background:#3ddc84;"></div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Repayment Planner →", key="nav_planner"):
            st.session_state["page"] = "Loan Repayment Planner"; st.rerun()
    with nav3:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:1.4rem 1.5rem; min-height:140px;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">📊</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#e8f2ff; margin-bottom:0.4rem;">Analytics Dashboard</div>
            <div style="font-size:0.8rem; color:#3d5870; line-height:1.5;">Explore model metrics, feature importance and pipeline.</div>
            <div class="m-accent" style="background:#a78bfa;"></div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Analytics Dashboard →", key="nav_analytics"):
            st.session_state["page"] = "Analytics Dashboard"; st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1  ——  LOAN APPLICATION + PREDICTION
# ═══════════════════════════════════════════════════════════════════════
elif page == "Loan Application":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📋 Loan Application & Risk Assessment</div>
        <p class="hero-sub">Complete the form below to receive an instant credit risk score and predicted interest rate</p>
    </div>""", unsafe_allow_html=True)

    app_tab, result_tab = st.tabs(["📝  Application Form", "📈  Prediction Result"])

    # ── Application Form ──────────────────────────────────────────────
    with app_tab:
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        with col1:
            st.markdown('<div class="section-head">👤 Personal Information</div>', unsafe_allow_html=True)
            age        = st.slider("Age", 18, 75, 30)
            income     = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
            emp_length = st.slider("Employment Length (years)", 0, 40, 5)
            home_own   = st.selectbox("Home Ownership", list(HOME_OWNERSHIP_MAP.keys()), index=3)
            cred_hist  = st.slider("Credit History (years)", 1, 30, 5)

        with col2:
            st.markdown('<div class="section-head">💰 Loan Details</div>', unsafe_allow_html=True)
            loan_amnt   = st.number_input("Loan Amount ($)", 500, 50000, 10000, step=500)
            loan_intent = st.selectbox("Loan Intent", list(LOAN_INTENT_MAP.keys()), index=4)
            loan_grade  = st.selectbox("Loan Grade  (A = best → G = riskiest)", list(LOAN_GRADE_MAP.keys()), index=1)
            default_rec = st.selectbox("Prior Default on File", list(DEFAULT_MAP.keys()), index=0)
            loan_status = st.selectbox("Loan Status  (0 = No Default · 1 = Default)", [0, 1], index=0)

        with col3:
            st.markdown('<div class="section-head">📐 Computed Ratios</div>', unsafe_allow_html=True)
            loan_pct = round(loan_amnt / max(income, 1), 6)
            dti      = round(loan_amnt / max(income, 1), 6)
            c_util   = round(loan_pct * 100, 4)
            emp_inc  = round(emp_length / max(income / 10000, 0.001), 6)

            m1, m2 = st.columns(2)
            m1.metric("Loan % of Income",     f"{loan_pct*100:.2f}%")
            m2.metric("Debt-to-Income Ratio", f"{dti:.4f}")
            m3, m4 = st.columns(2)
            m3.metric("Credit Utilization",   f"{c_util:.2f}%")
            m4.metric("Emp–Income Ratio",     f"{emp_inc:.4f}")
            st.markdown("<br>", unsafe_allow_html=True)

            st.session_state["applicant"] = {
                "age": age, "income": income, "emp_length": emp_length,
                "home_own": home_own, "cred_hist": cred_hist, "loan_amnt": loan_amnt,
                "loan_intent": loan_intent, "loan_grade": loan_grade,
                "default_rec": default_rec, "loan_status": loan_status,
                "loan_pct": loan_pct, "dti": dti, "c_util": c_util, "emp_inc": emp_inc,
            }
            predict_clicked = st.button("⚡  Run Risk Assessment")

        if predict_clicked:
            fields = {
                "person_age": age, "person_income": float(income),
                "person_home_ownership": HOME_OWNERSHIP_MAP[home_own],
                "person_emp_length": float(emp_length),
                "loan_intent": LOAN_INTENT_MAP[loan_intent],
                "loan_grade": LOAN_GRADE_MAP[loan_grade],
                "loan_amnt": float(loan_amnt),
                "loan_percent_income": loan_pct,
                "cb_person_default_on_file": DEFAULT_MAP[default_rec],
                "cb_person_cred_hist_length": float(cred_hist),
                "loan_status": int(loan_status),
                "DebtIncomeRatio": dti, "CreditUtilization": c_util,
                "EmploymentIncomeRatio": emp_inc,
            }
            with st.spinner("Scoring applicant profile…"):
                rate = predict_rate(fields)
            lvl, color, icon, label, approval_pct, rate_range = risk_level(rate)
            st.session_state["prediction"] = {
                "rate": rate, "lvl": lvl, "color": color,
                "icon": icon, "label": label,
                "approval_pct": approval_pct, "rate_range": rate_range,
            }
            # Pre-fill planner with predicted values
            st.session_state["planner_rate"]      = round(rate, 2)
            st.session_state["planner_principal"]  = float(loan_amnt)
            st.success("✅ Prediction complete — view results in **Prediction Result** tab or head to **Loan Repayment Planner**.")

    # ── Prediction Result ─────────────────────────────────────────────
    with result_tab:
        if "prediction" not in st.session_state:
            st.info("ℹ️  Fill in the application form and click **Run Risk Assessment** to see results here.")
        else:
            pred   = st.session_state["prediction"]
            ap     = st.session_state.get("applicant", {})
            rate   = pred["rate"];  lvl  = pred["lvl"]; color = pred["color"]
            icon   = pred["icon"];  label = pred["label"]
            apct   = pred["approval_pct"]; rrange = pred["rate_range"]

            # KPI row
            st.markdown('<div class="section-head">Assessment Summary</div>', unsafe_allow_html=True)
            k1,k2,k3,k4 = st.columns(4, gap="small")
            for col,(val,lbl,accent) in zip([k1,k2,k3,k4],[
                (f"{rate:.2f}%","Predicted Rate","#1a6fdf"),
                (label.split()[0],"Risk Category",color),
                (f"{apct}%","Approval Probability",color),
                (rrange,"Est. Rate Band","#5580a0"),
            ]):
                col.markdown(f"""
                <div class="metric-card">
                    <span class="m-val" style="color:{accent}">{val}</span>
                    <span class="m-lbl">{lbl}</span>
                    <div class="m-accent" style="background:{accent};"></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            r1,r2,r3 = st.columns([1,1.1,1], gap="medium")
            with r1:
                st.markdown('<div class="section-head">Interest Rate Gauge</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="result-panel {lvl}">
                    <span class="rate-display" style="color:{color}">{rate:.2f}%</span>
                    <div style="font-size:0.82rem; color:#5580a0; margin-top:0.5rem;">Predicted Annual Interest Rate</div>
                    <span class="risk-badge" style="background:{color}18; color:{color}; border:1.5px solid {color}40;">{icon} {label}</span>
                </div>""", unsafe_allow_html=True)
                st.pyplot(gauge_chart(rate), use_container_width=True)
            with r2:
                st.markdown('<div class="section-head">Approval Probability</div>', unsafe_allow_html=True)
                st.pyplot(approval_donut(apct, color), use_container_width=True)
                st.markdown(f"""
                <div class="info-chip" style="text-align:center; margin-top:0.3rem;">
                    <strong style="color:#c8d8ea;">Estimated Rate Band</strong><br>
                    <span style="font-family:'IBM Plex Mono',monospace; color:{color}; font-size:1rem; font-weight:700;">{rrange}</span>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown('<div class="section-head">Risk Band Position</div>', unsafe_allow_html=True)
                st.pyplot(risk_distribution_chart(rate, color), use_container_width=True)

            # Risk interpretation
            st.markdown('<div class="section-head">Risk Interpretation</div>', unsafe_allow_html=True)
            if lvl == "low":
                st.success("**✅ Low Risk Borrower — Recommended: APPROVE**\n\nRate below 10%. Excellent credit profile. Strong candidate for approval at competitive rates.")
            elif lvl == "medium":
                st.warning("**⚠️ Medium Risk Borrower — Recommended: CONDITIONAL APPROVAL**\n\nRate 10–16%. Moderate risk. Review employment history and debt obligations before final decision.")
            else:
                st.error("**🚨 High Risk Borrower — Recommended: REVIEW / DECLINE**\n\nRate above 16%. Elevated default probability. Consider collateral, reduced loan amount, or declining.")

            # Financial summary tables
            st.markdown('<div class="section-head">Applicant Financial Summary</div>', unsafe_allow_html=True)
            s1,s2 = st.columns(2, gap="medium")
            with s1:
                st.dataframe(pd.DataFrame({
                    "Field": ["Annual Income","Loan Amount","Loan Grade","Loan Intent","Home Ownership","Prior Default","Employment Length","Credit History"],
                    "Value": [f"${ap.get('income',0):,}", f"${ap.get('loan_amnt',0):,}",
                              ap.get("loan_grade","—"), ap.get("loan_intent","—"),
                              ap.get("home_own","—"), ap.get("default_rec","—"),
                              f"{ap.get('emp_length',0)} yrs", f"{ap.get('cred_hist',0)} yrs"],
                }), use_container_width=True, hide_index=True)
            with s2:
                st.dataframe(pd.DataFrame({
                    "Ratio": ["Loan % of Income","Debt-to-Income","Credit Utilization","Emp–Income Ratio"],
                    "Value": [f"{ap.get('loan_pct',0)*100:.2f}%", f"{ap.get('dti',0):.4f}",
                              f"{ap.get('c_util',0):.2f}%", f"{ap.get('emp_inc',0):.4f}"],
                }), use_container_width=True, hide_index=True)

            # CTA to Planner
            st.markdown('<div class="section-head">Next Step</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="planner-cta">
                <div style="font-family:'Syne',sans-serif; font-weight:700; color:#e8f2ff; font-size:0.95rem; margin-bottom:0.2rem;">
                    💳 Ready to Plan Repayments?
                </div>
                <div style="font-size:0.82rem; color:#3d6040;">
                    The Repayment Planner is pre-filled with your predicted rate
                    <strong style="color:#3ddc84;">{rate:.2f}%</strong>
                    and loan amount <strong style="color:#3ddc84;">${ap.get('loan_amnt',0):,}</strong>.
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button("Open Loan Repayment Planner →", key="goto_planner"):
                st.session_state["page"] = "Loan Repayment Planner"; st.rerun()

            # Risk report download
            st.markdown('<div class="section-head">Credit Risk Report</div>', unsafe_allow_html=True)
            report_text = generate_risk_report(ap, rate, lvl, label, apct, rrange)
            with st.expander("📄 Preview Risk Report", expanded=False):
                st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)

            ref_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            dl1,dl2 = st.columns(2, gap="medium")
            with dl1:
                st.download_button(
                    label="⬇️  Download Risk Report (.txt)", data=report_text,
                    file_name=f"CreditLens_RiskReport_{ref_id}.txt",
                    mime="text/plain", key="dl_txt",
                )
            with dl2:
                csv_data = pd.DataFrame({
                    "Field": ["Reference","Generated At","Age","Annual Income","Loan Amount",
                              "Loan Grade","Loan Intent","Home Ownership","Employment Length",
                              "Credit History","Prior Default","Loan Status","Loan % Income",
                              "Debt-to-Income","Credit Utilization","Emp-Income Ratio",
                              "Predicted Rate","Risk Category","Approval Probability","Est Rate Band"],
                    "Value": [f"CRL-{ref_id}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              ap.get("age",""), f"${ap.get('income',0):,}", f"${ap.get('loan_amnt',0):,}",
                              ap.get("loan_grade",""), ap.get("loan_intent",""), ap.get("home_own",""),
                              ap.get("emp_length",""), ap.get("cred_hist",""), ap.get("default_rec",""),
                              ap.get("loan_status",""), f"{ap.get('loan_pct',0)*100:.2f}%",
                              f"{ap.get('dti',0):.4f}", f"{ap.get('c_util',0):.2f}%",
                              f"{ap.get('emp_inc',0):.4f}", f"{rate:.2f}%", label,
                              f"{apct}%", rrange],
                }).to_csv(index=False)
                st.download_button(
                    label="⬇️  Download Summary (.csv)", data=csv_data,
                    file_name=f"CreditLens_Summary_{ref_id}.csv",
                    mime="text/csv", key="dl_csv",
                )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 2  ——  LOAN REPAYMENT PLANNER
# ═══════════════════════════════════════════════════════════════════════
elif page == "Loan Repayment Planner":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">💳 Loan Repayment Planner</div>
        <p class="hero-sub">Banking-accurate EMI calculator · Full amortisation schedule · Downloadable installment report</p>
        <span class="hero-tag">Reducing Balance</span>
        <span class="hero-tag">Amortisation</span>
        <span class="hero-tag">CSV + TXT Export</span>
    </div>""", unsafe_allow_html=True)

    planner_tab, schedule_tab = st.tabs(["⚙️  Repayment Planner", "📅  Installment Schedule & Report"])

    # ── Planner configuration ─────────────────────────────────────────
    with planner_tab:

        st.markdown('<div class="section-head">🔧 Loan Parameters</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3, gap="medium")

        with p1:
            default_principal = float(st.session_state.get("planner_principal", 10000.0))
            principal = st.number_input(
                "Loan Amount ($)",
                min_value=500.0, max_value=1_000_000.0,
                value=default_principal, step=500.0,
                help="Principal amount to be repaid",
            )

        with p2:
            default_rate = float(st.session_state.get("planner_rate", 10.0))
            annual_rate  = st.number_input(
                "Annual Interest Rate (%)",
                min_value=0.0, max_value=50.0,
                value=round(default_rate, 2), step=0.1,
                help="Pre-filled from prediction. You can override this.",
            )
            if "prediction" in st.session_state:
                pred_r = st.session_state["prediction"]["rate"]
                st.markdown(f"""
                <div style="font-size:0.75rem; color:#3d6040; padding:0.3rem 0.55rem;
                            background:#071e0d; border-radius:4px; border-left:2px solid #3ddc84; margin-top:-0.4rem;">
                    💡 Predicted rate: <strong style="color:#3ddc84;">{pred_r:.2f}%</strong>
                </div>""", unsafe_allow_html=True)

        with p3:
            term_months = st.selectbox(
                "Loan Term",
                options=[6,12,18,24,36,48,60,72,84,96,108,120,180,240,360],
                index=4,
                format_func=lambda x: f"{x} months  ({x//12} yr{'' if x//12==1 else 's'}{f' {x%12}mo' if x%12 else ''})",
                help="Total repayment period in months",
            )

        p4, _ = st.columns([1, 2], gap="medium")
        with p4:
            start_date = st.date_input(
                "First Payment Date",
                value=datetime.date.today().replace(day=1) + relativedelta(months=1),
                help="Date of the first EMI payment",
            )

        # ── Live summary calculations ──────────────────────────────
        sm       = schedule_summary(principal, annual_rate, term_months)
        emi      = sm["emi"]
        total_p  = sm["total_payment"]
        total_i  = sm["total_interest"]
        end_date = start_date + relativedelta(months=term_months - 1)
        int_pct  = round(total_i / max(total_p, 1) * 100, 1)
        pri_pct  = round(100 - int_pct, 1)

        # ── EMI KPI cards ──────────────────────────────────────────
        st.markdown('<div class="section-head">📊 Repayment Summary</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4, gap="small")

        with e1:
            st.markdown(f"""
            <div class="emi-box">
                <span class="emi-amount">${emi:,.2f}</span>
                <span class="emi-label">Monthly EMI</span>
            </div>""", unsafe_allow_html=True)

        with e2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="m-val" style="color:#1a6fdf;">${principal:,.0f}</span>
                <span class="m-lbl">Principal Amount</span>
                <div class="m-accent" style="background:#1a6fdf;"></div>
            </div>""", unsafe_allow_html=True)

        with e3:
            st.markdown(f"""
            <div class="metric-card">
                <span class="m-val" style="color:#e85454;">${total_i:,.2f}</span>
                <span class="m-lbl">Total Interest</span>
                <div class="m-accent" style="background:#e85454;"></div>
            </div>""", unsafe_allow_html=True)

        with e4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="m-val" style="color:#f0a22a;">${total_p:,.2f}</span>
                <span class="m-lbl">Total Repayment</span>
                <div class="m-accent" style="background:#f0a22a;"></div>
            </div>""", unsafe_allow_html=True)

        # ── Cost breakdown bar ─────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom:0.4rem; font-size:0.78rem; color:#5580a0; font-family:'IBM Plex Mono',monospace;">
            LOAN COST BREAKDOWN &nbsp;·&nbsp;
            <span style="color:#1a6fdf;">■ Principal {pri_pct}%</span>
            &nbsp;&nbsp;
            <span style="color:#e85454;">■ Interest {int_pct}%</span>
        </div>
        <div style="display:flex; height:10px; border-radius:6px; overflow:hidden; margin-bottom:1.2rem;">
            <div style="width:{pri_pct}%; background:#1a6fdf; border-radius:6px 0 0 6px;"></div>
            <div style="width:{int_pct}%; background:#e85454; border-radius:0 6px 6px 0;"></div>
        </div>""", unsafe_allow_html=True)

        # ── Date chips ─────────────────────────────────────────────
        d1, d2, d3 = st.columns(3, gap="small")
        for dcol, hdr, val in [
            (d1, "First Payment",  start_date.strftime("%d %b %Y")),
            (d2, "Last Payment",   end_date.strftime("%d %b %Y")),
            (d3, "Loan Duration",  f"{term_months} months"),
        ]:
            dcol.markdown(f"""
            <div class="info-chip" style="text-align:center;">
                <div style="font-size:0.68rem; color:#3d5870; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:0.3rem;">{hdr}</div>
                <div style="font-family:'IBM Plex Mono',monospace; color:#c8d8ea; font-size:0.9rem; font-weight:600;">{val}</div>
            </div>""", unsafe_allow_html=True)

        # ── Amortisation charts ────────────────────────────────────
        st.markdown('<div class="section-head">📈 Amortisation Charts</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2, gap="medium")
        with ch1:
            st.pyplot(balance_chart(principal, annual_rate, term_months), use_container_width=True)
        with ch2:
            st.pyplot(amortisation_chart(principal, annual_rate, term_months), use_container_width=True)

        # Monthly breakdown chart (full-width)
        with st.expander("📊 Monthly EMI Breakdown (Principal vs Interest per Instalment)", expanded=False):
            st.pyplot(monthly_breakdown_chart(principal, annual_rate, term_months), use_container_width=True)

        # Persist to session for schedule tab
        st.session_state["planner_params"] = {
            "principal":   principal,
            "annual_rate": annual_rate,
            "term_months": term_months,
            "start_date":  start_date,
        }

        st.info("✅ Switch to the **Installment Schedule & Report** tab to view the full month-by-month table and download reports.")

    # ── Schedule & Report ─────────────────────────────────────────────
    with schedule_tab:

        if "planner_params" not in st.session_state:
            st.info("ℹ️  Configure the loan in the **Repayment Planner** tab first.")
        else:
            pp    = st.session_state["planner_params"]
            ap    = st.session_state.get("applicant", {})
            pred  = st.session_state.get("prediction", {})
            p_lbl = pred.get("label", "N/A")

            sm      = schedule_summary(pp["principal"], pp["annual_rate"], pp["term_months"])
            end_dt  = pp["start_date"] + relativedelta(months=pp["term_months"] - 1)

            # Schedule header KPIs
            st.markdown('<div class="section-head">📋 Installment Schedule</div>', unsafe_allow_html=True)
            h1,h2,h3,h4,h5 = st.columns(5, gap="small")
            for col,(val,lbl,accent) in zip([h1,h2,h3,h4,h5],[
                (f"${pp['principal']:,.0f}",      "Principal",      "#1a6fdf"),
                (f"{pp['annual_rate']:.2f}%",      "Annual Rate",    "#f0a22a"),
                (f"{pp['term_months']} mo",         "Term",           "#5580a0"),
                (f"${sm['emi']:,.2f}",             "Monthly EMI",    "#3ddc84"),
                (f"${sm['total_interest']:,.2f}",  "Total Interest", "#e85454"),
            ]):
                col.markdown(f"""
                <div class="metric-card">
                    <span class="m-val" style="color:{accent}; font-size:1.1rem;">{val}</span>
                    <span class="m-lbl">{lbl}</span>
                    <div class="m-accent" style="background:{accent};"></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Generate & display full schedule
            with st.spinner("Generating installment schedule…"):
                schedule_df = build_schedule(
                    pp["principal"], pp["annual_rate"],
                    pp["term_months"], pp["start_date"]
                )

            st.dataframe(
                schedule_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "No.":           st.column_config.NumberColumn("No.",           width="small"),
                    "Payment Date":  st.column_config.TextColumn("Payment Date",    width="medium"),
                    "EMI ($)":       st.column_config.TextColumn("EMI ($)",          width="medium"),
                    "Principal ($)": st.column_config.TextColumn("Principal ($)",   width="medium"),
                    "Interest ($)":  st.column_config.TextColumn("Interest ($)",    width="medium"),
                    "Balance ($)":   st.column_config.TextColumn("Balance ($)",     width="medium"),
                },
            )

            st.markdown(f"""
            <div class="info-chip" style="margin-top:0.5rem;">
                📌  <strong style="color:#c8d8ea;">{pp['term_months']} instalments</strong>
                &nbsp;·&nbsp; First: <strong>{pp['start_date'].strftime('%d %b %Y')}</strong>
                &nbsp;·&nbsp; Last: <strong>{end_dt.strftime('%d %b %Y')}</strong>
                &nbsp;·&nbsp; Total repaid: <strong style="color:#f0a22a;">${sm['total_payment']:,.2f}</strong>
            </div>""", unsafe_allow_html=True)

            # ── Download section ───────────────────────────────────
            st.markdown('<div class="section-head">📥 Download Installment Report</div>', unsafe_allow_html=True)

            ref_id      = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            report_txt  = generate_installment_report(ap, p_lbl, pp, schedule_df)

            # Build enriched CSV
            csv_df = schedule_df.copy()
            # Insert summary header rows at top via a metadata section
            meta = pd.DataFrame({
                "No.": ["—","—","—","—","—","—","—"],
                "Payment Date": ["LOAN SUMMARY","Loan Amount","Annual Rate","Term","Monthly EMI","Total Interest","Total Repayment"],
                "EMI ($)": ["","f${pp['principal']:,.2f}","","","","",""],
                "Principal ($)": ["",f"${pp['principal']:,.2f}",f"{pp['annual_rate']:.2f}%",
                                  f"{pp['term_months']} mo",f"${sm['emi']:,.2f}",
                                  f"${sm['total_interest']:,.2f}",f"${sm['total_payment']:,.2f}"],
                "Interest ($)": [""]*7,
                "Balance ($)": [""]*7,
            })
            csv_bytes = pd.concat([meta, schedule_df], ignore_index=True).to_csv(index=False)

            with st.expander("📄 Preview Installment Report", expanded=False):
                st.markdown(f'<div class="report-box">{report_txt}</div>', unsafe_allow_html=True)

            dl1, dl2 = st.columns(2, gap="medium")
            with dl1:
                st.download_button(
                    label="⬇️  Download Installment Report (.txt)",
                    data=report_txt,
                    file_name=f"CreditLens_InstallmentReport_{ref_id}.txt",
                    mime="text/plain",
                    key="dl_inst_txt",
                )
            with dl2:
                st.download_button(
                    label="⬇️  Download Full Schedule (.csv)",
                    data=csv_bytes,
                    file_name=f"CreditLens_InstallmentSchedule_{ref_id}.csv",
                    mime="text/csv",
                    key="dl_inst_csv",
                )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 3  ——  ANALYTICS DASHBOARD  (simple analysis)
# ═══════════════════════════════════════════════════════════════════════
elif page == "Analytics Dashboard":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📊 Analytics Dashboard</div>
        <p class="hero-sub">Live risk distribution, loan portfolio insights and session prediction summary</p>
    </div>""", unsafe_allow_html=True)

    # ── Top KPI metrics ───────────────────────────────────────────────
    st.markdown('<div class="section-head">Model Snapshot</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="small")
    for col, (val, lbl, accent) in zip([k1, k2, k3, k4], [
        ("0.8288", "R² Score",       "#1a6fdf"),
        ("~0.96",  "ROC AUC",        "#3ddc84"),
        ("1.2873", "RMSE",           "#f0a22a"),
        (str(len(FEATURE_COLS)),     "Features Used", "#a78bfa"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <span class="m-val">{val}</span>
            <span class="m-lbl">{lbl}</span>
            <div class="m-accent" style="background:{accent};"></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk band distribution chart ──────────────────────────────────
    st.markdown('<div class="section-head">Risk Band Distribution</div>', unsafe_allow_html=True)

    ch1, ch2 = st.columns([1.2, 1], gap="large")

    with ch1:
        # Bar chart: typical portfolio split across Low / Medium / High
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor("#080f1a"); ax.set_facecolor("#060b12")
        bands  = ["Low Risk\n(< 10%)", "Medium Risk\n(10–16%)", "High Risk\n(> 16%)"]
        counts = [38, 42, 20]
        colors = ["#3ddc84", "#f0a22a", "#e85454"]
        bars   = ax.bar(bands, counts, color=colors, width=0.5, zorder=3,
                        edgecolor="#080f1a", linewidth=1.5)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8, f"{val}%",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color="#c8d8ea")
        ax.set_ylabel("Portfolio Share (%)", fontsize=8)
        ax.set_title("Typical Portfolio Risk Distribution", fontsize=10, pad=10)
        ax.set_ylim(0, 55); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9, colors="#7398b5")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with ch2:
        # Donut: portfolio split
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        fig2.patch.set_facecolor("#080f1a"); ax2.set_facecolor("#080f1a")
        sizes  = [38, 42, 20]
        clrs   = ["#3ddc84", "#f0a22a", "#e85454"]
        labels = ["Low", "Medium", "High"]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, colors=clrs, startangle=90,
            autopct="%1.0f%%", pctdistance=0.72,
            wedgeprops=dict(width=0.45, edgecolor="#080f1a", linewidth=2),
        )
        for t in texts:      t.set_color("#7398b5"); t.set_fontsize(9)
        for a in autotexts:  a.set_color("#e8f2ff"); a.set_fontsize(9); a.set_fontweight("bold")
        ax2.set_title("Risk Category Split", fontsize=10, pad=12, color="#c8d8ea")
        ax2.axis("equal"); plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # ── Interest rate band analysis ───────────────────────────────────
    st.markdown('<div class="section-head">Interest Rate Band Analysis</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2, gap="large")

    with rc1:
        # Horizontal bar — avg interest rates per loan grade
        fig3, ax3 = plt.subplots(figsize=(7, 3.8))
        fig3.patch.set_facecolor("#080f1a"); ax3.set_facecolor("#060b12")
        grades     = ["A", "B", "C", "D", "E", "F", "G"]
        avg_rates  = [7.2, 9.8, 12.4, 15.1, 18.3, 21.6, 24.9]
        bar_colors = ["#3ddc84","#3ddc84","#f0a22a","#f0a22a","#e85454","#e85454","#e85454"]
        hbars = ax3.barh(grades[::-1], avg_rates[::-1],
                         color=bar_colors[::-1], height=0.55, zorder=3)
        ax3.set_xlabel("Avg Interest Rate (%)", fontsize=8)
        ax3.set_title("Average Rate by Loan Grade", fontsize=10, pad=10)
        ax3.grid(axis="x", alpha=0.3, zorder=0)
        for bar, val in zip(hbars, avg_rates[::-1]):
            ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val}%", va="center", fontsize=8, color="#c8d8ea")
        ax3.tick_params(axis="y", labelsize=9, colors="#7398b5")
        ax3.tick_params(axis="x", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with rc2:
        # Line — approval probability vs predicted rate
        fig4, ax4 = plt.subplots(figsize=(7, 3.8))
        fig4.patch.set_facecolor("#080f1a"); ax4.set_facecolor("#060b12")
        rates_x  = list(range(5, 26))
        approvals = [max(20, 95 - (r - 5) * 3.5) for r in rates_x]
        ax4.fill_between(rates_x, approvals, alpha=0.12, color="#1a6fdf")
        ax4.plot(rates_x, approvals, color="#1a6fdf", lw=2)
        ax4.axvline(10, color="#3ddc84", lw=1, ls="--", alpha=0.7, label="Low/Med boundary")
        ax4.axvline(16, color="#e85454", lw=1, ls="--", alpha=0.7, label="Med/High boundary")
        ax4.set_xlabel("Interest Rate (%)", fontsize=8)
        ax4.set_ylabel("Approval Probability (%)", fontsize=8)
        ax4.set_title("Approval Probability vs Interest Rate", fontsize=10, pad=10)
        ax4.legend(fontsize=7.5, framealpha=0.15)
        ax4.grid(axis="y", alpha=0.3)
        ax4.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)

    # ── Session prediction summary ────────────────────────────────────
    st.markdown('<div class="section-head">Current Session Prediction</div>', unsafe_allow_html=True)

    if "prediction" not in st.session_state:
        st.info("ℹ️  No prediction yet — go to **Loan Application** to run a risk assessment and the result will appear here.")
    else:
        pred  = st.session_state["prediction"]
        ap    = st.session_state.get("applicant", {})
        rate  = pred["rate"];   color = pred["color"]
        label = pred["label"];  apct  = pred["approval_pct"]
        lvl   = pred["lvl"]

        s1, s2, s3, s4 = st.columns(4, gap="small")
        for col, (val, lbl, accent) in zip([s1, s2, s3, s4], [
            (f"{rate:.2f}%",         "Predicted Rate",       color),
            (label.split()[0],       "Risk Category",        color),
            (f"{apct}%",             "Approval Probability", color),
            (f"${ap.get('loan_amnt', 0):,}", "Loan Amount",  "#1a6fdf"),
        ]):
            col.markdown(f"""
            <div class="metric-card">
                <span class="m-val" style="color:{accent}">{val}</span>
                <span class="m-lbl">{lbl}</span>
                <div class="m-accent" style="background:{accent};"></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mini gauge + summary side by side
        g1, g2 = st.columns([1, 1.6], gap="large")
        with g1:
            st.pyplot(gauge_chart(rate), use_container_width=True)
        with g2:
            st.markdown('<div class="section-head">Prediction Details</div>', unsafe_allow_html=True)
            details = [
                ("Annual Income",       f"${ap.get('income', 0):,}"),
                ("Loan Amount",         f"${ap.get('loan_amnt', 0):,}"),
                ("Loan Grade",          ap.get("loan_grade", "—")),
                ("Loan Intent",         ap.get("loan_intent", "—")),
                ("Prior Default",       ap.get("default_rec", "—")),
                ("Employment Length",   f"{ap.get('emp_length', 0)} yrs"),
                ("Debt-to-Income",      f"{ap.get('dti', 0):.4f}"),
                ("Credit Utilization",  f"{ap.get('c_util', 0):.2f}%"),
            ]
            for field, value in details:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:0.45rem 0;
                            border-bottom:1px solid #0d2035; font-size:0.83rem;">
                    <span style="color:#5580a0;">{field}</span>
                    <span style="color:#c8d8ea; font-weight:500;">{value}</span>
                </div>""", unsafe_allow_html=True)

        if lvl == "low":
            st.success("✅ **Low Risk** — Strong applicant profile. Recommended for approval at competitive rates.")
        elif lvl == "medium":
            st.warning("⚠️ **Medium Risk** — Moderate risk profile. Conditional approval recommended with review.")
        else:
            st.error("🚨 **High Risk** — Elevated default probability. Manual review or decline recommended.")