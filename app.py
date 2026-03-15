import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — sleek dark finance UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #080c14;
    color: #cdd6f4;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0e1420 !important;
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #cdd6f4 !important; }

/* ── Header banner ── */
.hero {
    background: linear-gradient(120deg, #0e1f3d 0%, #091628 60%, #050d1a 100%);
    border: 1px solid #1e3a5f;
    border-left: 5px solid #4f9cf9;
    border-radius: 16px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(79,156,249,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #f0f4ff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p { color: #6b8cba; font-size: 0.95rem; margin: 0; }
.hero .badge {
    display: inline-block;
    background: rgba(79,156,249,0.12);
    border: 1px solid rgba(79,156,249,0.3);
    color: #4f9cf9;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.75rem;
    margin-right: 0.4rem;
    letter-spacing: 0.5px;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: #0e1420;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover { border-color: #4f9cf9; transform: translateY(-2px); }
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}
.stat-card.blue::after  { background: #4f9cf9; }
.stat-card.green::after { background: #3ddc84; }
.stat-card.amber::after { background: #f5a623; }
.stat-card.purple::after{ background: #a78bfa; }
.stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #f0f4ff;
    display: block;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.4rem;
    display: block;
}

/* ── Section label ── */
.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #4f9cf9;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.8rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* ── Input card ── */
.input-card {
    background: #0e1420;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.4rem 1.6rem 1.8rem;
}

/* ── Result box ── */
.result-wrap {
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    margin: 1.2rem 0;
    position: relative;
    overflow: hidden;
}
.result-wrap.low    { background: linear-gradient(135deg,#091e14,#0d2a1a); border: 2px solid #3ddc84; }
.result-wrap.medium { background: linear-gradient(135deg,#1e1509,#2b1e07); border: 2px solid #f5a623; }
.result-wrap.high   { background: linear-gradient(135deg,#1e0909,#2b0f0f); border: 2px solid #ff5c5c; }
.result-rate {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    display: block;
    line-height: 1;
}
.result-sub { font-size: 0.9rem; color: #6b8cba; margin-top: 0.5rem; }
.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 1.1rem;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 0.9rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Info chip ── */
.chip {
    background: #0e1e35;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: #8aadcc;
    line-height: 1.5;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a4fa8, #2d6ef0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(45,110,240,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2d6ef0, #5b9bf9) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(45,110,240,0.4) !important;
}

/* ── Sidebar nav ── */
.stRadio > div { gap: 0.3rem; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.95rem; }

/* ── Slider + select ── */
.stSlider > div > div > div > div { background: #4f9cf9 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0e1420;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a6080 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: #1a4fa8 !important;
    color: #fff !important;
}

/* ── Table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Hide branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0e1420",
    "axes.facecolor":    "#080c14",
    "axes.edgecolor":    "#1e2d45",
    "axes.labelcolor":   "#6b8cba",
    "axes.titlecolor":   "#cdd6f4",
    "xtick.color":       "#4a6080",
    "ytick.color":       "#4a6080",
    "grid.color":        "#131c2e",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "text.color":        "#cdd6f4",
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────
#  LABEL ENCODINGS  — sklearn alphabetical order
# ─────────────────────────────────────────────
LOAN_GRADE_MAP     = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
HOME_OWNERSHIP_MAP = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
LOAN_INTENT_MAP    = {
    "DEBTCONSOLIDATION": 0, "EDUCATION": 1,
    "HOMEIMPROVEMENT":   2, "MEDICAL":   3,
    "PERSONAL":          4, "VENTURE":   5,
}
DEFAULT_MAP = {"N": 0, "Y": 1}

# ─────────────────────────────────────────────
#  LOAD MODEL  — then read feature names from it
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "rf_model.pkl"
    if not os.path.exists(path):
        st.error("❌ **rf_model.pkl not found.** Place it in the same folder as app.py")
        st.stop()
    return joblib.load(path)

rf_model = load_model()

# Dynamically get the exact feature list the model was trained on
FEATURE_COLS = list(rf_model.feature_names_in_)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def build_input_row(fields: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with exactly the columns the model expects."""
    row = pd.DataFrame([fields])
    # Add any missing columns the model needs (e.g. loan_status default = 0)
    for col in FEATURE_COLS:
        if col not in row.columns:
            row[col] = 0
    return row[FEATURE_COLS]

def predict_rate(fields: dict) -> float:
    return float(rf_model.predict(build_input_row(fields))[0])

def risk_level(rate: float):
    if rate < 10:  return "low",    "#3ddc84", "🟢", "LOW RISK"
    if rate < 16:  return "medium", "#f5a623", "🟡", "MEDIUM RISK"
    return              "high",   "#ff5c5c", "🔴", "HIGH RISK"

# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────
def gauge_chart(rate: float):
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0e1420")
    ax.set_facecolor("#0e1420")
    for t_s, t_e, col in [
        (np.pi, np.pi*2/3, "#3ddc84"),
        (np.pi*2/3, np.pi/3, "#f5a623"),
        (np.pi/3, 0, "#ff5c5c"),
    ]:
        t = np.linspace(t_s, t_e, 60)
        ax.fill_between(t, 0.55, 1.0, color=col, alpha=0.75)
        ax.fill_between(t, 0.5, 0.55, color=col, alpha=0.3)

    needle = np.pi - np.clip((rate - 5) / 20, 0, 1) * np.pi
    ax.annotate("", xy=(needle, 0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#f0f4ff", lw=2,
                                mutation_scale=12))
    ax.plot(0, 0, "o", color="#f0f4ff", ms=6, zorder=5)

    ax.set_ylim(0, 1.15)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.axis("off")
    ax.text(0, -0.1, f"{rate:.2f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#f0f4ff", transform=ax.transData)
    for angle, label, col in [
        (np.pi*0.85,  "Low",  "#3ddc84"),
        (np.pi*0.5,   "Med",  "#f5a623"),
        (np.pi*0.15,  "High", "#ff5c5c"),
    ]:
        ax.text(angle, 1.12, label, ha="center", va="center", fontsize=7.5,
                color=col, fontweight="bold")
    plt.tight_layout(pad=0)
    return fig

def feature_importance_chart():
    fi = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    top = fi.head(10)
    grad = ["#4f9cf9", "#3d87e0", "#2d72c8", "#1e5da0",
            "#1a4fa8", "#163f85", "#122f63", "#0e2040",
            "#0a1628", "#070e1a"]
    colors = grad[:len(top)]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                   color=colors[::-1], zorder=3, height=0.65)
    ax.set_title("Feature Importance — Random Forest", fontsize=11, pad=12)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.grid(axis="x", zorder=0)
    for bar, val in zip(bars, top["Importance"][::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="#6b8cba")
    plt.tight_layout()
    return fig, fi

def model_comparison_chart():
    models = ["Linear\nRegression", "Decision\nTree", "Random\nForest"]
    r2s    = [0.7976, 0.8237, 0.8288]
    rmses  = [1.3998, 1.3063, 1.2873]
    maes   = [1.0741, 0.9539, 0.9489]
    colors = ["#1e5da0", "#2d72c8", "#4f9cf9"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, title in zip(
        axes,
        [r2s, rmses, maes],
        ["R² Score  ↑ higher = better",
         "RMSE  ↓ lower = better",
         "MAE  ↓ lower = better"],
    ):
        bars = ax.bar(models, vals, color=colors, width=0.5, zorder=3,
                      edgecolor="#0e1420", linewidth=1.5)
        ax.set_title(title, fontsize=9, color="#6b8cba", pad=8)
        ax.set_ylim(0, max(vals) * 1.28)
        ax.grid(axis="y", zorder=0)
        ax.tick_params(axis="x", labelsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.025,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=8.5, color="#cdd6f4", fontweight="600")
    fig.suptitle("Model Performance Comparison", fontsize=12,
                 fontweight="bold", color="#f0f4ff", y=1.03)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💳 Credit Risk ML")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮  Predict Risk", "📊  Dashboard"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div class="chip" style="margin-top:0.75rem">
        ⚠️ <strong>Risk Bands</strong><br>
        🟢 Low &nbsp;&nbsp;&nbsp;— below 10%<br>
        🟡 Medium — 10% to 16%<br>
        🔴 High &nbsp;&nbsp;— above 16%
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="chip" style="margin-top:0.75rem; font-size:0.78rem;">
        🔧 <strong>Model features:</strong> {len(FEATURE_COLS)}<br>
        {' · '.join(FEATURE_COLS[:4])} ...
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💳 Credit Risk Predictor</h1>
    <p style="margin-bottom:0.9rem">ML-powered loan interest rate prediction for smarter lending decisions</p>
    <span class="badge">Random Forest</span>
    <span class="badge">R² = 0.8288</span>
    <span class="badge">ROC AUC ≈ 0.96</span>
    <span class="badge">13 Features</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
#  PAGE 1 — PREDICT RISK
# ════════════════════════════════════════════════════
if "Predict" in page:

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        st.markdown('<div class="sec-label">👤 Personal Info</div>', unsafe_allow_html=True)
        with st.container():
            age        = st.slider("Age", 18, 75, 30)
            income     = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
            emp_length = st.slider("Employment Length (yrs)", 0, 40, 5)
            home_own   = st.selectbox("Home Ownership", list(HOME_OWNERSHIP_MAP.keys()), index=3)
            cred_hist  = st.slider("Credit History (yrs)", 1, 30, 5)

    with col2:
        st.markdown('<div class="sec-label">💰 Loan Details</div>', unsafe_allow_html=True)
        with st.container():
            loan_amnt   = st.number_input("Loan Amount ($)", 500, 50000, 10000, step=500)
            loan_intent = st.selectbox("Loan Intent", list(LOAN_INTENT_MAP.keys()), index=4)
            loan_grade  = st.selectbox("Loan Grade  (A = best → G = riskiest)", list(LOAN_GRADE_MAP.keys()), index=1)
            default_rec = st.selectbox("Prior Default on File", list(DEFAULT_MAP.keys()), index=0)
            loan_status = st.selectbox("Loan Status  (0 = No Default · 1 = Default)", [0, 1], index=0)

    with col3:
        st.markdown('<div class="sec-label">📐 Computed Ratios</div>', unsafe_allow_html=True)
        loan_pct = round(loan_amnt / max(income, 1), 6)
        dti      = round(loan_amnt / max(income, 1), 6)
        c_util   = round(loan_pct * 100, 4)
        emp_inc  = round(emp_length / max(income / 10000, 0.001), 6)

        st.metric("Loan % of Income",        f"{loan_pct*100:.2f}%")
        st.metric("Debt-to-Income Ratio",    f"{dti:.4f}")
        st.metric("Credit Utilization",      f"{c_util:.2f}%")
        st.metric("Employment-Income Ratio", f"{emp_inc:.4f}")
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡  Predict Interest Rate")

    # ── RESULT ─────────────────────────────────
    if predict_clicked:
        fields = {
            "person_age":                 age,
            "person_income":              float(income),
            "person_home_ownership":      HOME_OWNERSHIP_MAP[home_own],
            "person_emp_length":          float(emp_length),
            "loan_intent":                LOAN_INTENT_MAP[loan_intent],
            "loan_grade":                 LOAN_GRADE_MAP[loan_grade],
            "loan_amnt":                  float(loan_amnt),
            "loan_percent_income":        loan_pct,
            "cb_person_default_on_file":  DEFAULT_MAP[default_rec],
            "cb_person_cred_hist_length": float(cred_hist),
            "loan_status":                int(loan_status),
            "DebtIncomeRatio":            dti,
            "CreditUtilization":          c_util,
            "EmploymentIncomeRatio":      emp_inc,
        }

        with st.spinner("Running model..."):
            rate = predict_rate(fields)

        lvl, color, icon, label = risk_level(rate)

        st.markdown("---")
        r1, r2 = st.columns([1, 1.6], gap="large")

        with r1:
            st.markdown(f"""
            <div class="result-wrap {lvl}">
                <span class="result-rate" style="color:{color}">{rate:.2f}%</span>
                <div class="result-sub">Predicted Loan Interest Rate</div>
                <span class="risk-pill"
                    style="background:{color}18; color:{color}; border:1.5px solid {color}40">
                    {icon} {label}
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(gauge_chart(rate), use_container_width=True)

        with r2:
            st.markdown('<div class="sec-label">📋 Risk Interpretation</div>', unsafe_allow_html=True)
            if lvl == "low":
                st.success("**✅ Low Risk Borrower**\n\nRate below 10%. Excellent credit profile — strong candidate for approval at competitive terms.")
            elif lvl == "medium":
                st.warning("**⚠️ Medium Risk Borrower**\n\nRate 10–16%. Moderate risk — review employment history and debt obligations before approving.")
            else:
                st.error("**🚨 High Risk Borrower**\n\nRate above 16%. Elevated default probability — consider additional collateral, reduced loan amount, or rejection.")

            st.markdown('<div class="sec-label">📄 Applicant Summary</div>', unsafe_allow_html=True)
            summary = pd.DataFrame({
                "Feature": ["Loan Grade", "Annual Income", "Loan Amount",
                            "Debt-to-Income", "Loan % Income", "Prior Default",
                            "Employment", "Loan Status"],
                "Value":   [loan_grade, f"${income:,}", f"${loan_amnt:,}",
                            f"{dti:.4f}", f"{loan_pct*100:.1f}%", default_rec,
                            f"{emp_length} yrs", str(loan_status)]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════
#  PAGE 2 — DASHBOARD
# ════════════════════════════════════════════════════
elif "Dashboard" in page:

    # KPI row
    st.markdown('<div class="sec-label">🏆 Best Model — Random Forest Performance</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    for col, val, lbl, accent in zip(
        [k1, k2, k3, k4],
        ["0.8288", "1.2873", "0.9489", "~0.96"],
        ["R² Score", "RMSE", "MAE", "ROC AUC"],
        ["blue", "green", "amber", "purple"],
    ):
        col.markdown(f"""
        <div class="stat-card {accent}">
            <span class="stat-val">{val}</span>
            <span class="stat-lbl">{lbl}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊  Model Comparison", "🔍  Feature Importance", "📋  Results Table"])

    with tab1:
        st.markdown('<div class="sec-label">Model Comparison — All 3 Models</div>', unsafe_allow_html=True)
        st.pyplot(model_comparison_chart(), use_container_width=True)
        st.markdown("""
        <div class="chip" style="margin-top:1rem">
            💡 <strong>Takeaway:</strong> Random Forest outperforms across all metrics —
            highest R² (82.88%), lowest RMSE (1.2873) and MAE (0.9489).
            CV R² (0.8227) ≈ Test R² (0.8288) confirms no overfitting.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="sec-label">Feature Importance — from your trained model</div>', unsafe_allow_html=True)
        fig_fi, fi_df = feature_importance_chart()
        st.pyplot(fig_fi, use_container_width=True)

        fc1, fc2 = st.columns([1, 1], gap="large")
        with fc1:
            st.markdown('<div class="sec-label">All Features Ranked</div>', unsafe_allow_html=True)
            st.dataframe(
                fi_df.reset_index(drop=True)
                     .style.background_gradient(subset=["Importance"], cmap="Blues"),
                use_container_width=True, hide_index=True,
            )
        with fc2:
            st.markdown('<div class="sec-label">Key Insights</div>', unsafe_allow_html=True)
            for feat, desc in [
                ("🏆 loan_grade",               "Dominates with ~87% importance — #1 risk driver"),
                ("💰 DebtIncomeRatio",           "Loan size vs income — affordability signal"),
                ("📊 EmploymentIncomeRatio",     "Employment stability relative to income"),
                ("💵 person_income",             "Higher income → lower perceived risk"),
                ("🔢 loan_amnt",                 "Larger loans → higher interest rates"),
                ("📅 loan_percent_income",       "% of income committed to this loan"),
                ("🕑 cb_person_cred_hist_length","Longer credit history → more trust"),
            ]:
                st.markdown(
                    f'<div class="chip"><strong>{feat}</strong><br><span style="font-size:0.82rem">{desc}</span></div>',
                    unsafe_allow_html=True
                )

    with tab3:
        st.markdown('<div class="sec-label">Full Model Comparison Table</div>', unsafe_allow_html=True)
        df_comp = pd.DataFrame([
            {"Model": "Linear Regression",      "R²": 0.7976, "RMSE": 1.3998, "MAE": 1.0741, "ROC AUC": "~0.93", "CV R²": "~0.79"},
            {"Model": "Decision Tree",           "R²": 0.8237, "RMSE": 1.3063, "MAE": 0.9539, "ROC AUC": "~0.95", "CV R²": "~0.82"},
            {"Model": "✅ Random Forest (Best)", "R²": 0.8288, "RMSE": 1.2873, "MAE": 0.9489, "ROC AUC": "~0.96", "CV R²": "0.8227"},
        ])
        st.dataframe(
            df_comp.style
                .highlight_max(subset=["R²"],         color="#0d3320")
                .highlight_min(subset=["RMSE", "MAE"], color="#0d3320"),
            use_container_width=True, hide_index=True,
        )

        st.markdown('<div class="sec-label">Pipeline Overview</div>', unsafe_allow_html=True)
        steps = [
            ("1️⃣", "Data Collection",      "Loaded dataset with 13 raw features"),
            ("2️⃣", "Data Preparation",     "Handled nulls, label encoding, duplicate removal, IQR outlier detection"),
            ("3️⃣", "EDA",                  "Histograms, boxplots, scatter plots, correlation heatmap"),
            ("4️⃣", "Feature Engineering",  "Created DebtIncomeRatio, CreditUtilization, LoanBurden, EmploymentIncomeRatio"),
            ("5️⃣", "SMOTE Balancing",      "Fixed 78/22 class imbalance → 50/50 split"),
            ("6️⃣", "Train/Test Split",     "80/20 split · StandardScaler for Linear Regression"),
            ("7️⃣", "Model Training",       "GridSearchCV + 5-Fold CV for all 3 models"),
            ("8️⃣", "Evaluation",           "RMSE, MAE, R², ROC AUC, Predicted vs Actual plots"),
            ("9️⃣", "Feature Importance",   "LR coefficients + RF feature importances"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div class="chip" style="display:flex; gap:0.75rem; align-items:flex-start">
                <span style="font-size:1.2rem">{icon}</span>
                <div><strong style="color:#cdd6f4">{title}</strong><br>
                <span style="font-size:0.82rem">{desc}</span></div>
            </div>
            """, unsafe_allow_html=True)