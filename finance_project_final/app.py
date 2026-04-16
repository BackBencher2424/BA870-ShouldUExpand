import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from project_utils import (
    DISPLAY_COLS,
    FEATURE_COLS,
    TARGET_DESCRIPTION,
    explain_ratios,
    fetch_live_features_from_yfinance,
    get_same_sic_peer_median,
    recommend_from_probability,
)

BASE_DIR = os.path.dirname(__file__)
ART_DIR = os.path.join(BASE_DIR, "artifacts")

st.set_page_config(page_title="Expansion Readiness Analyzer", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
:root {
  --bg: #040404;
  --panel: #09101f;
  --gold: #d4af37;
  --gold-soft: rgba(212,175,55,0.58);
  --text: #f4e9c1;
  --muted: #d9c58c;
}
.stApp {background: var(--bg); color: var(--text);}
.block-container {padding-top: 3.0rem; max-width: 1500px;}
h1, h2, h3, h4, h5, h6, p, li, label, span, div {color: var(--text);}
.stRadio label, .stSelectbox label, .stTextInput label, .stMetricLabel, .stMetricValue {color: var(--text) !important;}
div[data-testid="stTable"], [data-testid="stDataFrame"] {background: #0a0a0a;}
section[data-testid="stSidebar"] {background: #06090f !important; border-right: 1px solid rgba(212,175,55,0.22);}
section[data-testid="stSidebar"] * {color: var(--text) !important;}
.power-title {text-align: center; font-size: 2.95rem; line-height: 1.08; font-weight: 800; color: #f3e8bd; margin: 1.8rem 0 0.85rem 0;}
.power-sub {text-align: center; font-size: 1.12rem; color: var(--muted); max-width: 1080px; margin: 0 auto 2.4rem auto;}
.small-note {color: var(--muted);}
.home-grid {display:grid; grid-template-columns:repeat(4, minmax(220px, 1fr)); gap: 2.1rem 2.6rem; max-width: 1220px; margin: 2.0rem auto 0.8rem auto;}
.home-tile-link {text-decoration:none !important;}
.home-tile {width: 240px; height: 240px; margin: 0 auto; border-radius: 50%; background: var(--bg); border: 2px solid var(--gold-soft); display:flex; align-items:center; justify-content:center; text-align:center; padding: 1.25rem; transition: all 0.18s ease-in-out; box-sizing:border-box;}
.home-tile:hover {border-color: var(--gold); background:#09101f;}
.home-tile-inner {display:flex; flex-direction:column; align-items:center; justify-content:center; gap:0.55rem;}
.home-tile-title {color: var(--gold); font-weight: 800; font-size: 1.38rem; line-height: 1.18;}
.home-tile-desc {color: var(--gold); font-style: italic; font-size: 0.90rem; line-height: 1.38;}
@media (max-width: 1200px) { .home-grid {grid-template-columns:repeat(2, minmax(220px, 1fr));} }
@media (max-width: 720px) { .home-grid {grid-template-columns:repeat(1, minmax(220px, 1fr));} .home-tile {width:220px;height:220px;} .power-title{font-size:2.3rem;} }
div[data-testid="stButton"] > button[kind="primary"] {background: linear-gradient(180deg, #e1c259 0%, #c89f24 100%) !important; color: #141414 !important; border: 1px solid #f1d97c !important; border-radius: 16px !important; font-weight: 700 !important; padding: 0.55rem 1rem !important;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PAGE_NAMES = [
    "Home",
    "Info about the App",
    "Model Details",
    "Company Details",
    "Comparison Metrics",
    "Model Evaluation",
    "Summary Statistics",
    "Goal of the Project",
]

def sync_page_from_query_params():
    try:
        qp_page = st.query_params.get('page', None)
        if qp_page in PAGE_NAMES:
            st.session_state.page = qp_page
    except Exception:
        pass


def set_query_page(page_name: str):
    try:
        if page_name == 'Home':
            st.query_params.clear()
        else:
            st.query_params['page'] = page_name
    except Exception:
        pass

FRIENDLY_LABELS = {
    "revenue_growth": "Revenue Growth",
    "roa": "ROA (Return on Assets)",
    "debt_ratio": "Debt Ratio",
    "cfo_assets": "CFO / Assets (Cash Flow from Operations / Total Assets)",
    "current_ratio": "Current Ratio",
    "profit_margin": "Profit Margin",
}

FORMULAS = {
    "revenue_growth": r"Revenue\ Growth = \frac{Sales_t - Sales_{t-1}}{Sales_{t-1}}",
    "roa": r"ROA = \frac{Net\ Income}{Total\ Assets}",
    "debt_ratio": r"Debt\ Ratio = \frac{Current\ Debt + Long\text{-}Term\ Debt}{Total\ Assets}",
    "cfo_assets": r"CFO / Assets = \frac{Cash\ Flow\ from\ Operations}{Total\ Assets}",
    "current_ratio": r"Current\ Ratio = \frac{Current\ Assets}{Current\ Liabilities}",
    "profit_margin": r"Profit\ Margin = \frac{Net\ Income}{Sales}",
}

ABBREVIATIONS = {
    "WRDS": "Wharton Research Data Services",
    "SIC": "Standard Industrial Classification",
    "ROA": "Return on Assets",
    "CFO": "Cash Flow from Operations",
    "CSV": "Comma-Separated Values file",
    "ROC-AUC": "Receiver Operating Characteristic - Area Under the Curve",
    "F1": "The harmonic mean of precision and recall",
    "C": "Regularization strength parameter in logistic regression",
}


def patch_loaded_model(model):
    try:
        lr = None
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            lr = model.named_steps["model"]
        elif model.__class__.__name__ == "LogisticRegression":
            lr = model
        if lr is not None:
            if not hasattr(lr, "multi_class"):
                lr.multi_class = "auto"
            if not hasattr(lr, "solver") or lr.solver is None:
                lr.solver = "liblinear"
        return model
    except Exception:
        return model


@st.cache_data
def load_artifacts():
    latest = pd.read_csv(os.path.join(ART_DIR, "latest_company_features.csv"))
    peers = pd.read_csv(os.path.join(ART_DIR, "peer_medians_latest.csv"))
    coeffs = pd.read_csv(os.path.join(ART_DIR, "model_coefficients.csv"))
    with open(os.path.join(ART_DIR, "model_metrics.json"), "r") as f:
        metrics = json.load(f)
    with open(os.path.join(ART_DIR, "expansion_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return latest, peers, coeffs, metrics, patch_loaded_model(model)


@st.cache_data(show_spinner=False)
def cached_live_features(ticker: str):
    return fetch_live_features_from_yfinance(ticker)


latest_df, peer_df, coeff_df, metrics, model = load_artifacts()

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "source" not in st.session_state:
    st.session_state.source = "Latest WRDS record"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = sorted(latest_df["tic"].dropna().unique())[0]
if "live_ticker" not in st.session_state:
    st.session_state.live_ticker = "AAPL"

sync_page_from_query_params()

if st.session_state.page == "Home":
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {display:none !important;}
        .block-container {padding-top: 4.8rem !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        div[data-testid="collapsedControl"] {display:block !important;}
        .block-container {padding-top: 2.1rem !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def navigate(page_name: str):
    st.session_state.page = page_name
    set_query_page(page_name)


def top_home_button():
    if st.session_state.page != "Home":
        c1, c2 = st.columns([8, 2])
        with c2:
            if st.button("← Back to Home", key=f"back_home_{st.session_state.page}", type="primary"):
                navigate("Home")
                st.rerun()


def render_sidebar():
    if st.session_state.page == "Home":
        return
    with st.sidebar:
        st.markdown("## Navigation")
        sidebar_pages = PAGE_NAMES
        current_idx = sidebar_pages.index(st.session_state.page) if st.session_state.page in sidebar_pages else 0
        selected = st.radio(
            "Open page",
            sidebar_pages,
            index=current_idx,
            key="sidebar_page_selector",
        )
        if selected != st.session_state.page:
            navigate(selected)
            st.rerun()
        st.markdown("---")
        if st.session_state.get("source", "Latest WRDS record") == "Latest WRDS record":
            st.caption(f"Current ticker: {st.session_state.get('selected_ticker', 'Not selected')}")
        else:
            st.caption(f"Live ticker: {st.session_state.get('live_ticker', 'Not selected')}")
        st.caption("Use the Company Details page to change the company selection used across the app.")


def get_current_company_context():
    source = st.session_state.get("source", "Latest WRDS record")
    live_warning = None
    company_row = None
    peer_row = None

    if source == "Latest WRDS record":
        ticker = st.session_state.get("selected_ticker")
        if ticker in set(latest_df["tic"].dropna().unique()):
            company_row = latest_df[latest_df["tic"] == ticker].iloc[0]
            peer_row = get_same_sic_peer_median(latest_df, company_row)
    else:
        ticker = st.session_state.get("live_ticker", "").upper().strip()
        if ticker:
            try:
                live_dict = cached_live_features(ticker)
                company_row = pd.Series(live_dict)
                if ticker in set(latest_df["tic"].dropna().unique()):
                    ref_row = latest_df[latest_df["tic"] == ticker].iloc[0]
                    company_row["conm"] = ref_row.get("conm", ticker)
                    company_row["sic_final"] = ref_row.get("sic_final", np.nan)
                    peer_row = get_same_sic_peer_median(latest_df, ref_row)
                else:
                    live_warning = (
                        "This live ticker is not in the training dataset, so a same-SIC peer comparison is not available. "
                        "The app can still score the company using the live ratios."
                    )
            except Exception as e:
                live_warning = f"Live data could not be loaded for this ticker. Details: {e}"

    probability = None
    recommendation = None
    input_df = None
    if company_row is not None:
        input_df = pd.DataFrame([company_row])[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        probability = float(model.predict_proba(input_df)[0, 1])
        recommendation = recommend_from_probability(probability)

    return company_row, peer_row, probability, recommendation, live_warning


render_sidebar()

company_row, peer_row, probability, recommendation, live_warning = get_current_company_context()


def render_glossary():
    with st.expander("Meaning of abbreviations used in this app"):
        for abbr, meaning in ABBREVIATIONS.items():
            st.write(f"**{abbr}**: {meaning}")




def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to plain strings so Streamlit/Arrow can render them safely."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(lambda x: "Not available" if pd.isna(x) else str(x))
    return out

def format_value(x):
    if pd.isna(x):
        return "Not available"
    if abs(float(x)) < 10:
        return f"{float(x):.2f}"
    return f"{float(x):,.2f}"


def build_comparison_table(row, peer):
    rows = []
    lower_is_better = {"debt_ratio"}
    for col in DISPLAY_COLS:
        company_val = row.get(col, np.nan)
        peer_val = peer.get(col, np.nan) if peer is not None else np.nan
        diff = np.nan if pd.isna(company_val) or pd.isna(peer_val) else company_val - peer_val
        if pd.isna(company_val) or pd.isna(peer_val):
            result = "Not enough data"
        elif col in lower_is_better:
            result = "Better than peers" if company_val < peer_val else "Worse than peers"
        else:
            result = "Better than peers" if company_val > peer_val else "Worse than peers"
        rows.append({
            "Ratio": FRIENDLY_LABELS[col],
            "Company": company_val,
            "Same-SIC Peer Median": peer_val,
            "Difference": diff,
            "Result": result,
        })
    return pd.DataFrame(rows)


# Home page styling
if st.session_state.page == "Home":
    st.markdown('<div class="power-title">Public Company Expansion Readiness Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="power-sub">A beginner-friendly financial analytics tool that scores a selected public company and compares it only with companies that share the same SIC industry code.</div>',
        unsafe_allow_html=True,
    )

    card_info = {
        "Home": "Return to the main page and move to any section of the project.",
        "Info about the App": "Understand the project flow, the ratios used, and the formulas behind the analysis.",
        "Model Details": "See why we chose this model, what goes into it, and how the probability is created.",
        "Company Details": "Select the company ticker and view the basic financial information used in the app.",
        "Comparison Metrics": "Compare the selected company with same-SIC peers on the core financial ratios.",
        "Model Evaluation": "Understand the model performance numbers and what they mean in simple terms.",
        "Summary Statistics": "Read the final recommendation and a simple explanation of the result.",
        "Goal of the Project": "See the overall purpose, limitations, and future scope of the project.",
    }

    rows = [
        ["Home", "Info about the App", "Model Details", "Company Details"],
        ["Comparison Metrics", "Model Evaluation", "Summary Statistics", "Goal of the Project"],
    ]

    cards_html = ['<div class="home-grid">']
    for row_labels in rows:
        for label in row_labels:
            href = '?' if label == 'Home' else '?page=' + label.replace(' ', '%20')
            cards_html.append(
                f'<a class="home-tile-link" href="{href}" target="_self"><div class="home-tile"><div class="home-tile-inner"><div class="home-tile-title">{label}</div><div class="home-tile-desc">{card_info[label]}</div></div></div></a>'
            )
    cards_html.append('</div>')
    st.markdown(''.join(cards_html), unsafe_allow_html=True)
    st.markdown('<p class="small-note" style="text-align:center; margin-top:0.9rem;">Use the Company Details page first if you want to choose a different company before reviewing the comparison and summary pages.</p>', unsafe_allow_html=True)

elif st.session_state.page == "Info about the App":
    top_home_button()
    st.title("Info about the App")
    st.write(
        "This project uses historical company data from WRDS Compustat to train a simple logistic regression model. "
        "The model looks at a small set of financial ratios and estimates whether a company appears financially ready for healthy future growth. "
        "The app then compares the selected company only with other companies that have the same SIC code in the training dataset."
    )

    st.subheader("How the project works")
    st.write(
        "1. We train the model locally using annual company data.  "
        "2. We compute a small number of financial ratios.  "
        "3. We save the trained model.  "
        "4. In the app, we load the saved model, score the selected company, and compare it with same-SIC peers."
    )

    st.subheader("Ratios and formulas used in the project")
    for col in DISPLAY_COLS:
        st.markdown(f"**{FRIENDLY_LABELS[col]}**")
        st.latex(FORMULAS[col])
        if col == "revenue_growth":
            st.write("This tells us whether the company is increasing sales compared with the previous year.")
        elif col == "roa":
            st.write("This shows how efficiently the company is turning its asset base into profit.")
        elif col == "debt_ratio":
            st.write("This shows how much of the company is financed with debt. Lower values are usually safer.")
        elif col == "cfo_assets":
            st.write("This checks whether the company is generating cash from operations relative to its asset base.")
        elif col == "current_ratio":
            st.write("This measures short-term liquidity, which helps us judge whether the company can meet near-term obligations.")
        elif col == "profit_margin":
            st.write("This shows how much profit the company keeps from each dollar of sales.")

    st.subheader("Target used for training")
    st.write(TARGET_DESCRIPTION)
    st.write(
        "In simpler words, the target marks a company-year as successful when the following year shows healthy growth, positive profitability, positive operating cash support, and manageable leverage."
    )
    render_glossary()

elif st.session_state.page == "Model Details":
    top_home_button()
    st.title("Model Details")
    st.write("This page explains the model setup only. It does not show the model results.")

    st.subheader("Why we chose this model")
    st.write(
        "We chose **logistic regression** because our project ends with a simple yes-or-no style question: "
        "does the company look financially ready to expand, or not? This model is a good fit for that kind of decision. "
        "It is also easier to explain than more complex machine-learning models, which is important for a classroom project and for non-technical users."
    )
    st.write(
        "In plain language, the model looks at a few key financial signals together and turns them into a probability. "
        "A higher probability means the company currently looks more financially prepared for growth based on patterns learned from historical data."
    )

    st.subheader("Model used")
    st.write("We use a logistic regression model because the final decision is binary: financially ready to expand, or not financially ready to expand.")
    st.latex(r"P(Y=1|X)=\frac{1}{1+e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_3 + \beta_4x_4)}}")
    st.caption("The equation above is the mathematical form of the model. It combines the selected financial ratios into one probability score between 0 and 1.")

    st.subheader("Why these inputs were used")
    st.write(
        "We intentionally kept the model small and focused so it stays understandable. "
        "Each input captures a different part of financial strength."
    )
    input_explanations = {
        "revenue_growth": "Shows whether the company is growing its sales.",
        "roa": "Shows how efficiently the company turns assets into profit.",
        "debt_ratio": "Shows how much of the business is funded by debt.",
        "cfo_assets": "Shows whether the company is generating operating cash relative to its asset base.",
    }
    for f in FEATURE_COLS:
        st.write(f"- **{FRIENDLY_LABELS[f]}**: {input_explanations.get(f, 'Used as a core financial signal in the model.')}")

    st.subheader("Why we did not use a more complex model")
    st.write(
        "A more complex model might look more advanced, but it would also be harder to explain and defend. "
        "For this project, we preferred a model that is transparent, stable, and easy to connect back to financial logic."
    )

    st.subheader("Pre-processing used before the regression")
    st.write(
        "Before training the model, we cleaned the data so unusual values would not distort the results."
    )
    st.write(
        "- winsorization at the 1st and 99th percentile to reduce extreme ratio outliers  \n"
        "- median imputation for missing values  \n"
        "- standard scaling so the variables are on comparable ranges"
    )

    st.subheader("Model choices")
    st.write(
        "These settings help the model stay practical and reliable:"
    )
    st.write(
        "- class weight: balanced, so the model does not ignore the smaller class  \n"
        "- solver: liblinear, a stable optimization method for this type of regression  \n"
        f"- regularization parameter C: {metrics['selected_C']}, which controls how flexible the model is  \n"
        f"- probability threshold used for final judgment: {metrics['selected_threshold']}, which decides when the score becomes a positive recommendation"
    )

    st.subheader("How the model turns into a recommendation")
    st.write(
        "After the model gives a probability, we convert it into a recommendation. "
        "Higher probability means the company looks more financially prepared for growth. "
        "We also use same-SIC peer comparison to explain the score in a more intuitive way."
    )
    st.write(
        "So the final recommendation is not based on one ratio alone. It is based on the combined pattern across growth, profitability, debt, and cash-flow support."
    )

elif st.session_state.page == "Comparison Metrics":
    top_home_button()
    st.title("Comparison Metrics")
    if live_warning:
        st.warning(live_warning)

    if company_row is None:
        st.info("Please go to the Company Details page and select a company first.")
    else:
        st.subheader(f"Selected company: {company_row.get('conm', 'Company')} ({company_row.get('tic', '')})")
        if peer_row is None:
            st.info("A same-SIC peer comparison is not available for this company.")
        else:
            peer_count = int(peer_row.get("peer_count", 0)) if hasattr(peer_row, "get") else 0
            st.write(f"Comparison group: companies with the same SIC code. Peer count used for the median: **{peer_count}**.")
            comp_df = build_comparison_table(pd.Series(company_row), pd.Series(peer_row))
            st.dataframe(comp_df, use_container_width=True)

            st.subheader("What these comparison results mean")
            for comment in explain_ratios(pd.Series(company_row), pd.Series(peer_row)):
                st.write(f"- {comment}")

            st.subheader("How comparison links to our recommendation")
            st.write(
                "We do not rely on one ratio alone. We look at the model probability and also check whether the company is stronger or weaker than the median company in the same SIC peer group."
            )
            st.markdown(
                "- **Expand**: the model probability is high and the company is generally stronger than same-SIC peers on the main ratios, such as higher **Revenue Growth**, higher **Return on Assets (ROA)**, higher **Cash Flow from Operations divided by Total Assets (CFO/Assets)**, and lower **Debt Ratio**."
            )
            st.markdown(
                "- **Proceed Carefully**: the company has mixed signals. It may be better than peers on some ratios but weaker on others, so expansion may be possible but needs more review."
            )
            st.markdown(
                "- **Not Ready**: the model probability is low and the company is weaker than same-SIC peers on several core ratios, especially if growth, profitability, or cash-flow support is weak or if debt is relatively high."
            )
            st.write(
                "In simple terms, our recommendation is strongest when the model and the same-SIC comparison both point in the same direction."
            )

        st.subheader("Formulas behind the comparison")
        for col in DISPLAY_COLS:
            st.markdown(f"**{FRIENDLY_LABELS[col]}**")
            st.latex(FORMULAS[col])

elif st.session_state.page == "Model Evaluation":
    top_home_button()
    st.title("Model Evaluation")
    st.write("This page shows how the model performed on data that was not used for model fitting.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
    c2.metric("F1 Score", f"{metrics['test_f1']:.3f}")
    c3.metric("ROC-AUC", f"{metrics['test_auc']:.3f}")
    c4.metric("Threshold", f"{metrics['selected_threshold']:.2f}")

    st.subheader("What each evaluation number means")
    st.write(
        f"- **Accuracy** tells us how often the model was correct overall. Here it is {metrics['test_accuracy']:.3f}.  \n"
        f"- **F1 Score** balances precision and recall. Here it is {metrics['test_f1']:.3f}.  \n"
        f"- **ROC-AUC** tells us how well the model separates stronger companies from weaker ones across many thresholds. Here it is {metrics['test_auc']:.3f}.  \n"
        f"- **Threshold** is the cutoff used to convert probability into a yes/no style recommendation. Here it is {metrics['selected_threshold']:.2f}."
    )

    st.subheader("Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.dataframe(cm_df, use_container_width=True)
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negatives", tn)
    c2.metric("False Positives", fp)
    c3.metric("False Negatives", fn)
    c4.metric("True Positives", tp)
    st.write(
        f"- **True Negatives ({tn})**: companies the model correctly identified as not ready to expand.  \n"
        f"- **False Positives ({fp})**: companies the model incorrectly flagged as ready to expand.  \n"
        f"- **False Negatives ({fn})**: companies the model missed even though they were actually in the positive class.  \n"
        f"- **True Positives ({tp})**: companies the model correctly identified as ready to expand."
    )
    st.write(
        "In plain English, the confusion matrix helps us see whether the model is making the right kind of mistakes. For this project, we especially care about not overstating expansion readiness."
    )

    st.subheader("Coefficient table")
    coeff_show = coeff_df.copy()
    coeff_show["feature"] = coeff_show["feature"].map(FRIENDLY_LABELS)
    st.dataframe(coeff_show, use_container_width=True)
    st.write(
        "Positive coefficients increase the probability that a company looks ready to expand. Negative coefficients reduce that probability."
    )

elif st.session_state.page == "Company Details":
    top_home_button()
    st.title("Company Details")
    st.write("Select the company here. The app will use this choice across the comparison and summary pages.")

    source_choice = st.radio(
        "Choose data source",
        ["Latest WRDS record", "Live public data from Yahoo Finance"],
        index=0 if st.session_state.source == "Latest WRDS record" else 1,
        horizontal=True,
    )
    st.session_state.source = source_choice

    if source_choice == "Latest WRDS record":
        ticker_choice = st.selectbox(
            "Select company ticker",
            sorted(latest_df["tic"].dropna().unique()),
            index=sorted(latest_df["tic"].dropna().unique()).index(st.session_state.selected_ticker)
            if st.session_state.selected_ticker in sorted(latest_df["tic"].dropna().unique()) else 0,
        )
        st.session_state.selected_ticker = ticker_choice
    else:
        ticker_input = st.text_input("Enter public ticker", value=st.session_state.live_ticker)
        st.session_state.live_ticker = ticker_input.upper().strip()

    company_row, peer_row, probability, recommendation, live_warning = get_current_company_context()
    if live_warning:
        st.warning(live_warning)

    if company_row is not None:
        st.subheader(f"{company_row.get('conm', 'Selected Company')} ({company_row.get('tic', '')})")
        data = {
            "Company Name": company_row.get("conm", "Not available"),
            "Ticker": company_row.get("tic", "Not available"),
            "SIC (Industry Code)": company_row.get("sic_final", "Not available"),
            "Revenue Growth": format_value(company_row.get("revenue_growth", np.nan)),
            "ROA (Return on Assets)": format_value(company_row.get("roa", np.nan)),
            "Debt Ratio": format_value(company_row.get("debt_ratio", np.nan)),
            "CFO / Assets": format_value(company_row.get("cfo_assets", np.nan)),
            "Current Ratio": format_value(company_row.get("current_ratio", np.nan)),
            "Profit Margin": format_value(company_row.get("profit_margin", np.nan)),
        }
        details_df = pd.DataFrame({"Field": list(data.keys()), "Value": list(data.values())})
        st.table(make_arrow_safe(details_df))
    else:
        st.info("Select a company to see its details.")

elif st.session_state.page == "Summary Statistics":
    top_home_button()
    st.title("Summary Statistics")
    if live_warning:
        st.warning(live_warning)

    if company_row is None or probability is None:
        st.info("Please go to the Company Details page and select a company first.")
    else:
        st.subheader(f"Summary for {company_row.get('conm', 'Selected Company')} ({company_row.get('tic', '')})")
        m1, m2 = st.columns(2)
        m1.metric("Expansion-readiness probability", f"{probability:.1%}")
        m2.metric("Recommendation", recommendation)

        st.subheader("Layman explanation")
        if recommendation == "Expand":
            st.write(
                "This result suggests that the company currently looks financially strong enough to support growth. "
                "It is showing a combination of acceptable growth, profitability, cash generation, and manageable debt."
            )
        elif recommendation == "Proceed Carefully":
            st.write(
                "This result suggests that the company has some strengths, but the financial signals are mixed. "
                "Growth may be possible, but it should be approached carefully because not every core ratio is clearly strong."
            )
        else:
            st.write(
                "This result suggests that the company may not be financially ready for expansion right now. "
                "Some of the main warning signals are likely coming from weak profitability, weak cash support, or a debt profile that looks less comfortable."
            )

        if peer_row is not None:
            st.subheader("How the company compares with same-SIC peers")
            comp_df = build_comparison_table(pd.Series(company_row), pd.Series(peer_row))
            better = int((comp_df["Result"] == "Better than peers").sum())
            worse = int((comp_df["Result"] == "Worse than peers").sum())
            st.write(f"The company is better than the same-SIC peer median on **{better}** ratio(s) and worse on **{worse}** ratio(s).")
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.info("A same-SIC peer comparison is not available for this selected company.")

elif st.session_state.page == "Goal of the Project":
    top_home_button()
    st.title("Goal of the Project")
    st.subheader("Project goal")
    st.write(
        "The goal of this project is to build a simple and easy-to-understand financial analytics tool that helps a user judge whether a public company looks financially ready to expand."
    )

    st.subheader("What the project does well")
    st.write(
        "- uses a small and interpretable model  \n"
        "- compares the selected company with same-SIC peers instead of the full dataset  \n"
        "- explains the financial ratios in plain language  \n"
        "- keeps the model training separate from the app so the app stays light"
    )

    st.subheader("Limitations")
    st.write(
        "- the model uses public financial statement data, so it cannot capture management quality, strategy execution, or private information  \n"
        "- the target is a practical proxy, not a perfect real-world label for an expansion decision  \n"
        "- live peer comparison is only available when the ticker exists in the training dataset and has a known SIC mapping"
    )

    st.subheader("Future scope")
    st.write(
        "In a future version, we could add more industries, more years, valuation ratios, text analysis from company filings, or a second model such as a tree-based classifier for comparison."
    )
