
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cleaned ratios kept in the regression model.
FEATURE_COLS = [
    "revenue_growth",
    "roa",
    "debt_ratio",
    "cfo_assets",
]

# A few additional ratios kept only for explanation and peer comparison.
DISPLAY_COLS = [
    "revenue_growth",
    "roa",
    "debt_ratio",
    "cfo_assets",
    "current_ratio",
    "profit_margin",
]

TARGET_DESCRIPTION = (
    "Target = 1 when next-year revenue growth beats either the industry median "
    "or 5%, next-year ROA is above 2%, next-year CFO/Assets is positive, and "
    "next-year Debt/Assets is below 0.60."
)

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric feature to the training-set quantile range."""
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        self.feature_names_in_ = list(X_df.columns)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if hasattr(self, "feature_names_in_") and len(self.feature_names_in_) == X_df.shape[1]:
            X_df.columns = self.feature_names_in_
        return X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def load_wrds_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "fyear", "gvkey", "sic", "sich",
        "act", "at", "dlc", "dltt", "invt", "lct", "rect",
        "ni", "oiadp", "sale", "xint", "oancf"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sich" in df.columns:
        df["sic_final"] = df["sich"].fillna(df["sic"])
    else:
        df["sic_final"] = df["sic"]

    df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    return df


def build_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("gvkey", group_keys=False)

    lag_cols = ["sale", "at", "ni", "act", "lct", "invt", "dlc", "dltt", "oiadp", "xint", "oancf", "rect"]
    for col in lag_cols:
        df[f"{col}_lag"] = grp[col].shift(1)
        df[f"{col}_lead"] = grp[col].shift(-1)

    # Cleaner current-year ratios
    df["revenue_growth"] = safe_divide(df["sale"] - df["sale_lag"], df["sale_lag"])
    df["roa"] = safe_divide(df["ni"], df["at"])
    df["debt_ratio"] = safe_divide(df["dlc"].fillna(0) + df["dltt"].fillna(0), df["at"])
    df["cfo_assets"] = safe_divide(df["oancf"], df["at"])

    # Extra diagnostics for interpretation only.
    df["current_ratio"] = safe_divide(df["act"], df["lct"])
    df["profit_margin"] = safe_divide(df["ni"], df["sale"])

    # Next-year outcomes for the target.
    df["revenue_growth_next"] = safe_divide(df["sale_lead"] - df["sale"], df["sale"])
    df["roa_next"] = safe_divide(df["ni_lead"], df["at_lead"])
    df["debt_ratio_next"] = safe_divide(df["dlc_lead"].fillna(0) + df["dltt_lead"].fillna(0), df["at_lead"])
    df["cfo_assets_next"] = safe_divide(df["oancf_lead"], df["at_lead"])

    df["rg_next_peer_med"] = df.groupby(["sic_final", "fyear"])["revenue_growth_next"].transform("median")

    has_target_inputs = df[["revenue_growth_next", "roa_next", "debt_ratio_next", "cfo_assets_next"]].notna().all(axis=1)
    df["target"] = np.where(
        has_target_inputs
        & ((df["revenue_growth_next"] > df["rg_next_peer_med"]) | (df["revenue_growth_next"] > 0.05))
        & (df["roa_next"] > 0.02)
        & (df["cfo_assets_next"] > 0)
        & (df["debt_ratio_next"] < 0.60),
        1,
        0,
    )
    df.loc[~has_target_inputs, "target"] = np.nan

    ratio_cols = DISPLAY_COLS + [
        "revenue_growth_next",
        "roa_next",
        "debt_ratio_next",
        "cfo_assets_next",
    ]
    df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)
    return df


def prepare_model_data(df: pd.DataFrame, start_year: int = 2010) -> pd.DataFrame:
    model_df = df.copy()
    model_df = model_df[model_df["fyear"] >= start_year]
    model_df = model_df[(model_df["sale"] > 0) & (model_df["at"] > 0)]
    model_df = model_df.drop_duplicates(["gvkey", "fyear"])
    model_df = model_df.dropna(subset=["target"]).copy()
    model_df["target"] = model_df["target"].astype(int)

    # Remove especially weak rows where most model inputs are missing.
    non_missing_counts = model_df[FEATURE_COLS].notna().sum(axis=1)
    model_df = model_df[non_missing_counts >= 2].copy()
    return model_df


def split_train_val_test_by_year(
    model_df: pd.DataFrame, train_end_year: int = 2020, val_end_year: int = 2022
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = model_df[model_df["fyear"] <= train_end_year].copy()
    val_df = model_df[(model_df["fyear"] > train_end_year) & (model_df["fyear"] <= val_end_year)].copy()
    test_df = model_df[model_df["fyear"] > val_end_year].copy()
    return train_df, val_df, test_df


def build_pipeline(C: float = 0.2) -> Pipeline:
    return Pipeline(
        steps=[
            ("winsorizer", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=C,
                solver="liblinear",
                random_state=42,
            )),
        ]
    )


def _score_threshold(y_true: pd.Series, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def train_and_tune_logistic(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    c_grid: Optional[List[float]] = None,
    threshold_grid: Optional[List[float]] = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    if c_grid is None:
        c_grid = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    if threshold_grid is None:
        threshold_grid = [round(x, 2) for x in np.arange(0.35, 0.71, 0.01)]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"].astype(int)
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["target"].astype(int)

    best = None
    for C in c_grid:
        pipeline = build_pipeline(C=C)
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_prob))

        for threshold in threshold_grid:
            s = _score_threshold(y_val, val_prob, threshold)
            # Favor accuracy, but still reward F1 so the model does not become too conservative.
            score = 0.60 * s["accuracy"] + 0.40 * s["f1"]
            candidate = {
                "score": score,
                "C": C,
                "threshold": s["threshold"],
                "val_accuracy": s["accuracy"],
                "val_f1": s["f1"],
                "val_precision": s["precision"],
                "val_recall": s["recall"],
                "val_balanced_accuracy": s["balanced_accuracy"],
                "val_auc": val_auc,
                "pipeline": pipeline,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    best_pipeline = best.pop("pipeline")
    return best_pipeline, best


def evaluate_model(
    model: Pipeline,
    test_df: pd.DataFrame,
    threshold: float,
) -> Dict[str, float]:
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"].astype(int)
    test_prob = model.predict_proba(X_test)[:, 1]
    s = _score_threshold(y_test, test_prob, threshold)
    s["auc"] = float(roc_auc_score(y_test, test_prob))
    s["confusion_matrix"] = confusion_matrix(y_test, (test_prob >= threshold).astype(int)).tolist()
    return s


def get_model_coefficients(model: Pipeline) -> pd.DataFrame:
    lr = model.named_steps["model"]
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": lr.coef_[0],
        "abs_coefficient": np.abs(lr.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)
    return coef_df


def get_latest_company_rows(full_df: pd.DataFrame) -> pd.DataFrame:
    latest_idx = full_df.groupby("tic")["fyear"].idxmax()
    latest = full_df.loc[latest_idx].copy().sort_values("tic")
    latest = latest[latest[DISPLAY_COLS].notna().sum(axis=1) >= 2].copy()
    return latest


def get_peer_medians(latest_df: pd.DataFrame) -> pd.DataFrame:
    peers = latest_df.groupby("sic_final")[DISPLAY_COLS].median().reset_index()
    return peers




def get_same_sic_peer_median(latest_df: pd.DataFrame, company_row: pd.Series) -> Optional[pd.Series]:
    """Return same-SIC peer medians, excluding the selected company when possible."""
    if latest_df is None or company_row is None:
        return None
    sic_val = company_row.get("sic_final", np.nan)
    if pd.isna(sic_val):
        return None

    peers = latest_df[latest_df["sic_final"] == sic_val].copy()
    # Exclude the selected company from peers when possible.
    if "gvkey" in peers.columns and pd.notna(company_row.get("gvkey", np.nan)):
        peers = peers[peers["gvkey"] != company_row.get("gvkey")]
    elif "tic" in peers.columns and pd.notna(company_row.get("tic", np.nan)):
        peers = peers[peers["tic"] != company_row.get("tic")]

    # If there are no other firms in that SIC, fall back to the same SIC including the firm itself.
    if peers.empty:
        peers = latest_df[latest_df["sic_final"] == sic_val].copy()
    if peers.empty:
        return None

    med = peers[DISPLAY_COLS].median(numeric_only=True)
    med["sic_final"] = sic_val
    med["peer_count"] = int(len(peers))
    return med

def recommend_from_probability(prob: float) -> str:
    if prob >= 0.62:
        return "Expand"
    if prob >= 0.45:
        return "Proceed Carefully"
    return "Do Not Expand Yet"


def save_artifacts(
    base_dir: str,
    model: Pipeline,
    latest_df: pd.DataFrame,
    peer_medians: pd.DataFrame,
    metrics: Dict,
    coef_df: pd.DataFrame,
) -> str:
    if base_dir is None:
        base_dir = os.getcwd()
    art_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    with open(os.path.join(art_dir, "expansion_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(art_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    latest_df.to_csv(os.path.join(art_dir, "latest_company_features.csv"), index=False)
    peer_medians.to_csv(os.path.join(art_dir, "peer_medians_latest.csv"), index=False)
    coef_df.to_csv(os.path.join(art_dir, "model_coefficients.csv"), index=False)

    with open(os.path.join(art_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return art_dir


def explain_ratios(row: pd.Series, peer_row: pd.Series) -> List[str]:
    comments = []
    comparisons = {
        "revenue_growth": "recent growth momentum",
        "roa": "profitability",
        "debt_ratio": "leverage",
        "cfo_assets": "cash-flow support",
        "current_ratio": "short-term liquidity",
        "profit_margin": "profit margin quality",
    }
    lower_is_better = {"debt_ratio"}
    for col, label in comparisons.items():
        firm_val = row.get(col, np.nan)
        peer_val = peer_row.get(col, np.nan)
        if pd.isna(firm_val) or pd.isna(peer_val):
            continue
        if col in lower_is_better:
            if firm_val < peer_val:
                comments.append(f"The company has lower {label} pressure than the peer median, which is a positive sign.")
            else:
                comments.append(f"The company has higher {label} pressure than the peer median, which is a caution flag.")
        else:
            if firm_val > peer_val:
                comments.append(f"The company is stronger than the peer median on {label}.")
            else:
                comments.append(f"The company is weaker than the peer median on {label}.")
    return comments


def _pick_first_available(df: pd.DataFrame, labels: List[str]) -> Optional[float]:
    for label in labels:
        if label in df.index:
            value = df.loc[label]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def fetch_live_features_from_yfinance(ticker: str) -> Dict[str, object]:
    import yfinance as yf

    t = yf.Ticker(ticker)
    income = t.financials
    balance = t.balance_sheet
    cashflow = t.cashflow

    if income is None or income.empty or balance is None or balance.empty:
        raise ValueError("yfinance did not return enough financial statement data for this ticker.")

    sale = _pick_first_available(income, ["Total Revenue", "Revenue"])
    ni = _pick_first_available(income, ["Net Income", "Net Income Common Stockholders"])
    at = _pick_first_available(balance, ["Total Assets"])
    act = _pick_first_available(balance, ["Current Assets", "Total Current Assets"])
    lct = _pick_first_available(balance, ["Current Liabilities", "Total Current Liabilities"])
    dlc = _pick_first_available(balance, ["Current Debt", "Current Debt And Capital Lease Obligation", "Current Debt & Capital Lease Obligation"])
    dltt = _pick_first_available(balance, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt & Capital Lease Obligation"])
    total_debt = _pick_first_available(balance, ["Total Debt"])
    oancf = _pick_first_available(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])

    if dlc is None and total_debt is not None:
        dlc = 0.0
    if dltt is None and total_debt is not None:
        dltt = total_debt

    sale_prev = None
    if income is not None and not income.empty and income.shape[1] >= 2:
        prev_col = income.columns[1]
        sale_prev = _pick_first_available(income[[prev_col]], ["Total Revenue", "Revenue"])

    live = {
        "tic": ticker,
        "conm": ticker,
        "sic_final": np.nan,
        "revenue_growth": np.nan if sale is None or sale_prev in [None, 0] else (sale - sale_prev) / sale_prev,
        "roa": np.nan if ni is None or at in [None, 0] else ni / at,
        "debt_ratio": np.nan if at in [None, 0] else ((0 if dlc is None else dlc) + (0 if dltt is None else dltt)) / at,
        "cfo_assets": np.nan if oancf is None or at in [None, 0] else oancf / at,
        "current_ratio": np.nan if act is None or lct in [None, 0] else act / lct,
        "profit_margin": np.nan if ni is None or sale in [None, 0] else ni / sale,
    }
    return live
