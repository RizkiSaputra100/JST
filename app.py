import hashlib
import html
import io
import os
import random
import re
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from local_translator import translate_column_name, translate_value_label


RANDOM_STATE = 42
TARGET_COLUMN = "Satisfaction Score"
DATA_DIR = Path("data")
MAX_CATEGORY_LEVELS = 30
MAX_CATEGORICAL_UNIQUE = 80
MAX_CATEGORICAL_UNIQUE_RATIO = 0.35
DEFAULT_MAX_TRAINING_ROWS = 30000

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)



st.set_page_config(
    page_title="Dashboard JST - Kepuasan Pelanggan",
    page_icon="JST",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --line: #d8dee8;
        --panel: #f9fbff;
        --soft: #e7eef8;
        --page: #e8eff7;
        --field: #f5f8fc;
        --brand: #2f6fda;
        --brand-dark: #1f55b5;
        --accent-teal: #0f9f9a;
        --accent-amber: #d99021;
        --accent-rose: #d1495b;
        --accent-violet: #7c3aed;
        --ok: #16a34a;
    }
    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #e6eef7 0%, #edf3f8 42%, #e8eff7 100%);
        color: var(--ink);
    }
    section[data-testid="stSidebar"] > div {
        background: #e2ebf5;
        color: var(--ink);
        border-right: 1px solid #c9d5e4;
    }
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp div {
        border-color: inherit;
    }
    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.92);
        backdrop-filter: blur(8px);
    }
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    .hero {
        background:
            radial-gradient(circle at 88% 18%, rgba(15, 159, 154, 0.24) 0, rgba(15, 159, 154, 0.05) 28%, transparent 42%),
            linear-gradient(135deg, #16315f 0%, #2458aa 52%, #0f8f91 100%);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 8px;
        padding: 30px 32px;
        margin-bottom: 18px;
        box-shadow: 0 18px 40px rgba(23, 49, 95, 0.18);
        position: relative;
        overflow: hidden;
    }
    .hero h1 {
        margin: 8px 0 10px 0;
        font-size: 2.18rem;
        line-height: 1.2;
        letter-spacing: 0;
        color: #ffffff;
        max-width: 880px;
    }
    .hero p {
        margin: 0;
        color: #dce8fb;
        font-size: 1rem;
        max-width: 880px;
    }
    .hero-kicker {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        color: #eaf2ff;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0;
    }
    .hero-grid {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }
    .hero-chip {
        color: #f8fbff;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 0.84rem;
        font-weight: 700;
    }
    .section-note {
        border-left: 5px solid var(--accent-teal);
        background: var(--panel);
        border-radius: 6px;
        padding: 12px 14px;
        color: #344054;
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        margin: 8px 0 16px 0;
        box-shadow: 0 4px 14px rgba(23, 32, 51, 0.04);
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #475467 !important;
        opacity: 1 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700;
    }
    div[data-testid="stMetric"] * {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    .metric-card {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(245,248,252,0.96) 100%);
        border: 1px solid var(--line);
        border-left: 5px solid var(--brand);
        border-radius: 8px;
        padding: 16px 18px 17px 18px;
        min-height: 126px;
        box-shadow: 0 6px 18px rgba(23, 32, 51, 0.05);
    }
    div[data-testid="column"]:nth-of-type(2) .metric-card {
        border-left-color: var(--accent-teal);
    }
    div[data-testid="column"]:nth-of-type(3) .metric-card {
        border-left-color: var(--accent-amber);
    }
    div[data-testid="column"]:nth-of-type(4) .metric-card {
        border-left-color: var(--accent-rose);
    }
    .metric-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .metric-icon {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-weight: 900;
        background: var(--brand);
        box-shadow: 0 8px 16px rgba(47, 111, 218, 0.18);
    }
    .metric-icon.teal { background: var(--accent-teal); box-shadow: 0 8px 16px rgba(15, 159, 154, 0.16); }
    .metric-icon.amber { background: var(--accent-amber); box-shadow: 0 8px 16px rgba(217, 144, 33, 0.16); }
    .metric-icon.rose { background: var(--accent-rose); box-shadow: 0 8px 16px rgba(209, 73, 91, 0.16); }
    .metric-badge {
        border-radius: 999px;
        padding: 4px 8px;
        background: #eaf1ff;
        color: var(--brand-dark);
        font-size: 0.74rem;
        font-weight: 800;
    }
    .metric-card .metric-label {
        color: #475467;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .metric-card .metric-value {
        color: #172033;
        font-size: 1.85rem;
        line-height: 1.1;
        font-weight: 800;
        overflow-wrap: anywhere;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        background: var(--panel);
        box-shadow: 0 4px 14px rgba(23, 32, 51, 0.04);
    }
    .stButton > button, .stDownloadButton > button {
        background: var(--field) !important;
        color: #172033 !important;
        border: 1px solid #cfd8e3 !important;
        border-radius: 6px;
        font-weight: 600;
        min-height: 44px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--brand) !important;
        color: var(--brand-dark) !important;
        background: #edf5ff !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--brand) !important;
        color: #ffffff !important;
        border-color: var(--brand) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--brand-dark) !important;
        color: #ffffff !important;
        border-color: var(--brand-dark) !important;
    }
    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 4px solid var(--accent-teal);
        border-radius: 8px;
        padding: 18px 18px 8px 18px;
        box-shadow: 0 8px 22px rgba(23, 32, 51, 0.06);
    }
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] label p,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p {
        color: #344054 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }
    div[data-baseweb="select"] > div {
        background: var(--field) !important;
        border-color: #cfd8e3 !important;
        color: #172033 !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] input {
        background: var(--field) !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        border-color: #cfd8e3 !important;
        opacity: 1 !important;
    }
    div[data-testid="stNumberInput"] button {
        background: #edf2f7 !important;
        color: #344054 !important;
        border-color: #cfd8e3 !important;
    }
    div[data-testid="stNumberInput"] button svg {
        color: #344054 !important;
        fill: #344054 !important;
        stroke: #344054 !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="radio"] *,
    [data-testid="stNumberInput"] input {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--brand) !important;
        box-shadow: 0 0 0 1px var(--brand) !important;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--brand) !important;
        color: #ffffff !important;
        border: 1px solid var(--brand) !important;
        border-radius: 6px !important;
        min-height: 44px;
        font-weight: 700;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--brand-dark) !important;
        border-color: var(--brand-dark) !important;
        color: #ffffff !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] * {
        color: #172033 !important;
        opacity: 1 !important;
    }
    [data-testid="stTabs"] button {
        color: #344054 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 14px !important;
    }
    [data-testid="stTabs"] button p {
        color: #344054 !important;
        opacity: 1 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        background: #dceafe !important;
        border: 1px solid #b8cef0 !important;
        border-bottom-color: transparent !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--brand-dark) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--brand) !important;
    }
    div[data-testid="stAlert"] {
        color: #172033 !important;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
    }
    div[data-testid="stAlert"] * {
        color: #172033 !important;
        opacity: 1 !important;
    }
    .train-panel {
        background:
            linear-gradient(135deg, rgba(47, 111, 218, 0.08), rgba(15, 159, 154, 0.06)),
            var(--panel);
        border: 1px solid #c8d7ea;
        border-left: 5px solid var(--brand);
        border-radius: 8px;
        padding: 18px 20px;
        margin: 16px 0;
        box-shadow: 0 6px 18px rgba(23, 32, 51, 0.05);
    }
    .train-panel strong {
        color: var(--ink);
    }
    .train-grid {
        display: grid;
        grid-template-columns: 1.2fr 1fr;
        gap: 10px 18px;
        align-items: start;
    }
    .train-title {
        font-size: 1rem;
        color: #13213a;
        font-weight: 850;
        margin-bottom: 4px;
    }
    .train-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
    }
    .train-chip {
        background: #ffffff;
        border: 1px solid #cfd8e3;
        color: #344054;
        border-radius: 999px;
        padding: 6px 9px;
        font-size: 0.78rem;
        font-weight: 800;
        max-width: 100%;
        overflow-wrap: anywhere;
    }
    .workflow-steps {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 18px 0;
    }
    .workflow-step {
        background: rgba(255,255,255,0.72);
        border: 1px solid #d5dfeb;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(23,32,51,0.035);
    }
    .workflow-step span {
        display: inline-flex;
        width: 26px;
        height: 26px;
        border-radius: 7px;
        background: #dbeafe;
        color: var(--brand-dark);
        align-items: center;
        justify-content: center;
        font-weight: 900;
        margin-bottom: 8px;
    }
    .workflow-step strong {
        display: block;
        color: #172033;
        font-size: 0.9rem;
        margin-bottom: 3px;
    }
    .workflow-step p {
        margin: 0;
        color: #667085;
        font-size: 0.8rem;
        line-height: 1.35;
    }
    .upload-panel {
        background:
            linear-gradient(135deg, rgba(47, 111, 218, 0.08), rgba(124, 58, 237, 0.06)),
            #f9fbff;
        border: 1px dashed #9fb8df;
        border-left: 5px solid var(--accent-violet);
        border-radius: 8px;
        padding: 14px 16px;
        margin: 6px 0 10px 0;
        box-shadow: 0 5px 16px rgba(23, 32, 51, 0.04);
    }
    .upload-panel strong {
        color: #172033;
        font-size: 1rem;
    }
    .upload-panel p {
        color: #667085;
        margin: 4px 0 0 0;
        font-size: 0.88rem;
    }
    @media (max-width: 900px) {
        .workflow-steps,
        .train-grid {
            grid-template-columns: 1fr;
        }
        .train-meta {
            justify-content: flex-start;
        }
    }
    .status-pill {
        min-height: 44px;
        border-radius: 6px;
        border: 1px solid #cfd8e3;
        background: var(--field);
        color: #172033;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 14px;
        font-weight: 700;
        text-align: center;
    }
    .status-pill.ready {
        border-color: #bbf7d0;
        background: #f0fdf4;
        color: #166534;
    }
    .status-pill.auto {
        border-color: #bfdbfe;
        background: #eff6ff;
        color: #1d4ed8;
    }
    .status-pill.waiting {
        border-color: #fde68a;
        background: #fffbeb;
        color: #92400e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_column_name(column_name):
    return str(column_name).strip().lower().replace("_", " ")


def translate_column(column_name):
    return translate_column_name(column_name)


def translate_value(value):
    if pd.isna(value):
        return value
    return translate_value_label(value)


def translate_dataframe_for_display(df):
    display_df = df.copy()
    display_df.columns = [translate_column(column) for column in display_df.columns]
    object_columns = display_df.select_dtypes(include=["object", "string", "category"]).columns
    for column in object_columns:
        display_df[column] = display_df[column].map(translate_value)
    return display_df


def metric_card(label, value, icon="•", tone="blue", badge="Ringkasan"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <div class="metric-icon {html.escape(str(tone))}">{html.escape(str(icon))}</div>
                <div class="metric-badge">{html.escape(str(badge))}</div>
            </div>
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(text, status):
    st.markdown(
        f'<div class="status-pill {status}">{html.escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


def build_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=MAX_CATEGORY_LEVELS,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def find_target_column(columns):
    candidates = [
        TARGET_COLUMN,
        "Satisfaction",
        "Customer Satisfaction",
        "Satisfaction Level",
        "Rating",
        "Score",
    ]
    normalized_map = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]

    satisfaction_columns = [
        column
        for column in columns
        if "satisf" in normalize_column_name(column)
    ]
    if satisfaction_columns:
        for column in satisfaction_columns:
            normalized = normalize_column_name(column)
            if "overall" in normalized or "experience" in normalized:
                return column
        return satisfaction_columns[0]

    rating_columns = [
        column
        for column in columns
        if any(token in normalize_column_name(column) for token in ["rating", "score", "1-5", "1 ="])
    ]
    return rating_columns[0] if rating_columns else None
    return None


def find_id_columns(columns):
    id_columns = []
    for column in columns:
        normalized = normalize_column_name(column)
        tokens = normalized.replace("-", " ").replace("_", " ").split()
        compact = normalized.replace(" ", "").replace("_", "").replace("-", "")
        if (
            normalized in {"id", "customer", "customer id", "customerid", "unique id", "order id"}
            or compact in {"customerid", "uniqueid", "orderid"}
            or tokens[-1:] == ["id"]
        ):
            id_columns.append(column)
    return id_columns


def looks_like_datetime(column_name, series):
    if pd.api.types.is_numeric_dtype(series):
        return False

    normalized = normalize_column_name(column_name)
    if any(token in normalized for token in ["date", "time", "reported", "responded", "created", "updated"]):
        return True

    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(200)
    if sample.empty:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() >= 0.75


def engineer_datetime_features(X):
    result = X.copy()
    datetime_columns = []

    for column in X.columns:
        if looks_like_datetime(column, X[column]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(X[column], errors="coerce", dayfirst=True)
            if parsed.notna().sum() == 0:
                continue

            safe_name = translate_column(column)
            result[f"{safe_name} - Bulan"] = parsed.dt.month.astype("float32")
            result[f"{safe_name} - Hari"] = parsed.dt.day.astype("float32")
            result[f"{safe_name} - Jam"] = parsed.dt.hour.astype("float32")
            result[f"{safe_name} - Hari Minggu"] = parsed.dt.dayofweek.astype("float32")
            datetime_columns.append(column)

    return result.drop(columns=datetime_columns, errors="ignore"), datetime_columns


def find_high_cardinality_columns(X):
    high_cardinality_columns = []
    row_count = max(len(X), 1)

    for column in X.select_dtypes(include=["object", "string", "category"]).columns:
        normalized = normalize_column_name(column)
        tokens = normalized.replace("-", " ").replace("_", " ").split()
        unique_count = X[column].nunique(dropna=True)
        unique_ratio = unique_count / row_count
        average_length = X[column].dropna().astype(str).str.len().mean()

        is_free_text = average_length and average_length > 35
        is_name_or_remark = (
            any(token in normalized for token in ["remark", "comment", "description"])
            or ("name" in tokens and unique_count > 20)
        )
        is_too_unique = unique_count > MAX_CATEGORICAL_UNIQUE or unique_ratio > MAX_CATEGORICAL_UNIQUE_RATIO

        if is_free_text or is_name_or_remark or is_too_unique:
            high_cardinality_columns.append(column)

    return high_cardinality_columns


def transform_satisfaction_target(target_series):
    numeric_target = pd.to_numeric(target_series, errors="coerce")
    non_missing_values = numeric_target.dropna()

    if not non_missing_values.empty and non_missing_values.between(1, 5).all():
        return pd.cut(
            numeric_target,
            bins=[-np.inf, 2, 3, np.inf],
            labels=["Tidak Puas", "Netral", "Puas"],
        ).astype(str)

    return target_series.astype(str).str.strip().map(translate_value)


def prepare_dataset(df, selected_target_column, max_training_rows=None):
    target_column = selected_target_column
    id_columns = [column for column in find_id_columns(df.columns) if column != target_column]
    cleaned_df = df.drop(columns=id_columns, errors="ignore").copy()

    y_raw = transform_satisfaction_target(cleaned_df[target_column])
    X = cleaned_df.drop(columns=[target_column])

    valid_rows = y_raw.notna() & (y_raw.astype(str).str.lower() != "nan") & (y_raw.astype(str).str.strip() != "")
    X = X.loc[valid_rows].reset_index(drop=True)
    y_raw = y_raw.loc[valid_rows].reset_index(drop=True)

    if max_training_rows and len(X) > max_training_rows:
        sampled_index = X.sample(n=max_training_rows, random_state=RANDOM_STATE).index.sort_values()
        X = X.loc[sampled_index].reset_index(drop=True)
        y_raw = y_raw.loc[sampled_index].reset_index(drop=True)

    X, datetime_columns = engineer_datetime_features(X)
    high_cardinality_columns = find_high_cardinality_columns(X)
    X = X.drop(columns=high_cardinality_columns, errors="ignore")

    if X.empty:
        raise ValueError("Dataset tidak memiliki fitur training setelah kolom target dan ID dipisahkan.")
    if len(X) < 10:
        raise ValueError("Dataset terlalu sedikit untuk training. Gunakan minimal 10 baris data yang valid.")
    if y_raw.nunique() < 2:
        raise ValueError("Target hanya memiliki satu kelas. Model klasifikasi membutuhkan minimal dua kelas.")
    if y_raw.nunique() > 20:
        raise ValueError("Target memiliki terlalu banyak kelas. Pilih target kepuasan, rating 1-5, atau label kategori.")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Tidak Diketahui")),
                        ("encoder", build_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return {
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "target_column": target_column,
        "id_columns": id_columns,
        "datetime_columns": datetime_columns,
        "dropped_high_cardinality_columns": high_cardinality_columns,
        "training_rows_used": len(X),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
    }


def build_keras_model(input_dim, class_count):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def report_to_row(model_name, y_true, y_pred, labels):
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "Metode": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
    }


def get_local_datasets():
    DATA_DIR.mkdir(exist_ok=True)
    return sorted(DATA_DIR.glob("*.csv"))


def safe_dataset_filename(filename):
    base_name = Path(filename).name
    stem = Path(base_name).stem.strip() or "dataset"
    suffix = Path(base_name).suffix.lower()
    if suffix != ".csv":
        suffix = ".csv"

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not safe_stem:
        safe_stem = "dataset"
    return f"{safe_stem}{suffix}"


def save_uploaded_dataset(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    pd.read_csv(io.BytesIO(raw_bytes), nrows=5)

    filename = safe_dataset_filename(uploaded_file.name)
    target_path = DATA_DIR / filename
    target_path.write_bytes(raw_bytes)
    read_csv_path.clear()
    read_all_csv_paths.clear()
    return target_path


def handle_dataset_upload(uploaded_file, container, key_prefix):
    if uploaded_file is None:
        return

    upload_signature = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    saved_key = f"{key_prefix}:{uploaded_file.name}:{upload_signature}"
    if st.session_state.get("saved_upload_key") == saved_key:
        return

    try:
        saved_path = save_uploaded_dataset(uploaded_file)
        st.session_state["saved_upload_key"] = saved_key
        container.success(f"Dataset disimpan ke folder data/: {saved_path.name}")
        st.rerun()
    except Exception as exc:
        container.error(f"Upload gagal: {exc}")


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw_bytes):
    return pd.read_csv(io.BytesIO(raw_bytes))


@st.cache_data(show_spinner=False)
def read_csv_path(path_string):
    return pd.read_csv(path_string)


@st.cache_data(show_spinner=False)
def read_all_csv_paths(path_strings):
    frames = []
    summaries = []
    errors = []

    for path_string in path_strings:
        path = Path(path_string)
        try:
            frame = pd.read_csv(path_string)
            frames.append((path.name, frame))
            summaries.append(
                {
                    "File": path.name,
                    "Baris": frame.shape[0],
                    "Kolom": frame.shape[1],
                    "Missing Value": int(frame.isna().sum().sum()),
                    "Status": "Terbaca",
                }
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            summaries.append(
                {
                    "File": path.name,
                    "Baris": 0,
                    "Kolom": 0,
                    "Missing Value": 0,
                    "Status": f"Gagal: {exc}",
                }
            )

    return frames, pd.DataFrame(summaries), errors


def combine_compatible_frames(frames):
    if not frames:
        raise ValueError("Tidak ada dataset CSV yang berhasil dibaca.")

    base_columns = list(frames[0][1].columns)
    incompatible = [
        name
        for name, frame in frames
        if list(frame.columns) != base_columns
    ]

    if incompatible:
        raise ValueError(
            "Tidak semua CSV memiliki struktur kolom yang sama. "
            "Pilih satu dataset saja atau samakan nama dan urutan kolom pada file: "
            + ", ".join(incompatible)
        )

    combined_frames = []
    for name, frame in frames:
        copy_frame = frame.copy()
        copy_frame["Sumber Dataset"] = name
        combined_frames.append(copy_frame)

    return pd.concat(combined_frames, ignore_index=True)


def make_signature(source_name, df, target_column):
    raw = f"{source_name}:{df.shape}:{','.join(map(str, df.columns))}:{target_column}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def plot_keras_history(history, metric_name, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(history.history[metric_name], label=f"Training {ylabel}", linewidth=2)
    validation_metric = f"val_{metric_name}"
    if validation_metric in history.history:
        ax.plot(history.history[validation_metric], label=f"Validation {ylabel}", linewidth=2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        linecolor="#e5e7eb",
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def plot_numeric_correlation(df, target_column, id_columns):
    numeric_df = df.drop(columns=id_columns + [target_column], errors="ignore").select_dtypes(include=["number"])
    numeric_df = numeric_df.rename(columns={column: translate_column(column) for column in numeric_df.columns})

    if numeric_df.shape[1] < 2:
        st.info("Heatmap korelasi membutuhkan minimal dua fitur numerik.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.3, ax=ax)
    ax.set_title("Korelasi Fitur Numerik", fontsize=12, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def make_evaluation_csv(comparison_df, keras_report_df, mlp_report_df):
    output = io.StringIO()
    output.write("Tabel Komparasi\n")
    comparison_df.to_csv(output, index=False)
    output.write("\nClassification Report Keras Sequential\n")
    keras_report_df.to_csv(output)
    output.write("\nClassification Report MLPClassifier\n")
    mlp_report_df.to_csv(output)
    return output.getvalue().encode("utf-8")


def render_dataset_source():
    st.sidebar.header("Sumber Data")

    st.markdown(
        """
        <div class="upload-panel">
            <div>
                <strong>Upload Dataset Baru</strong>
                <p>File CSV akan langsung disimpan ke folder <b>data/</b> dan muncul sebagai pilihan dataset.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    main_uploaded_file = st.file_uploader(
        "Upload dan Simpan Dataset CSV",
        type=["csv"],
        key="main_dataset_upload",
        help="File yang diupload akan disimpan ke folder data/.",
    )
    handle_dataset_upload(main_uploaded_file, st, "main")

    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset CSV",
        type=["csv"],
        key="sidebar_dataset_upload",
        help="File yang diupload akan disimpan ke folder data/ lalu muncul sebagai pilihan dataset.",
    )
    handle_dataset_upload(uploaded_file, st.sidebar, "sidebar")

    local_datasets = get_local_datasets()

    if local_datasets:
        frames, inventory_df, errors = read_all_csv_paths([str(path) for path in local_datasets])
        st.sidebar.caption(f"{len(local_datasets)} file CSV terdeteksi di folder data/.")

        mode = st.sidebar.radio(
            "Mode pembacaan dataset",
            options=["Pilih satu dataset", "Gabungkan semua CSV"],
            help="Mode gabungan hanya bisa digunakan jika semua CSV memiliki nama dan urutan kolom yang sama.",
        )

        with st.sidebar.expander("Daftar dataset terbaca", expanded=False):
            st.dataframe(inventory_df, width="stretch", hide_index=True)
            if errors:
                st.warning("Beberapa file gagal dibaca. Lihat kolom Status pada tabel.")

        if mode == "Gabungkan semua CSV":
            try:
                combined_df = combine_compatible_frames(frames)
                return combined_df, f"Gabungan {len(frames)} CSV", inventory_df
            except ValueError as exc:
                st.sidebar.error(str(exc))
                st.warning(str(exc))
                st.stop()

        selected_path = st.sidebar.selectbox(
            "Pilih dataset aktif",
            options=[Path(name) for name, _ in frames],
            format_func=lambda path: path.name,
        )
        selected_name = selected_path.name
        selected_df = next(frame for name, frame in frames if name == selected_name)
        return selected_df, selected_name, inventory_df

    st.sidebar.warning("Belum ada CSV di folder data/.")
    st.info("Upload dataset CSV dari sidebar atau letakkan file CSV langsung di folder `data/`, lalu refresh aplikasi.")
    st.stop()


def render_dataset_overview(df):
    info_df = pd.DataFrame(
        {
            "Kolom Asli": df.columns,
            "Nama Indonesia": [translate_column(column) for column in df.columns],
            "Tipe Data": [str(dtype) for dtype in df.dtypes],
            "Missing Value": df.isna().sum().values,
        }
    )

    st.markdown("#### Preview Dataset")
    st.dataframe(translate_dataframe_for_display(df.head(12)), width="stretch", hide_index=True)

    st.markdown("#### Struktur dan Kualitas Data")
    st.dataframe(info_df, width="stretch", hide_index=True)


def train_models(prepared, epochs, batch_size, mlp_max_iter):
    class_counts = np.bincount(prepared["y"])
    test_count = int(np.ceil(len(prepared["y"]) * 0.2))
    train_count = len(prepared["y"]) - test_count
    can_stratify = class_counts.min() >= 2 and test_count >= len(class_counts) and train_count >= len(class_counts)

    X_train, X_test, y_train, y_test = train_test_split(
        prepared["X"],
        prepared["y"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=prepared["y"] if can_stratify else None,
    )

    X_train_processed = prepared["preprocessor"].fit_transform(X_train).astype(np.float32)
    X_test_processed = prepared["preprocessor"].transform(X_test).astype(np.float32)

    class_labels = prepared["label_encoder"].classes_.tolist()
    display_class_labels = [translate_value(label) for label in class_labels]
    class_count = len(class_labels)

    y_train_categorical = to_categorical(y_train, num_classes=class_count)
    keras_model = build_keras_model(X_train_processed.shape[1], class_count)
    history = keras_model.fit(
        X_train_processed,
        y_train_categorical,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
    )

    keras_probabilities = keras_model.predict(X_test_processed, verbose=0)
    keras_predictions = np.argmax(keras_probabilities, axis=1)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=mlp_max_iter,
        random_state=RANDOM_STATE,
    )
    mlp_model.fit(X_train_processed, y_train)
    mlp_predictions = mlp_model.predict(X_test_processed)

    comparison_df = pd.DataFrame(
        [
            report_to_row("TensorFlow/Keras Sequential", y_test, keras_predictions, display_class_labels),
            report_to_row("Scikit-Learn MLPClassifier", y_test, mlp_predictions, display_class_labels),
        ]
    )
    keras_report_df = pd.DataFrame(
        classification_report(
            y_test,
            keras_predictions,
            labels=np.arange(len(display_class_labels)),
            target_names=display_class_labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    mlp_report_df = pd.DataFrame(
        classification_report(
            y_test,
            mlp_predictions,
            labels=np.arange(len(display_class_labels)),
            target_names=display_class_labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    return {
        "keras_model": keras_model,
        "mlp_model": mlp_model,
        "history": history,
        "comparison_df": comparison_df,
        "keras_report_df": keras_report_df,
        "mlp_report_df": mlp_report_df,
        "keras_predictions": keras_predictions,
        "mlp_predictions": mlp_predictions,
        "y_test": y_test,
        "display_class_labels": display_class_labels,
        "prepared": prepared,
    }


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">JST ANALYTICS DASHBOARD</div>
        <h1>Prediksi Kepuasan Pelanggan</h1>
        <p>
            Dashboard komparasi metode Jaringan Syaraf Tiruan untuk membaca dataset,
            menjalankan preprocessing otomatis, melatih TensorFlow/Keras Sequential dan
            Scikit-Learn MLPClassifier, lalu menyajikan evaluasi model secara ringkas.
        </p>
        <div class="hero-grid">
            <div class="hero-chip">TensorFlow/Keras Sequential</div>
            <div class="hero-chip">Scikit-Learn MLPClassifier</div>
            <div class="hero-chip">Preprocessing Otomatis</div>
            <div class="hero-chip">Prediksi Pelanggan Baru</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Pengaturan Model")
    epochs = st.slider("Epoch Keras", min_value=10, max_value=150, value=40, step=10)
    batch_size = st.selectbox("Batch Size Keras", options=[16, 32, 64, 128], index=1)
    mlp_max_iter = st.slider("Max Iteration MLPClassifier", min_value=100, max_value=1000, value=400, step=100)
    max_training_rows = st.slider(
        "Maksimal Baris Training",
        min_value=5000,
        max_value=100000,
        value=DEFAULT_MAX_TRAINING_ROWS,
        step=5000,
        help="Batas aman agar Docker tidak kehabisan memori saat dataset besar. Jika dataset lebih kecil, semua baris tetap dipakai.",
    )
    auto_train = st.toggle("Training otomatis", value=False, help="Jika aktif, model langsung dilatih saat data/target berubah.")


df, source_name, inventory_df = render_dataset_source()
detected_target_column = find_target_column(df.columns)

top1, top2, top3, top4 = st.columns(4)
with top1:
    metric_card("Sumber Data", source_name, icon="D", tone="blue", badge="CSV")
with top2:
    metric_card("Jumlah Baris", f"{df.shape[0]:,}", icon="R", tone="teal", badge="Records")
with top3:
    metric_card("Jumlah Kolom", f"{df.shape[1]:,}", icon="C", tone="amber", badge="Features")
with top4:
    metric_card("Missing Value", f"{int(df.isna().sum().sum()):,}", icon="M", tone="rose", badge="Quality")

st.markdown(
    """
    <div class="workflow-steps">
        <div class="workflow-step">
            <span>1</span>
            <strong>Pilih Data</strong>
            <p>Dataset dibaca dari folder data dan target kepuasan dideteksi otomatis.</p>
        </div>
        <div class="workflow-step">
            <span>2</span>
            <strong>Preprocessing</strong>
            <p>ID dihapus, missing value ditangani, kategori di-encode, fitur dinormalisasi.</p>
        </div>
        <div class="workflow-step">
            <span>3</span>
            <strong>Training JST</strong>
            <p>Keras Sequential dan MLPClassifier dilatih dengan pembagian data 80:20.</p>
        </div>
        <div class="workflow-step">
            <span>4</span>
            <strong>Evaluasi</strong>
            <p>Accuracy, precision, recall, F1-score, grafik, dan confusion matrix dibandingkan.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if detected_target_column is None:
    st.error(
        "Kolom target kepuasan tidak ditemukan. Pastikan dataset memiliki kolom seperti "
        "`Satisfaction Score`, `CSAT Score`, `Satisfaction`, `Rating`, atau kolom skor kepuasan 1-5."
    )
    st.stop()

target_options = df.columns.tolist()
default_target_index = target_options.index(detected_target_column)
selected_target_column = st.selectbox(
    "Kolom target prediksi",
    options=target_options,
    index=default_target_index,
    format_func=lambda column: f"{translate_column(column)} ({column})",
    help="Pilih kolom kepuasan pelanggan. Jika skor 1-5, aplikasi otomatis mengubahnya menjadi Tidak Puas, Netral, dan Puas.",
)

try:
    prepared = prepare_dataset(df, selected_target_column, max_training_rows)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

signature = make_signature(source_name, df, selected_target_column) + f":{max_training_rows}"
if st.session_state.get("trained_signature") != signature:
    st.session_state.pop("trained", None)

st.markdown(
    f"""
    <div class="train-panel">
        <div class="train-grid">
            <div>
                <div class="train-title">Control Panel Training Model</div>
                <span class="small-muted">
                    Klik tombol training untuk menjalankan Keras Sequential dan MLPClassifier.
                    Hasil evaluasi akan muncul pada tab Evaluasi Model dan Visualisasi.
                </span>
            </div>
            <div class="train-meta">
                <div class="train-chip">Dataset: {html.escape(str(source_name))}</div>
                <div class="train-chip">Target: {html.escape(str(translate_column(selected_target_column)))}</div>
                <div class="train-chip">Limit: {max_training_rows:,} baris</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

action_col1, action_col2, action_col3 = st.columns([1.25, 1, 1])
with action_col1:
    train_requested = st.button("Latih Model Sekarang", type="primary", width="stretch")
with action_col2:
    retrain_requested = st.button("Training Ulang", width="stretch")
with action_col3:
    if "trained" in st.session_state:
        status_pill("Model sudah dilatih", "ready")
    elif auto_train:
        status_pill("Training otomatis aktif", "auto")
    else:
        status_pill("Model belum dilatih", "waiting")

should_train = train_requested or retrain_requested or (auto_train and "trained" not in st.session_state)
if should_train:
    with st.spinner("Melatih Keras Sequential dan MLPClassifier. Mohon tunggu..."):
        try:
            st.session_state["trained"] = train_models(prepared, epochs, batch_size, mlp_max_iter)
            st.session_state["trained_signature"] = signature
            st.rerun()
        except Exception as exc:
            st.error(f"Training gagal: {exc}")
            st.stop()

tab_data, tab_prep, tab_eval, tab_visual, tab_predict = st.tabs(
    ["Data", "Preprocessing", "Evaluasi Model", "Visualisasi", "Prediksi Baru"]
)

with tab_data:
    st.markdown("#### Inventaris Dataset di Folder Data")
    st.dataframe(inventory_df, width="stretch", hide_index=True)
    render_dataset_overview(df)

with tab_prep:
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        metric_card("Target", translate_column(prepared["target_column"]))
    with p2:
        metric_card("Kolom ID Dihapus", len(prepared["id_columns"]))
    with p3:
        metric_card("Fitur Numerik", len(prepared["numeric_columns"]))
    with p4:
        metric_card("Fitur Kategorikal", len(prepared["categorical_columns"]))

    st.caption(f"Baris yang digunakan untuk training: {prepared['training_rows_used']:,}")

    st.markdown("#### Distribusi Target Setelah Transformasi")
    target_distribution_df = prepared["y_raw"].value_counts().rename_axis("Kelas").reset_index(name="Jumlah")
    target_distribution_df["Kelas"] = target_distribution_df["Kelas"].map(translate_value)
    st.dataframe(target_distribution_df, width="stretch", hide_index=True)

    with st.expander("Detail proses preprocessing yang dilakukan aplikasi", expanded=True):
        st.write(
            """
            1. Kolom ID seperti Customer ID, Unique ID, atau Order ID dihapus agar model tidak belajar dari identitas unik.
            2. Baris dengan target kosong dibuang karena tidak dapat digunakan sebagai label training.
            3. Target skor 1-5 diubah menjadi tiga kategori: Tidak Puas, Netral, dan Puas.
            4. Kolom tanggal/jam diubah menjadi fitur numerik bulan, hari, jam, dan hari dalam minggu.
            5. Kolom teks bebas, nama, ID terselubung, atau kategori yang terlalu unik tidak dipakai untuk training.
            6. Fitur numerik diisi nilai median dan dinormalisasi menggunakan StandardScaler.
            7. Fitur kategorikal diisi nilai "Tidak Diketahui" lalu diubah menjadi angka dengan One-Hot Encoder.
            8. Data dibagi menjadi 80% training dan 20% testing dengan random_state=42.
            """
        )

    if prepared["datetime_columns"]:
        st.markdown("#### Kolom Tanggal/Jam yang Diubah")
        st.dataframe(
            pd.DataFrame({"Kolom": prepared["datetime_columns"], "Perlakuan": "Diubah menjadi fitur waktu numerik"}),
            width="stretch",
            hide_index=True,
        )

    if prepared["dropped_high_cardinality_columns"]:
        st.markdown("#### Kolom yang Tidak Dipakai untuk Training")
        st.dataframe(
            pd.DataFrame(
                {
                    "Kolom": prepared["dropped_high_cardinality_columns"],
                    "Alasan": "Teks bebas, nama, ID terselubung, atau kategori terlalu banyak",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with tab_eval:
    if "trained" not in st.session_state:
        st.info("Klik tombol Latih Model Sekarang di bagian atas halaman untuk menampilkan hasil evaluasi.")
    else:
        trained = st.session_state["trained"]
        comparison = trained["comparison_df"]
        best_row = comparison.sort_values("F1-Score", ascending=False).iloc[0]

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            metric_card("Model Terbaik", best_row["Metode"])
        with e2:
            metric_card("Accuracy", f"{best_row['Accuracy']:.4f}")
        with e3:
            metric_card("Precision", f"{best_row['Precision']:.4f}")
        with e4:
            metric_card("F1-Score", f"{best_row['F1-Score']:.4f}")

        st.markdown("#### Tabel Komparasi")
        st.dataframe(
            comparison.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Classification Report Keras")
            st.dataframe(trained["keras_report_df"].style.format("{:.4f}"), width="stretch")
        with c2:
            st.markdown("#### Classification Report MLPClassifier")
            st.dataframe(trained["mlp_report_df"].style.format("{:.4f}"), width="stretch")

        st.download_button(
            "Download Hasil Evaluasi CSV",
            data=make_evaluation_csv(comparison, trained["keras_report_df"], trained["mlp_report_df"]),
            file_name="hasil_evaluasi_komparasi_jst.csv",
            mime="text/csv",
        )

with tab_visual:
    if "trained" not in st.session_state:
        st.info("Visualisasi evaluasi akan muncul setelah model dilatih.")
    else:
        trained = st.session_state["trained"]
        chart_tabs = st.tabs(["Training Keras", "Confusion Matrix", "Korelasi Numerik"])
        with chart_tabs[0]:
            st.markdown('<p class="small-muted">Grafik ini membantu melihat apakah model Keras belajar stabil atau mulai overfitting.</p>', unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                plot_keras_history(trained["history"], "accuracy", "Training dan Validation Accuracy", "Accuracy")
            with g2:
                plot_keras_history(trained["history"], "loss", "Training dan Validation Loss", "Loss")
        with chart_tabs[1]:
            st.markdown('<p class="small-muted">Nilai diagonal menunjukkan prediksi benar. Nilai di luar diagonal menunjukkan kelas yang tertukar.</p>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                plot_confusion_matrix(trained["y_test"], trained["keras_predictions"], trained["display_class_labels"], "Keras Sequential")
            with m2:
                plot_confusion_matrix(trained["y_test"], trained["mlp_predictions"], trained["display_class_labels"], "MLPClassifier")
        with chart_tabs[2]:
            st.markdown('<p class="small-muted">Heatmap korelasi digunakan untuk membaca hubungan linear antar fitur numerik.</p>', unsafe_allow_html=True)
            plot_numeric_correlation(df, trained["prepared"]["target_column"], trained["prepared"]["id_columns"])

with tab_predict:
    if "trained" not in st.session_state:
        st.info("Form prediksi akan aktif setelah model dilatih.")
    else:
        trained = st.session_state["trained"]
        st.markdown("#### Form Prediksi Pelanggan Baru")
        st.markdown('<p class="small-muted">Isi nilai fitur pelanggan, pilih model, lalu lihat prediksi tingkat kepuasannya.</p>', unsafe_allow_html=True)
        with st.form("new_customer_prediction_form"):
            form_values = {}
            input_columns = trained["prepared"]["X"].columns.tolist()
            columns = st.columns(2)
            for index, column in enumerate(input_columns):
                with columns[index % 2]:
                    if column in trained["prepared"]["numeric_columns"]:
                        median_value = pd.to_numeric(trained["prepared"]["X"][column], errors="coerce").median()
                        if pd.isna(median_value):
                            median_value = 0.0
                        form_values[column] = st.number_input(
                            translate_column(column),
                            value=float(median_value),
                            key=f"num_{column}",
                        )
                    else:
                        available_values = (
                            trained["prepared"]["X"][column]
                            .dropna()
                            .astype(str)
                            .sort_values()
                            .unique()
                            .tolist()
                        )
                        if not available_values:
                            available_values = [""]
                        form_values[column] = st.selectbox(
                            translate_column(column),
                            options=available_values,
                            format_func=translate_value,
                            key=f"cat_{column}",
                        )

            selected_model = st.radio(
                "Pilih model untuk prediksi",
                ["TensorFlow/Keras Sequential", "Scikit-Learn MLPClassifier"],
                horizontal=True,
            )
            submitted = st.form_submit_button("Prediksi Kepuasan")

        if submitted:
            new_customer_df = pd.DataFrame([form_values], columns=input_columns)
            new_customer_processed = trained["prepared"]["preprocessor"].transform(new_customer_df)

            if selected_model == "TensorFlow/Keras Sequential":
                probability = trained["keras_model"].predict(new_customer_processed, verbose=0)
                predicted_class_index = int(np.argmax(probability, axis=1)[0])
            else:
                predicted_class_index = int(trained["mlp_model"].predict(new_customer_processed)[0])

            predicted_label = translate_value(
                trained["prepared"]["label_encoder"].inverse_transform([predicted_class_index])[0]
            )
            st.success(f"Hasil prediksi kepuasan pelanggan: {predicted_label}")
