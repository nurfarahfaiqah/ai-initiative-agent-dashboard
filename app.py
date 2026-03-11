import json
import re
import difflib
from typing import Dict, Any, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil import parser


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="InsightPilot AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 InsightPilot AI")
st.caption(
    "Upload datasets → clean & normalize → analyze → generate interactive dashboard insights for leadership decisions"
)


# -----------------------------
# Constants
# -----------------------------
STANDARD_VALUE_MAP = {
    "status": {
        "close": "closed",
        "closed": "closed",
        "clsed": "closed",
        "complete": "closed",
        "completed": "closed",
        "resolved": "closed",
        "resolve": "closed",
        "done": "closed",
        "open": "open",
        "opened": "open",
        "reopen": "open",
        "reopened": "open",
        "new": "open",
        "in progress": "in_progress",
        "in-progress": "in_progress",
        "inprogress": "in_progress",
        "progressing": "in_progress",
        "ongoing": "in_progress",
        "wip": "in_progress",
        "working": "in_progress",
        "cancel": "cancelled",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "pending": "pending",
        "pendng": "pending",
        "awaiting": "pending",
        "on hold": "on_hold",
        "hold": "on_hold",
    },
    "priority": {
        "hi": "high",
        "high": "high",
        "hgh": "high",
        "urgent": "high",
        "critical": "high",
        "med": "medium",
        "medium": "medium",
        "mid": "medium",
        "normal": "medium",
        "low": "low",
        "lo": "low",
    },
    "yes_no": {
        "y": "yes",
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "n": "no",
        "no": "no",
        "false": "no",
        "0": "no",
    },
}

IMPORTANT_COLUMN_HINTS = {
    "status": "status",
    "ticket_status": "status",
    "case_status": "status",
    "sr_status": "status",
    "request_status": "status",
    "priority": "priority",
    "severity": "priority",
    "urgency": "priority",
    "churn_flag": "yes_no",
    "active_flag": "yes_no",
    "retained_flag": "yes_no",
    "yn": "yes_no",
}

DEFAULT_BUSINESS_GOAL = (
    "Identify customer pain points, derive initiative opportunities, recommend KPI/KRIs, "
    "and reduce analyst time spent on manual data cleaning, analysis, and brainstorming."
)

INSIGHT_SCHEMA_EXAMPLE = {
    "executive_problem_statement": "High-level business problem",
    "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
    "data_quality_limitations": ["Limitation 1", "Limitation 2"],
    "root_cause_hypotheses": ["Cause 1", "Cause 2", "Cause 3"],
    "initiative_opportunities": [
        {
            "initiative_name": "Initiative A",
            "issue_solved": "Problem area",
            "why_it_matters": "Why it matters",
            "expected_business_value": "Expected value",
            "effort_level": "medium",
        }
    ],
    "kpi_recommendations": [
        {
            "leading_kpi": "Leading KPI",
            "lagging_kpi": "Lagging KPI",
            "suggested_baseline": "Current baseline",
            "suggested_target": "Target",
            "why_it_matters": "Why it matters",
        }
    ],
    "dashboard_story": {
        "headline": "One-line summary of what the dashboard shows",
        "priority_focus": ["Focus 1", "Focus 2", "Focus 3"],
        "recommended_views": [
            "Trend over time",
            "Top categories",
            "Segment comparison",
            "Data quality summary"
        ]
    }
}


# -----------------------------
# Session state defaults
# -----------------------------
defaults = {
    "analysis_complete": False,
    "analysis_output": "",
    "generated_prompt": "",
    "cleaned_datasets": {},
    "cleaning_reports": {},
    "dataset_profiles": {},
    "dataset_category_insights": {},
    "dataset_numeric_summaries": {},
    "machine_findings": {},
    "join_key_report": {},
    "executive_json": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Cleaning helpers
# -----------------------------
def standardize_column_name(col: Any) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^\w\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    return col


def clean_text_value(x: Any):
    if pd.isna(x):
        return x
    if isinstance(x, str):
        x = x.strip()
        x = re.sub(r"\s+", " ", x)
        if x.lower() in {"n/a", "na", "null", "none", "nil", "blank", "missing", "nan", ""}:
            return np.nan
    return x


def normalize_for_match(x: Any):
    if pd.isna(x):
        return x
    x = str(x).strip().lower()
    x = re.sub(r"[_\-]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x


def detect_column_semantic_type(col_name: str) -> Optional[str]:
    col = col_name.lower()
    if col in IMPORTANT_COLUMN_HINTS:
        return IMPORTANT_COLUMN_HINTS[col]
    if any(k in col for k in ["status", "ticket_status", "case_status", "sr_status", "request_status"]):
        return "status"
    if any(k in col for k in ["priority", "severity", "urgency"]):
        return "priority"
    if any(k in col for k in ["active", "flag", "indicator", "yn", "yes_no", "churned", "retained"]):
        return "yes_no"
    return None


def standardize_categorical_values(
    df: pd.DataFrame,
    max_unique_ratio: float = 0.2,
    fuzzy_threshold: float = 0.88,
) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    standardization_report = {}

    for col in df.columns:
        if not (df[col].dtype == "object" or str(df[col].dtype) == "category"):
            continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        unique_values = non_null.astype(str).nunique()
        unique_ratio = unique_values / max(len(non_null), 1)
        if unique_ratio > max_unique_ratio:
            continue

        semantic_type = detect_column_semantic_type(col)
        original_series = df[col].copy()
        normalized = df[col].map(normalize_for_match)

        if semantic_type in STANDARD_VALUE_MAP:
            value_map = STANDARD_VALUE_MAP[semantic_type]
            df[col] = normalized.map(lambda x: value_map.get(x, x) if pd.notna(x) else x)
        else:
            df[col] = normalized

        value_counts = df[col].dropna().astype(str).value_counts()
        unique_cleaned = value_counts.index.tolist()

        canonical_values = []
        fuzzy_map = {}
        for val in unique_cleaned:
            if not canonical_values:
                canonical_values.append(val)
                fuzzy_map[val] = val
                continue
            best_match = difflib.get_close_matches(val, canonical_values, n=1, cutoff=fuzzy_threshold)
            if best_match:
                fuzzy_map[val] = best_match[0]
            else:
                canonical_values.append(val)
                fuzzy_map[val] = val

        df[col] = df[col].map(lambda x: fuzzy_map.get(str(x), x) if pd.notna(x) else x)

        changed_count = int(
            (original_series.astype(str).fillna("<<NA>>") != df[col].astype(str).fillna("<<NA>>")).sum()
        )
        if changed_count > 0:
            sample_before_after = pd.DataFrame(
                {
                    "before": original_series.astype(str).fillna("<<NA>>"),
                    "after": df[col].astype(str).fillna("<<NA>>"),
                }
            )
            sample_before_after = sample_before_after[
                sample_before_after["before"] != sample_before_after["after"]
            ].drop_duplicates().head(15)

            standardization_report[col] = {
                "semantic_type": semantic_type if semantic_type else "generic_categorical",
                "unique_before": int(original_series.dropna().astype(str).nunique()),
                "unique_after": int(df[col].dropna().astype(str).nunique()),
                "changed_rows": changed_count,
                "sample_mappings": dict(zip(sample_before_after["before"], sample_before_after["after"])),
            }

    return df, standardization_report


def try_parse_dates(df: pd.DataFrame, threshold: float = 0.7) -> Tuple[pd.DataFrame, list]:
    df = df.copy()
    converted_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(50)
            if len(sample) == 0:
                continue

            parsed_success = 0
            for val in sample:
                try:
                    parser.parse(val)
                    parsed_success += 1
                except Exception:
                    pass

            if (parsed_success / len(sample)) >= threshold:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    converted_cols.append(col)
                except Exception:
                    pass

    return df, converted_cols


def try_parse_numeric(df: pd.DataFrame, threshold: float = 0.8) -> Tuple[pd.DataFrame, list]:
    df = df.copy()
    converted_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            series = df[col].dropna().astype(str).str.replace(",", "", regex=False).str.strip()
            if len(series) == 0:
                continue

            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().mean() >= threshold:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
                )
                converted_cols.append(col)

    return df, converted_cols


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    return df, before - after


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    fill_report = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            missing_count = int(df[col].isna().sum())
            if missing_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                fill_report[col] = {
                    "strategy": "filled_numeric_with_median",
                    "missing_filled": missing_count,
                }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            pass
        else:
            missing_count = int(df[col].isna().sum())
            if missing_count > 0:
                mode_vals = df[col].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
                df[col] = df[col].fillna(fill_val)
                fill_report[col] = {
                    "strategy": "filled_text_with_mode_or_unknown",
                    "missing_filled": missing_count,
                }

    return df, fill_report


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    outlier_report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())

        outlier_report[col] = {
            "outlier_count": count,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }

    return outlier_report


def clean_dataframe(df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple[pd.DataFrame, dict]:
    original_shape = df.shape
    df = df.copy()
    cleaning_report = {
        "dataset_name": dataset_name,
        "original_rows": int(original_shape[0]),
        "original_columns": int(original_shape[1]),
        "steps": {},
    }

    old_cols = df.columns.tolist()
    new_cols = [standardize_column_name(c) for c in df.columns]
    df.columns = new_cols
    cleaning_report["steps"]["column_standardization"] = {
        "old_columns": old_cols,
        "new_columns": new_cols,
    }

    df = df.apply(lambda col: col.map(clean_text_value))
    cleaning_report["steps"]["text_cleaning"] = "trimmed spaces, normalized blanks/null-like values"

    df, categorical_standardization_report = standardize_categorical_values(df)
    cleaning_report["steps"]["categorical_value_standardization"] = categorical_standardization_report

    df, duplicates_removed = remove_duplicates(df)
    cleaning_report["steps"]["duplicates_removed"] = duplicates_removed

    df, date_cols_converted = try_parse_dates(df)
    cleaning_report["steps"]["date_columns_converted"] = date_cols_converted

    df, numeric_cols_converted = try_parse_numeric(df)
    cleaning_report["steps"]["numeric_columns_converted"] = numeric_cols_converted

    df, missing_fill_report = handle_missing_values(df)
    cleaning_report["steps"]["missing_value_handling"] = missing_fill_report

    outlier_report = detect_outliers_iqr(df)
    cleaning_report["steps"]["outlier_detection"] = outlier_report

    cleaning_report["final_rows"] = int(df.shape[0])
    cleaning_report["final_columns"] = int(df.shape[1])

    return df, cleaning_report


# -----------------------------
# Data helpers
# -----------------------------
def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    lower = uploaded_file.name.lower()

    if lower.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")

    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    return None


def profile_dataframe(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_values_after_cleaning": df.isna().sum().sort_values(ascending=False).head(10).to_dict(),
        "sample_rows": df.head(3).astype(str).to_dict(orient="records"),
    }


def get_top_categories(df: pd.DataFrame, max_cols: int = 5, top_n: int = 5) -> dict:
    results = {}
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols[:max_cols]:
        try:
            results[col] = df[col].astype(str).value_counts(dropna=False).head(top_n).to_dict()
        except Exception:
            pass

    return results


def get_numeric_summary(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {}
    return df[numeric_cols].describe().round(2).to_dict()


def detect_possible_join_keys(datasets_dict: Dict[str, pd.DataFrame]) -> dict:
    all_columns = {}
    for dataset_name, df in datasets_dict.items():
        for col in df.columns:
            all_columns.setdefault(col, []).append(dataset_name)

    shared_columns = {col: ds_list for col, ds_list in all_columns.items() if len(ds_list) > 1}
    key_patterns = ["id", "account", "customer", "service", "number", "no"]
    likely_keys = {
        col: ds_list
        for col, ds_list in shared_columns.items()
        if any(k in col.lower() for k in key_patterns)
    }
    return {"shared_columns": shared_columns, "likely_join_keys": likely_keys}


def build_machine_findings(datasets: Dict[str, pd.DataFrame]) -> dict:
    con = duckdb.connect()
    findings = {}

    for idx, (name, df) in enumerate(datasets.items()):
        safe_base = standardize_column_name(name) or "dataset"
        relation_name = f"ds_{idx}_{safe_base}"
        con.register(relation_name, df)

        item = {}
        try:
            item["row_count"] = con.execute(
                f'SELECT COUNT(*) AS cnt FROM "{relation_name}"'
            ).df().to_dict(orient="records")
        except Exception as e:
            item["row_count_error"] = str(e)

        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if text_cols:
            top_col = text_cols[0]
            try:
                q = f'''
                SELECT "{top_col}" AS category, COUNT(*) AS total
                FROM "{relation_name}"
                GROUP BY 1
                ORDER BY total DESC
                LIMIT 10
                '''
                item["top_category_table"] = con.execute(q).df().to_dict(orient="records")
            except Exception as e:
                item["top_category_table_error"] = str(e)

        findings[name] = item

    return findings


def build_analysis_payload(
    business_goal: str,
    cleaning_reports: dict,
    dataset_profiles: dict,
    dataset_category_insights: dict,
    dataset_numeric_summaries: dict,
    machine_findings: dict,
    join_key_report: dict,
) -> dict:
    return {
        "business_goal": business_goal,
        "cleaning_reports": cleaning_reports,
        "dataset_profiles": dataset_profiles,
        "dataset_category_insights": dataset_category_insights,
        "dataset_numeric_summaries": dataset_numeric_summaries,
        "machine_findings": machine_findings,
        "join_key_report": join_key_report,
    }


def build_analysis_prompt_from_payload(payload: dict) -> str:
    return f"""
You are a senior strategy consultant, customer experience expert, and data analyst.

Context:
The machine has already performed initial data handling and cleaning on all uploaded datasets, including:
- column name standardization
- text cleanup
- null normalization
- duplicate removal
- date parsing
- numeric parsing
- missing value treatment
- outlier detection summary
- category normalization for low-cardinality fields
- normalization of equivalent business values such as close/closed/clsed into one category where appropriate
- initial profiling and summary tables

Business goal:
{payload["business_goal"]}

Cleaning reports:
{json.dumps(payload["cleaning_reports"], indent=2, default=str)}

Dataset profiles after cleaning:
{json.dumps(payload["dataset_profiles"], indent=2, default=str)}

Category insights:
{json.dumps(payload["dataset_category_insights"], indent=2, default=str)}

Numeric summaries:
{json.dumps(payload["dataset_numeric_summaries"], indent=2, default=str)}

Machine-generated findings:
{json.dumps(payload["machine_findings"], indent=2, default=str)}

Possible join keys across datasets:
{json.dumps(payload["join_key_report"], indent=2, default=str)}

Instructions:
Return strict JSON only using this exact schema:
{json.dumps(INSIGHT_SCHEMA_EXAMPLE, indent=2)}
""".strip()


def normalize_executive_payload(data: Any) -> Optional[dict]:
    # Plain object already in the right format
    if isinstance(data, dict):
        # Common wrapper patterns from n8n / manual responses
        if "response" in data and isinstance(data["response"], dict):
            return data["response"]
        if "data" in data and isinstance(data["data"], dict):
            return data["data"]
        if "json" in data and isinstance(data["json"], dict):
            return normalize_executive_payload(data["json"])
        return data

    # n8n often returns a list of items
    if isinstance(data, list):
        if not data:
            return None

        # If the first item is a wrapped n8n item, unwrap it
        first = data[0]
        if isinstance(first, dict):
            if "json" in first and isinstance(first["json"], dict):
                return normalize_executive_payload(first["json"])
            if "response" in first and isinstance(first["response"], dict):
                return normalize_executive_payload(first["response"])
            if "data" in first and isinstance(first["data"], dict):
                return normalize_executive_payload(first["data"])
            return normalize_executive_payload(first)

    return None
    first = data[0]
    if isinstance(first, dict):
        if "json" in first and isinstance(first["json"], dict):
            return normalize_executive_payload(first["json"])
        return normalize_executive_payload(first)

    return None


def call_n8n_webhook(webhook_url: str, payload: dict) -> Tuple[Optional[dict], str]:
    try:
        response = requests.post(webhook_url, json=payload, timeout=180)
        response.raise_for_status()
        raw_data = response.json()
        normalized = normalize_executive_payload(raw_data)
        return normalized, json.dumps(raw_data)
    except Exception as e:
        return None, f"n8n webhook error: {e}"


def safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def parse_fallback_json(raw_text: str) -> Optional[dict]:
    try:
        parsed = json.loads(raw_text)
        return normalize_executive_payload(parsed)
    except Exception:
        return None


# -----------------------------
# Dashboard builders
# -----------------------------
def build_kpi_cards(datasets: Dict[str, pd.DataFrame]):
    total_rows = sum(df.shape[0] for df in datasets.values())
    total_cols = sum(df.shape[1] for df in datasets.values())
    total_datasets = len(datasets)
    total_missing = int(sum(df.isna().sum().sum() for df in datasets.values()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", f"{total_datasets}")
    c2.metric("Total Rows", f"{total_rows:,}")
    c3.metric("Total Columns", f"{total_cols:,}")
    c4.metric("Remaining Missing Cells", f"{total_missing:,}")


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        return datetime_cols[0]

    for col in df.columns:
        lower = col.lower()
        if any(k in lower for k in ["date", "month", "created", "closed", "updated"]):
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().mean() >= 0.5:
                    return col
            except Exception:
                pass
    return None


def detect_category_columns(df: pd.DataFrame) -> list:
    candidates = []
    for col in df.columns:
        if df[col].dtype == "object":
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique <= 30:
                candidates.append(col)
    return candidates[:5]


def detect_numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()[:5]


def render_dataset_dashboard(name: str, df: pd.DataFrame):
    st.markdown(f"### {name}")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    date_col = detect_date_column(df)
    category_cols = detect_category_columns(df)
    numeric_cols = detect_numeric_columns(df)

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if category_cols:
            selected_cat = st.selectbox(
                f"Top categories - {name}",
                options=category_cols,
                key=f"cat_{name}",
            )
            cat_series = df[selected_cat].astype(str).fillna("unknown").value_counts().head(10)
            st.bar_chart(cat_series)
        else:
            st.info("No suitable categorical column found for this dataset.")

    with row1_col2:
        if date_col:
            temp = df.copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col])
            if not temp.empty:
                trend = (
                    temp.assign(period=temp[date_col].dt.to_period("M").astype(str))
                    .groupby("period")
                    .size()
                    .reset_index(name="count")
                    .sort_values("period")
                )
                if not trend.empty:
                    st.line_chart(trend.set_index("period")["count"])
                else:
                    st.info("No usable trend data after date cleaning.")
            else:
                st.info("No valid dates available for trend chart.")
        else:
            st.info("No suitable date column found for this dataset.")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if numeric_cols:
            selected_num = st.selectbox(
                f"Numeric distribution - {name}",
                options=numeric_cols,
                key=f"num_{name}",
            )
            num_series = df[selected_num].dropna()
            if not num_series.empty:
                hist_df = pd.DataFrame({selected_num: num_series})
                st.bar_chart(hist_df[selected_num].value_counts().head(20))
            else:
                st.info("No numeric values available.")
        else:
            st.info("No numeric columns found for this dataset.")

    with row2_col2:
        dq = pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": [int(df[col].isna().sum()) for col in df.columns],
                "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
            }
        ).sort_values("missing_count", ascending=False).head(10)
        st.dataframe(dq, use_container_width=True)


def render_cross_dataset_dashboard(cleaned_datasets: Dict[str, pd.DataFrame], join_key_report: dict):
    st.markdown("## Cross-Dataset Overview")
    build_kpi_cards(cleaned_datasets)

    st.markdown("### Shared fields across datasets")
    shared = join_key_report.get("shared_columns", {})
    likely = join_key_report.get("likely_join_keys", {})

    left, right = st.columns(2)
    with left:
        if shared:
            shared_df = pd.DataFrame(
                [{"column": k, "datasets": ", ".join(v), "count": len(v)} for k, v in shared.items()]
            ).sort_values("count", ascending=False)
            st.dataframe(shared_df, use_container_width=True)
        else:
            st.info("No shared columns detected across datasets.")

    with right:
        if likely:
            likely_df = pd.DataFrame(
                [{"likely_key": k, "datasets": ", ".join(v)} for k, v in likely.items()]
            )
            st.dataframe(likely_df, use_container_width=True)
        else:
            st.info("No likely join keys detected.")


def render_executive_output(data: dict):
    data = normalize_executive_payload(data) or {}
    st.markdown("## Executive Insights")
    st.info(data.get("executive_problem_statement", "No executive problem statement available."))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Key Insights")
        for item in safe_list(data.get("key_insights")):
            st.markdown(f"- {item}")

        st.markdown("### Root Cause Hypotheses")
        for item in safe_list(data.get("root_cause_hypotheses")):
            st.markdown(f"- {item}")

    with c2:
        st.markdown("### Data Quality / Analysis Limitations")
        for item in safe_list(data.get("data_quality_limitations")):
            st.markdown(f"- {item}")

        dashboard_story = data.get("dashboard_story", {})
        if dashboard_story:
            st.markdown("### Dashboard Story")
            st.markdown(f"**Headline:** {dashboard_story.get('headline', '-')}")
            st.markdown("**Priority focus:**")
            for item in safe_list(dashboard_story.get("priority_focus")):
                st.markdown(f"- {item}")
            st.markdown("**Recommended views:**")
            for item in safe_list(dashboard_story.get("recommended_views")):
                st.markdown(f"- {item}")

    st.markdown("### Initiative Opportunities")
    initiatives = pd.DataFrame(safe_list(data.get("initiative_opportunities")))
    if not initiatives.empty:
        st.dataframe(initiatives, use_container_width=True)
    else:
        st.info("No initiative opportunities returned.")

    st.markdown("### KPI Recommendations")
    kpis = pd.DataFrame(safe_list(data.get("kpi_recommendations")))
    if not kpis.empty:
        st.dataframe(kpis, use_container_width=True)
    else:
        st.info("No KPI recommendations returned.")


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    business_goal = st.text_area("Business goal", value=DEFAULT_BUSINESS_GOAL, height=120)
    auto_call_n8n = st.toggle("Auto-call n8n backend", value=True)
    n8n_webhook_url = st.text_input(
        "n8n webhook URL",
        value=st.secrets.get("N8N_WEBHOOK_URL", ""),
    )
    st.caption("Use a different repo and Streamlit app for this dashboard version.")
    st.caption("Update the n8n JSON to include dashboard_story and stronger insight text.")


# -----------------------------
# Main UI tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "1. Data Intake",
    "2. Insight Dashboard",
    "3. Info",
])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload one or more datasets.",
    )

    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_analysis:
        if not uploaded_files:
            st.error("Please upload at least one file.")
        else:
            raw_datasets = {}
            cleaned_datasets = {}
            cleaning_reports = {}

            with st.spinner("Loading, cleaning, and profiling datasets..."):
                for file in uploaded_files:
                    df = load_file(file)
                    if df is None:
                        st.warning(f"Skipped unsupported file: {file.name}")
                        continue

                    dataset_name = re.sub(r"\.[^.]+$", "", file.name)
                    raw_datasets[dataset_name] = df

                    cleaned_df, report = clean_dataframe(df, dataset_name=dataset_name)
                    cleaned_datasets[dataset_name] = cleaned_df
                    cleaning_reports[dataset_name] = report

                if not cleaned_datasets:
                    st.error("No supported files were loaded.")
                else:
                    dataset_profiles = {
                        name: profile_dataframe(df) for name, df in cleaned_datasets.items()
                    }
                    dataset_category_insights = {
                        name: get_top_categories(df) for name, df in cleaned_datasets.items()
                    }
                    dataset_numeric_summaries = {
                        name: get_numeric_summary(df) for name, df in cleaned_datasets.items()
                    }
                    join_key_report = detect_possible_join_keys(cleaned_datasets)
                    machine_findings = build_machine_findings(cleaned_datasets)

                    payload = build_analysis_payload(
                        business_goal,
                        cleaning_reports,
                        dataset_profiles,
                        dataset_category_insights,
                        dataset_numeric_summaries,
                        machine_findings,
                        join_key_report,
                    )

                    prompt = build_analysis_prompt_from_payload(payload)

                    st.session_state.cleaned_datasets = cleaned_datasets
                    st.session_state.cleaning_reports = cleaning_reports
                    st.session_state.dataset_profiles = dataset_profiles
                    st.session_state.dataset_category_insights = dataset_category_insights
                    st.session_state.dataset_numeric_summaries = dataset_numeric_summaries
                    st.session_state.join_key_report = join_key_report
                    st.session_state.machine_findings = machine_findings
                    st.session_state.generated_prompt = prompt
                    st.session_state.analysis_complete = True
                    st.session_state.analysis_output = ""
                    st.session_state.executive_json = None

                    if auto_call_n8n and n8n_webhook_url:
                        with st.spinner("Calling n8n backend..."):
                            parsed_json, raw_text = call_n8n_webhook(n8n_webhook_url, payload)

                        if parsed_json is not None:
                            st.session_state.executive_json = parsed_json
                            st.session_state.analysis_output = raw_text
                        else:
                            st.session_state.analysis_output = raw_text

                    st.success("Analysis completed.")

    #if st.session_state.analysis_complete:
        #st.subheader("Cleaning Reports")
        #st.json(st.session_state.cleaning_reports)

        #st.subheader("Prompt Sent to n8n")
        #st.code(st.session_state.generated_prompt, language="text")

with tab2:
    if not st.session_state.analysis_complete:
        st.info("Run the analysis first in Tab 1.")
    else:
        render_cross_dataset_dashboard(st.session_state.cleaned_datasets, st.session_state.join_key_report)

        st.markdown("## Dataset Dashboards")
        dataset_names = list(st.session_state.cleaned_datasets.keys())
        if dataset_names:
            selected_dataset = st.selectbox("Choose dataset", options=dataset_names)
            render_dataset_dashboard(selected_dataset, st.session_state.cleaned_datasets[selected_dataset])

        if st.session_state.executive_json is not None:
            render_executive_output(st.session_state.executive_json)
        else:
            st.warning("Executive insights are not generated yet.")
            st.markdown("**Option A: Auto mode** — enter a working n8n webhook URL in the sidebar and rerun analysis.")
            st.markdown("**Option B: Manual mode** — paste the JSON response below.")
            pasted_json = st.text_area(
                "Paste JSON output here",
                height=320,
                placeholder="Paste the n8n or model JSON response here...",
            )
            if st.button("Use Pasted JSON", use_container_width=True):
                parsed = parse_fallback_json(pasted_json)
                if parsed is None:
                    st.error("That is not valid JSON. Paste the exact JSON output.")
                else:
                    st.session_state.executive_json = parsed
                    st.session_state.analysis_output = pasted_json
                    st.success("Executive insights loaded successfully.")
                    st.rerun()

with tab3:
    presenter_text = """
### What changed in this dashboard version
- Slide generation is removed from the main flow.
- Tab 2 is now the main demo experience.
- The focus is on interactive charts, graphs, data quality views, and executive insights.

### Recommended new repo
Create a separate repo and deploy as a separate Streamlit app, for example:
- Repo: `ai-initiative-agent-dashboard`
- App name: `insightpilot-ai-v1.streamlit.app`

### n8n JSON changes needed
Your n8n response should still include:
- executive_problem_statement
- key_insights
- data_quality_limitations
- root_cause_hypotheses
- initiative_opportunities
- kpi_recommendations

Add this new block too:
- dashboard_story
  - headline
  - priority_focus
  - recommended_views

### Demo flow
1. Open the dashboard Streamlit link.
2. Upload datasets.
3. Click **Run Analysis**.
4. Show Tab 2.
5. Walk leaders through charts first, then executive insights.

### Recommended requirements.txt
```txt
streamlit
pandas
duckdb
openpyxl
numpy
python-dateutil
requests
```
"""
    st.markdown(presenter_text)