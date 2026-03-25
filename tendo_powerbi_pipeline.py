"""
=============================================================================
Tendo Model Monitoring — Monthly Power BI Data Pipeline
=============================================================================
Run this script once per month. It pulls all raw data from BigQuery / GCS,
applies every transformation from the original analysis notebook, and writes
one CSV per dashboard page into the OUTPUT_DIR folder.

Power BI setup (one-time):
  1. In Power BI Desktop → Get Data → Text/CSV → point to each file in OUTPUT_DIR
  2. Each CSV becomes one table — one table per dashboard page
  3. Refresh monthly: run this script, then hit Refresh in Power BI

Output files (one per slide/page):
  page02_model_performance_gini_cindex.csv
  page03_oop_rank_order_new_users.csv
  page04_oop_rank_order_existing_users.csv
  page05_attrition_rank_order_new_users.csv
  page06_attrition_rank_order_existing_users.csv
  page07_unit_bad_rate_mom.csv
  page08_peso_bad_rate_mom.csv
  page09_cl_plus_exposure.csv
  page10_new_borrower_exposure.csv
  page11_utilization_trend.csv
=============================================================================
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — edit these each month if needed
# =============================================================================

# GCP credentials — use ADC JSON or set GOOGLE_APPLICATION_CREDENTIALS env var
CREDENTIALS_PATH = r"C:\Users\Dwaipayan\AppData\Roaming\gcloud\legacy_credentials\dchakroborti@tonikbank.com\adc.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

PROJECT_ID  = "prj-prod-dataplatform"
GS_BUCKET   = "prod-asia-southeast1-tonik-aiml-workspace"

# Output folder — Power BI reads CSVs from here
OUTPUT_DIR  = r"C:\PowerBI\Tendo_Monitoring_Data"

# Rolling window: script always covers Jun 2025 → current month
# Extend this dict each month by adding the new month entry
REPORT_PERIODS = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    # ← ADD NEXT MONTH HERE e.g. "Jan 2026": {"start": "2026-01-01", "end": "2026-01-31"}
}

# Attrition segment ordering (used for sort column)
ATTRITION_ORDER = {"Very low": 1, "Low": 2, "Average": 3, "High": 4, "Very high": 5}

# OOP segment ordering
OOP_SEGMENT_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}


# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_csv(df: pd.DataFrame, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    log(f"  ✓ Saved {filename}  ({len(df):,} rows)")


def generate_bucket_url(filename, bucket_name):
    return f"gs://{bucket_name}/{filename}"


def calculate_gini(data, date_col, score_col, target_col, periods, weight_col=None):
    """
    Compute Gini for each period in `periods`.
    Returns a tidy DataFrame with one row per period.
    """
    dt = data[data[target_col].notna()].copy()
    dt[date_col] = pd.to_datetime(dt[date_col]).dt.date
    rows = []

    for period_name, period_info in periods.items():
        start = pd.to_datetime(period_info["start"]).date()
        end   = pd.to_datetime(period_info["end"]).date()
        mask  = (dt[date_col] >= start) & (dt[date_col] <= end)
        sub   = dt[mask].copy()

        sample_size = sub["ee_customer_id"].nunique()
        if sample_size == 0:
            rows.append({
                "Period": period_name, "Start_Date": start, "End_Date": end,
                "Sample_Size": 0, "Bad_Rate_Pct": None, "Gini": None
            })
            continue

        bad_rate = 100 * (
            1 - sub[["ee_customer_id", target_col]].drop_duplicates()[target_col].sum()
            / sample_size
        )

        if sub[target_col].nunique() < 2:
            gini = None
        else:
            try:
                kwargs = {"sample_weight": sub[weight_col]} if weight_col else {}
                auc  = roc_auc_score(sub[target_col], sub[score_col], **kwargs)
                gini = round(2 * auc - 1, 4)
            except Exception:
                gini = None

        rows.append({
            "Period": period_name, "Start_Date": start, "End_Date": end,
            "Sample_Size": sample_size, "Bad_Rate_Pct": round(bad_rate, 2),
            "Gini": gini
        })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 1 — LOAD RAW DATA FROM BIGQUERY & GCS
# =============================================================================

log("=== STEP 1: Loading raw data ===")
client = bigquery.Client(PROJECT_ID)

# --- Production API scores (new users) ---
log("Loading prod API scores (new users)...")
dt_prod_api = client.query("""
    SELECT
        employee_id           AS ee_customer_id,
        run_date,
        ee_attrition_risk_segment   AS attrition_risk_segment,
        ee_attrition_time_to_leave  AS attrition_time_to_leave,
        oop_score                   AS oop_score_prod,
        oop_risk_segment            AS oop_risk_segment_prod
    FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table_api`
""").to_dataframe()

# --- Production batch scores (existing users) ---
log("Loading prod batch scores (existing users)...")
dt_prod_batch = client.query("""
    SELECT
        employee_id           AS ee_customer_id,
        run_date,
        ee_attrition_risk_segment   AS attrition_risk_segment,
        ee_attrition_time_to_leave  AS attrition_time_to_leave,
        oop_score                   AS oop_score_prod,
        oop_risk_segment            AS oop_risk_segment_prod
    FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table`
""").to_dataframe()

# --- OOP matured targets ---
log("Loading OOP targets...")
dt_oop_targets = client.query("""
    SELECT user_id AS ee_customer_id, target AS oop_target
    FROM `prj-prod-dataplatform.tendo_mart.tendo_collection_target_master`
    WHERE target_maturity_flag = 1
""").to_dataframe()

# --- Backscored OOP new users ---
log("Loading backscored OOP — new users...")
dt_bs_oop_new = client.query("""
    SELECT *
    FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_new_users_jan23_jan26_20260201_oop_with_osbal`
""").to_dataframe()

# --- Backscored OOP existing users ---
log("Loading backscored OOP — existing users...")
dt_bs_oop_ex = client.query("""
    SELECT *
    FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_existing_users_jan23_jan26_20260201_oop`
""").to_dataframe()

# --- Backscored attrition ---
log("Loading backscored attrition...")
dt_bs_attr = client.query("""
    SELECT *
    FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_jan24_jan26_20260201_attrition`
""").to_dataframe()

# --- Raw features (onboarding dates etc.) from GCS ---
log("Loading raw feature data from GCS...")
dt_raw = pd.read_pickle(generate_bucket_url("Oleh/tendo/data/raw_data_14012026.pkl", GS_BUCKET))

# --- Resignation data from GCS ---
log("Loading resignation data from GCS...")
dt_res = pd.read_pickle(generate_bucket_url("Oleh/tendo/data/resignation_data_14012026.pkl", GS_BUCKET))

log("All raw data loaded.")


# =============================================================================
# STEP 2 — SHARED TRANSFORMATIONS
# =============================================================================

log("=== STEP 2: Shared transformations ===")

# --- Raw features ---
dt_raw["ee_customer_id"] = dt_raw["ee_customer_id"].astype("str")
dt_raw["ee_onboarding_date"] = pd.to_datetime(dt_raw["ee_onboarding_date"]).dt.tz_localize(None)

# --- Backscored OOP new ---
dt_bs_oop_new["ee_customer_id"] = dt_bs_oop_new["ee_customer_id"].astype("str")

# --- Backscored OOP existing: date offset -1 month ---
dt_bs_oop_ex["ee_customer_id"] = dt_bs_oop_ex["ee_customer_id"].astype("str")
dt_bs_oop_ex["calc_date"] = pd.to_datetime(dt_bs_oop_ex["calc_date"], errors="coerce")
dt_bs_oop_ex["calc_date_correct"] = dt_bs_oop_ex["calc_date"] - pd.DateOffset(months=1)
dt_bs_oop_ex["calc_date_ym"] = (
    dt_bs_oop_ex["calc_date_correct"].dt.year * 100
    + dt_bs_oop_ex["calc_date_correct"].dt.month
)

# --- Backscored attrition: date offset + new-customer flag ---
dt_bs_attr["ee_customer_id"] = dt_bs_attr["ee_customer_id"].astype("str")
dt_bs_attr["calc_date"] = pd.to_datetime(dt_bs_attr["calc_date"], errors="coerce")
dt_bs_attr["calc_date_correct"] = dt_bs_attr["calc_date"] - pd.DateOffset(months=1)
dt_bs_attr["calc_date_ym"] = (
    dt_bs_attr["calc_date_correct"].dt.year * 100
    + dt_bs_attr["calc_date_correct"].dt.month
)
dt_bs_attr["is_new_customer_flag_1m"] = (
    (dt_bs_attr["ee_onboarding_month"] == dt_bs_attr["calc_date_ym"]).astype(int)
)

# --- Prod batch: dedup + date fields ---
dt_prod_batch = dt_prod_batch.drop_duplicates(
    subset=["ee_customer_id", "run_date", "attrition_time_to_leave", "oop_score_prod"]
)
dt_prod_batch["run_date"] = pd.to_datetime(dt_prod_batch["run_date"], errors="coerce")
dt_prod_batch["run_date_ym"] = (
    dt_prod_batch["run_date"].dt.year * 100 + dt_prod_batch["run_date"].dt.month
)

# --- Prod API: date fields ---
dt_prod_api["run_date"] = pd.to_datetime(dt_prod_api["run_date"]).dt.normalize()
dt_prod_api["run_date_ym"] = (
    dt_prod_api["run_date"].dt.year * 100 + dt_prod_api["run_date"].dt.month
)

log("Shared transformations complete.")


# =============================================================================
# STEP 3 — BUILD OOP MASTER DATAFRAMES
# (reused across pages 2, 3, 4)
# =============================================================================

log("=== STEP 3: Building OOP master dataframes ===")

# ── OOP New Users master ──────────────────────────────────────────────────────
dt_prod_api_oop = (
    dt_prod_api
    .merge(dt_raw[["ee_customer_id", "ee_onboarding_date"]].drop_duplicates(), how="left", on="ee_customer_id")
    .merge(dt_oop_targets, how="left", on="ee_customer_id")
    .merge(
        dt_bs_oop_new[["ee_customer_id", "score_oop",
                        "osbal_as_of_resignation_date",
                        "osbal_as_of_oop_eligible_date",
                        "osbal_as_of_current_date"]],
        how="left", on="ee_customer_id"
    )
)

dt_prod_api_oop["ee_onboarding_date"] = pd.to_datetime(dt_prod_api_oop["ee_onboarding_date"]).dt.normalize()
dt_prod_api_oop["onb_rd_diff"] = abs(dt_prod_api_oop["run_date"] - dt_prod_api_oop["ee_onboarding_date"]).dt.days
dt_prod_api_oop["onboarding_date_ym"] = (
    dt_prod_api_oop["ee_onboarding_date"].dt.year * 100
    + dt_prod_api_oop["ee_onboarding_date"].dt.month
)
dt_prod_api_oop["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    dt_prod_api_oop["osbal_as_of_oop_eligible_date"]
)

# One row per customer, closest onboarding observation
oop_new_calc = (
    dt_prod_api_oop.dropna(subset=["ee_onboarding_date", "oop_target"])
    .sort_values(["onb_rd_diff", "run_date"])
    .drop_duplicates(subset=["ee_customer_id"], keep="first")
)

# ── OOP Existing Users master ─────────────────────────────────────────────────
dt_prod_batch_oop = (
    dt_prod_batch
    .merge(dt_raw[["ee_customer_id", "ee_onboarding_date"]].drop_duplicates(), how="left", on="ee_customer_id")
    .merge(dt_oop_targets, how="left", on="ee_customer_id")
    .merge(
        dt_bs_oop_new[["ee_customer_id", "osbal_as_of_resignation_date",
                        "osbal_as_of_oop_eligible_date", "osbal_as_of_current_date"]],
        how="left", on="ee_customer_id"
    )
    .merge(
        dt_bs_oop_ex[["ee_customer_id", "calc_date_ym", "score_oop"]],
        how="left",
        left_on=["ee_customer_id", "run_date_ym"],
        right_on=["ee_customer_id", "calc_date_ym"],
    )
)

# Fix column name collision after merge
if "ee_onboarding_date_x" in dt_prod_batch_oop.columns:
    dt_prod_batch_oop.rename(columns={"ee_onboarding_date_x": "ee_onboarding_date"}, inplace=True)

dt_prod_batch_oop["ee_onboarding_date"] = pd.to_datetime(dt_prod_batch_oop["ee_onboarding_date"]).dt.normalize()
dt_prod_batch_oop["onboarding_date_ym"] = (
    dt_prod_batch_oop["ee_onboarding_date"].dt.year * 100
    + dt_prod_batch_oop["ee_onboarding_date"].dt.month
)
dt_prod_batch_oop["onb_rd_diff"] = abs(dt_prod_batch_oop["run_date"] - dt_prod_batch_oop["ee_onboarding_date"]).dt.days
dt_prod_batch_oop["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    dt_prod_batch_oop["osbal_as_of_oop_eligible_date"]
)

oop_existing_calc = (
    dt_prod_batch_oop.dropna(subset=["ee_onboarding_date", "oop_target"])
    .sort_values(["ee_customer_id", "run_date"])
)

log("OOP master dataframes ready.")


# =============================================================================
# STEP 4 — BUILD ATTRITION MASTER DATAFRAMES
# (reused across pages 2, 5, 6)
# =============================================================================

log("=== STEP 4: Building Attrition master dataframes ===")

ATTRITION_SCORE_MAP = {
    1: "Very high", 2: "Very high", 3: "Very high",
    4: "High",      5: "High",      6: "High",
    7: "Average",   8: "Average",   9: "Average",
    10: "Low",     11: "Low",      12: "Low",
    15: "Very low",
}

def build_attrition_derived(df, date_col):
    """Add time_to_attrition, attrition_event, score corrections."""
    months_diff = (
        (df["ee_resignation_date_correct"].dt.year  - df[date_col].dt.year)  * 12
      + (df["ee_resignation_date_correct"].dt.month - df[date_col].dt.month)
    )
    df["time_to_attrition"] = np.where(df["ee_resignation_date_correct"].isna(), np.nan, months_diff)
    df["attrition_event"]   = df["ee_resignation_date_correct"].notna().astype(int)
    df["score_attr_corrected"] = df["score_attr"].replace(np.inf, 15)
    return df


# ── Attrition New Users (prod) ─────────────────────────────────────────────
attr_prod_api = (
    dt_prod_api
    .merge(dt_res, how="left", on="ee_customer_id")
    .merge(
        dt_bs_attr[["ee_customer_id", "calc_date_ym", "score_attr", "score_attr_segment", "is_new_customer_flag_1m"]],
        how="left",
        left_on=["ee_customer_id", "run_date_ym"],
        right_on=["ee_customer_id", "calc_date_ym"],
    )
)
attr_prod_api_calc = (
    attr_prod_api.dropna(subset=["ee_onboarding_date"])
    .sort_values(["onb_rd_diff", "run_date"])
    .drop_duplicates(subset=["ee_customer_id"], keep="first")
    .copy()
)
attr_prod_api_calc = build_attrition_derived(attr_prod_api_calc, "run_date")
attr_prod_api_calc["attrition_score_prod"] = (
    attr_prod_api_calc["attrition_time_to_leave"].replace("12+", "15").astype("float")
)
attr_prod_api_calc["attrition_risk_segment_prod"] = (
    attr_prod_api_calc["attrition_score_prod"].replace(ATTRITION_SCORE_MAP)
)

# ── Attrition New Users (backscored) ──────────────────────────────────────
attr_bs_new = (
    dt_bs_attr
    .merge(dt_res, how="left", on="ee_customer_id")
)
attr_bs_new_calc = attr_bs_new.query("is_new_customer_flag_1m == 1").copy()
attr_bs_new_calc = build_attrition_derived(attr_bs_new_calc, "calc_date_correct")

# ── Attrition Existing Users (prod) ───────────────────────────────────────
attr_prod_batch = (
    dt_prod_batch
    .merge(dt_res, how="left", on="ee_customer_id")
    .merge(
        dt_bs_attr[["ee_customer_id", "calc_date_ym", "score_attr", "score_attr_segment", "is_new_customer_flag_1m"]],
        how="left",
        left_on=["ee_customer_id", "run_date_ym"],
        right_on=["ee_customer_id", "calc_date_ym"],
    )
)
attr_prod_batch_calc = attr_prod_batch.dropna(subset=["ee_onboarding_date"]).copy()
attr_prod_batch_calc = build_attrition_derived(attr_prod_batch_calc, "run_date")
attr_prod_batch_calc["attrition_score_prod"] = (
    attr_prod_batch_calc["attrition_time_to_leave"].replace("12+", "15").astype("float")
)
attr_prod_batch_calc["attrition_risk_segment_prod"] = (
    attr_prod_batch_calc["attrition_score_prod"].replace(ATTRITION_SCORE_MAP)
)

# ── Attrition Existing Users (backscored) ─────────────────────────────────
attr_bs_existing_calc = (
    dt_bs_attr
    .merge(dt_res, how="left", on="ee_customer_id")
    .query("is_new_customer_flag_3m == 0")
    .copy()
)
attr_bs_existing_calc = build_attrition_derived(attr_bs_existing_calc, "calc_date_correct")
attr_bs_existing_calc["score_attr_corrected"] = attr_bs_existing_calc["score_attr"].replace(np.inf, 15)

log("Attrition master dataframes ready.")


# =============================================================================
# PAGE 2 — MODEL PERFORMANCE: GINI (OOP) & C-INDEX (ATTRITION)
# =============================================================================
# Slide: Summary table of Dev Gini vs Jul-Aug Gini, Dev C-Index vs Jul-Aug C-Index
# for New Users and Existing Users.
#
# In the original notebook the Dev Gini/C-Index come from the training experiment
# artifacts. We reconstruct the "Jul-Aug monitoring" Gini here dynamically using
# the same calculate_gini logic. Dev numbers are read from constants that match
# the slide (they don't change month-to-month).
# =============================================================================

log("=== PAGE 2: Model Performance Summary ===")

MONITORING_WINDOW = {
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Jul-Aug 2025": {"start": "2025-07-01", "end": "2025-08-31"},
}

# ── OOP Gini — New Users ──────────────────────────────────────────────────
gini_oop_new_prod = calculate_gini(
    oop_new_calc, "ee_onboarding_date", "oop_score_prod", "oop_target", MONITORING_WINDOW
).assign(Model="OOP Repay", ModelVersion="v1.0 Prod", UserType="New users (< 3MOB)")

gini_oop_new_bs = calculate_gini(
    oop_new_calc[oop_new_calc["score_oop"].notna()],
    "ee_onboarding_date", "score_oop", "oop_target", MONITORING_WINDOW
).assign(Model="OOP Repay", ModelVersion="v2.0 BS", UserType="New users (< 3MOB)")

# ── OOP Gini — Existing Users ─────────────────────────────────────────────
gini_oop_ex_prod = calculate_gini(
    oop_existing_calc, "run_date", "oop_score_prod", "oop_target", MONITORING_WINDOW
).assign(Model="OOP Repay", ModelVersion="v1.0 Prod", UserType="Existing users (>= 3MOB)")

gini_oop_ex_bs = calculate_gini(
    oop_existing_calc[oop_existing_calc["score_oop"].notna()],
    "run_date", "score_oop", "oop_target", MONITORING_WINDOW
).assign(Model="OOP Repay", ModelVersion="v2.0 BS", UserType="Existing users (>= 3MOB)")

# ── C-Index (Gini analogue) — Attrition New Users ────────────────────────
# C-Index = 2*AUC-1 on the attrition score vs attrition_event
gini_attr_new_prod = calculate_gini(
    attr_prod_api_calc[attr_prod_api_calc["attrition_score_prod"].notna()],
    "run_date", "attrition_score_prod", "attrition_event", MONITORING_WINDOW
).assign(Model="Attrition", ModelVersion="v1.0 Prod", UserType="New users (< 3MOB)")

gini_attr_new_bs = calculate_gini(
    attr_bs_new_calc[attr_bs_new_calc["score_attr_corrected"].notna()],
    "calc_date_correct", "score_attr_corrected", "attrition_event", MONITORING_WINDOW
).assign(Model="Attrition", ModelVersion="v2.0 BS", UserType="New users (< 3MOB)")

# ── C-Index — Attrition Existing Users ───────────────────────────────────
gini_attr_ex_prod = calculate_gini(
    attr_prod_batch_calc[attr_prod_batch_calc["attrition_score_prod"].notna()],
    "run_date", "attrition_score_prod", "attrition_event", MONITORING_WINDOW
).assign(Model="Attrition", ModelVersion="v1.0 Prod", UserType="Existing users (>= 3MOB)")

gini_attr_ex_bs = calculate_gini(
    attr_bs_existing_calc[attr_bs_existing_calc["score_attr_corrected"].notna()],
    "calc_date_correct", "score_attr_corrected", "attrition_event", MONITORING_WINDOW
).assign(Model="Attrition", ModelVersion="v2.0 BS", UserType="Existing users (>= 3MOB)")

# Combine all, rename Gini → MetricValue for unified column
page2 = pd.concat([
    gini_oop_new_prod, gini_oop_new_bs,
    gini_oop_ex_prod,  gini_oop_ex_bs,
    gini_attr_new_prod, gini_attr_new_bs,
    gini_attr_ex_prod,  gini_attr_ex_bs,
], ignore_index=True).rename(columns={"Gini": "MetricValue"})

# Add dev numbers from the slide (these are the fixed training-time numbers)
DEV_NUMBERS = [
    {"UserType": "New users (< 3MOB)",       "Model": "OOP Repay",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "MetricValue": 0.30},
    {"UserType": "Existing users (>= 3MOB)", "Model": "OOP Repay",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "MetricValue": 0.32},
    {"UserType": "New users (< 3MOB)",       "Model": "Attrition",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "MetricValue": 0.66},
    {"UserType": "Existing users (>= 3MOB)", "Model": "Attrition",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "MetricValue": 0.64},
]
page2 = pd.concat([page2, pd.DataFrame(DEV_NUMBERS)], ignore_index=True)
page2["MetricName"] = page2["Model"].map({"OOP Repay": "Gini", "Attrition": "C-Index"})
page2["RunMonth"] = datetime.now().strftime("%Y-%m")

save_csv(page2, "page02_model_performance_gini_cindex.csv")


# =============================================================================
# PAGE 3 — OOP RANK ORDER: NEW USERS
# =============================================================================
# Slide: Jun–Aug 2025 segment pivot. v1.0 prod vs v2.0 backscored.
# Columns: RiskGroup, Count, BadRate%, SumOsbalResignation,
#          SumOsbalBadsToday, OutstandingBadRate
# =============================================================================

log("=== PAGE 3: OOP Rank Order — New Users ===")

def build_oop_rank_order(df, risk_group_col, snapshot_filter_expr, model_version_label, date_label):
    """Build the OOP segment pivot table for one model version."""
    sub = df.query(snapshot_filter_expr).copy()

    sub["oop_target_bad"] = 1 - sub["oop_target"]
    sub["osbal_current_bad"] = sub["oop_target_bad"] * sub["osbal_as_of_current_date"]

    pt = (
        sub.groupby(risk_group_col)
        .agg(
            Count=("ee_customer_id", "count"),
            Bad_Count=("oop_target_bad", "sum"),
            Sum_Osbal_Resignation=("osbal_as_of_resignation_date", "sum"),
            Sum_Osbal_Bads_Today=("osbal_current_bad", "sum"),
            Sum_Osbal_Eligible=("osbal_as_of_oop_eligible_date", "sum"),
        )
        .reset_index()
        .rename(columns={risk_group_col: "RiskGroup"})
    )
    pt["Bad_Rate_Pct"]         = (pt["Bad_Count"] / pt["Count"] * 100).round(1)
    pt["Outstanding_Bad_Rate"] = (pt["Sum_Osbal_Bads_Today"] / pt["Sum_Osbal_Resignation"] * 100).round(1)
    pt["ModelVersion"]         = model_version_label
    pt["Period"]               = date_label
    pt["Segment_Order"]        = pt["RiskGroup"].map(OOP_SEGMENT_ORDER)
    pt["RunMonth"]             = datetime.now().strftime("%Y-%m")
    return pt.drop(columns=["Bad_Count"])


# First derive BS segment labels for v2.0 (same cut logic as original code)
# Use score_oop with production-derived bin edges
def derive_bs_segment(df, score_col, prod_segment_col, labels):
    """Infer cut edges from prod segments, apply to BS scores."""
    mm = (
        df.assign(
            _seg = df[prod_segment_col].astype(str),
            _score = pd.to_numeric(df[score_col], errors="coerce"),
        )
        .groupby("_seg")["_score"]
        .agg(["min", "max"])
        .reindex(list("ABCDEF"))
        .clip(0, 1)
        .interpolate(limit_direction="both")
    )
    order = list("ABCDEF")
    cuts = pd.Series(
        [(mm.loc[h, "min"] + mm.loc[l, "max"]) / 2 for h, l in zip(order[:-1], order[1:])],
        index=[f"{h}/{l}" for h, l in zip(order[:-1], order[1:])]
    ).cummin()
    edges = [0.0] + cuts.iloc[::-1].tolist() + [np.nextafter(1.0, 2.0)]
    return pd.cut(df[score_col], bins=edges, labels=list("FEDCBA"), right=False, include_lowest=True)


# New users — Jun-Aug 2025 filter
oop_new_calc["oop_risk_segment_bs"] = derive_bs_segment(
    oop_new_calc, "score_oop", "oop_risk_segment_prod", list("FEDCBA")
)

p3_v1 = build_oop_rank_order(
    oop_new_calc, "oop_risk_segment_prod",
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508",
    "OOP v1.0 Prod", "Jun-Aug 2025"
)
p3_v2 = build_oop_rank_order(
    oop_new_calc[oop_new_calc["score_oop"].notna()],
    "oop_risk_segment_bs",
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508",
    "OOP v2.0 BS", "Jun-Aug 2025"
)

page3 = pd.concat([p3_v1, p3_v2], ignore_index=True).sort_values(["ModelVersion", "Segment_Order"])
save_csv(page3, "page03_oop_rank_order_new_users.csv")


# =============================================================================
# PAGE 4 — OOP RANK ORDER: EXISTING USERS
# =============================================================================
# Slide: Jul 2025 snapshot. Same structure as page 3 but for existing users.
# =============================================================================

log("=== PAGE 4: OOP Rank Order — Existing Users ===")

oop_existing_calc["oop_risk_segment_bs"] = derive_bs_segment(
    oop_existing_calc, "score_oop", "oop_risk_segment_prod", list("FEDCBA")
)

p4_v1 = build_oop_rank_order(
    oop_existing_calc, "oop_risk_segment_prod",
    "run_date_ym == 202507",
    "OOP v1.0 Prod", "Jul 2025"
)
p4_v2 = build_oop_rank_order(
    oop_existing_calc[oop_existing_calc["score_oop"].notna()],
    "oop_risk_segment_bs",
    "run_date_ym == 202507",
    "OOP v2.0 BS", "Jul 2025"
)

page4 = pd.concat([p4_v1, p4_v2], ignore_index=True).sort_values(["ModelVersion", "Segment_Order"])
save_csv(page4, "page04_oop_rank_order_existing_users.csv")


# =============================================================================
# PAGE 5 — ATTRITION RANK ORDER: NEW USERS
# =============================================================================
# Slide: Jun–Aug 2025. Prod v1.0 and backscored v2.0 side by side.
# Columns: AttritionGroup, Count, AttritionRate%, ExpAvgTTA, ActualAvgTTA
# =============================================================================

log("=== PAGE 5: Attrition Rank Order — New Users ===")

def build_attrition_rank_order(df, segment_col, filter_expr, model_label, period_label, date_col="calc_date_correct"):
    sub = df.query(filter_expr).copy() if filter_expr else df.copy()
    pt = (
        sub.groupby(segment_col)
        .agg(
            Count=("ee_customer_id", "count"),
            Attrition_Rate=("attrition_event", "mean"),
            Expected_Avg_TTA=("score_attr_corrected", "mean"),
            Actual_Avg_TTA=("time_to_attrition", "mean"),
        )
        .reset_index()
        .rename(columns={segment_col: "AttritionGroup"})
    )
    pt["Attrition_Rate_Pct"] = (pt["Attrition_Rate"] * 100).round(1)
    pt["Expected_Avg_TTA"]   = pt["Expected_Avg_TTA"].round(1)
    pt["Actual_Avg_TTA"]     = pt["Actual_Avg_TTA"].round(1)
    pt["ModelVersion"]        = model_label
    pt["Period"]              = period_label
    pt["Segment_Order"]       = pt["AttritionGroup"].map(ATTRITION_ORDER)
    pt["RunMonth"]            = datetime.now().strftime("%Y-%m")
    return pt.drop(columns=["Attrition_Rate"])

# Prod v1.0 — new users, prod score, Jun-Aug
p5_v1 = build_attrition_rank_order(
    attr_prod_api_calc, "attrition_risk_segment_prod",
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508",
    "Attrition v1.0 Prod", "Jun-Aug 2025"
)
# BS v2.0 — new users, BS score, Jun-Aug (using ee_onboarding_month)
p5_v2 = build_attrition_rank_order(
    attr_bs_new_calc, "score_attr_segment",
    "ee_onboarding_month >= 202506 & ee_onboarding_month <= 202508",
    "Attrition v2.0 BS", "Jun-Aug 2025", date_col="calc_date_correct"
)

page5 = pd.concat([p5_v1, p5_v2], ignore_index=True).sort_values(["ModelVersion", "Segment_Order"])
save_csv(page5, "page05_attrition_rank_order_new_users.csv")


# =============================================================================
# PAGE 6 — ATTRITION RANK ORDER: EXISTING USERS
# =============================================================================
# Slide: Jul 2025 snapshot. Monotonicity observed in attrition rate.
# =============================================================================

log("=== PAGE 6: Attrition Rank Order — Existing Users ===")

# Prod v1.0 — existing users, prod score, Jul 2025 snapshot
p6_v1 = build_attrition_rank_order(
    attr_prod_batch_calc.query("run_date_ym == 202507"),
    "attrition_risk_segment_prod", None,
    "Attrition v1.0 Prod", "Jul 2025"
)

# BS v2.0 — existing users (is_new_customer_flag_3m == 0), BS score, Jul 2025
p6_v2 = build_attrition_rank_order(
    attr_bs_existing_calc.query("calc_date_ym == 202507"),
    "score_attr_segment", None,
    "Attrition v2.0 BS", "Jul 2025"
)

page6 = pd.concat([p6_v1, p6_v2], ignore_index=True).sort_values(["ModelVersion", "Segment_Order"])
save_csv(page6, "page06_attrition_rank_order_existing_users.csv")


# =============================================================================
# PAGE 7 — UNIT BAD RATE MoM: SC1.0 vs SC2.0
# =============================================================================
# Slide: Month-over-month unit bad rate for new customers.
# This requires a separate BigQuery query that tracks SC cohorts.
# Columns: Scorecard, OnboardingMonth, CntOnboarded, CntLeftJob,
#          CntHadOopOutstanding, CntOopFlagBad, UnitBadRate
# =============================================================================

log("=== PAGE 7: Unit Bad Rate MoM ===")

dt_unit_bad_rate = client.query("""
    WITH base AS (
        SELECT
            sc.employee_id                   AS ee_customer_id,
            sc.scorecard_version,
            FORMAT_DATE('%Y-%m', sc.onboarding_date) AS onboarding_month,

            -- did the employee leave by Jan 2026?
            CASE WHEN res.ee_resignation_date_correct <= DATE '2026-01-31' THEN 1 ELSE 0 END AS left_job,

            -- did they have OOP outstanding at the time they left?
            CASE WHEN osbal.osbal_as_of_resignation_date > 0 THEN 1 ELSE 0 END AS had_oop_outstanding,

            -- were they a bad OOP customer?
            tgt.target AS oop_target

        FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table_api` sc
        LEFT JOIN `prj-prod-dataplatform.tendo_mart.resignation_data` res
            ON sc.employee_id = res.ee_customer_id
        LEFT JOIN `prj-prod-dataplatform.risk_mart.tendo_backscored_new_users_jan23_jan26_20260201_oop_with_osbal` osbal
            ON sc.employee_id = osbal.ee_customer_id
        LEFT JOIN `prj-prod-dataplatform.tendo_mart.tendo_collection_target_master` tgt
            ON sc.employee_id = tgt.user_id AND tgt.target_maturity_flag = 1
        WHERE sc.onboarding_date BETWEEN DATE '2025-06-01' AND DATE '2025-12-31'
    )
    SELECT
        scorecard_version                                           AS Scorecard,
        onboarding_month                                            AS OnboardingMonth,
        COUNT(DISTINCT ee_customer_id)                             AS CntOnboarded,
        SUM(left_job)                                              AS CntLeftJob,
        SUM(CASE WHEN left_job = 1 AND had_oop_outstanding = 1 THEN 1 ELSE 0 END) AS CntHadOopOutstanding,
        SUM(CASE WHEN left_job = 1 AND had_oop_outstanding = 1 AND oop_target = 0 THEN 1 ELSE 0 END) AS CntOopFlagBad,
        SAFE_DIVIDE(
            SUM(CASE WHEN left_job = 1 AND had_oop_outstanding = 1 AND oop_target = 0 THEN 1 ELSE 0 END),
            NULLIF(SUM(left_job), 0)
        )                                                           AS UnitBadRate
    FROM base
    GROUP BY 1, 2
    ORDER BY 1, 2
""").to_dataframe()

dt_unit_bad_rate["UnitBadRate_Pct"] = (dt_unit_bad_rate["UnitBadRate"] * 100).round(2)
dt_unit_bad_rate["RunMonth"] = datetime.now().strftime("%Y-%m")
save_csv(dt_unit_bad_rate, "page07_unit_bad_rate_mom.csv")


# =============================================================================
# PAGE 8 — PESO BAD RATE MoM: SC1.0 vs SC2.0
# =============================================================================
# Slide: Same cohort as page 7 but peso (PHP outstanding balance) based.
# Columns: Scorecard, OnboardingMonth, TotCLAmt, TotOsAsOfResignation,
#          TotOsAsOfOopEligible, TotOsFromBadCustomers, PesoBadRate
# =============================================================================

log("=== PAGE 8: Peso Bad Rate MoM ===")

dt_peso_bad_rate = client.query("""
    WITH base AS (
        SELECT
            sc.employee_id                   AS ee_customer_id,
            sc.scorecard_version,
            FORMAT_DATE('%Y-%m', sc.onboarding_date) AS onboarding_month,
            sc.credit_limit                  AS cl_amt,

            -- resignation-time outstanding balances
            osbal.osbal_as_of_resignation_date,
            osbal.osbal_as_of_oop_eligible_date,
            osbal.osbal_as_of_current_date,

            CASE WHEN res.ee_resignation_date_correct <= DATE '2026-01-31' THEN 1 ELSE 0 END AS left_job,
            tgt.target AS oop_target
        FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table_api` sc
        LEFT JOIN `prj-prod-dataplatform.tendo_mart.resignation_data` res
            ON sc.employee_id = res.ee_customer_id
        LEFT JOIN `prj-prod-dataplatform.risk_mart.tendo_backscored_new_users_jan23_jan26_20260201_oop_with_osbal` osbal
            ON sc.employee_id = osbal.ee_customer_id
        LEFT JOIN `prj-prod-dataplatform.tendo_mart.tendo_collection_target_master` tgt
            ON sc.employee_id = tgt.user_id AND tgt.target_maturity_flag = 1
        WHERE sc.onboarding_date BETWEEN DATE '2025-06-01' AND DATE '2025-12-31'
    )
    SELECT
        scorecard_version                        AS Scorecard,
        onboarding_month                         AS OnboardingMonth,
        SUM(cl_amt)                              AS TotCLAmt,
        SUM(CASE WHEN left_job = 1 THEN osbal_as_of_resignation_date ELSE 0 END)  AS TotOsAsOfResignation,
        SUM(CASE WHEN left_job = 1 THEN osbal_as_of_oop_eligible_date ELSE 0 END) AS TotOsAsOfOopEligible,
        SUM(CASE WHEN left_job = 1 AND oop_target = 0 THEN osbal_as_of_current_date ELSE 0 END) AS TotOsFromBadCustomers,
        SAFE_DIVIDE(
            SUM(CASE WHEN left_job = 1 AND oop_target = 0 THEN osbal_as_of_current_date ELSE 0 END),
            NULLIF(SUM(CASE WHEN left_job = 1 THEN osbal_as_of_oop_eligible_date ELSE 0 END), 0)
        )                                        AS PesoBadRate
    FROM base
    GROUP BY 1, 2
    ORDER BY 1, 2
""").to_dataframe()

dt_peso_bad_rate["PesoBadRate_Pct"] = (dt_peso_bad_rate["PesoBadRate"] * 100).round(2)
dt_peso_bad_rate["RunMonth"] = datetime.now().strftime("%Y-%m")
save_csv(dt_peso_bad_rate, "page08_peso_bad_rate_mom.csv")


# =============================================================================
# PAGE 9 — CL+ INCREMENTAL PORTFOLIO EXPOSURE
# =============================================================================
# Slide: Delta CL before/after CL+ start for existing Tendo users.
# Columns: CLCategory, UserCount, SumOldCL, SumNewCL, CLDelta, PctCLIncrease
# =============================================================================

log("=== PAGE 9: CL+ Exposure ===")

dt_cl_plus = client.query("""
    WITH cl_deltas AS (
        SELECT
            user_id,
            old_cl,
            new_cl,
            (new_cl - old_cl) AS cl_delta,
            CASE
                WHEN new_cl > old_cl THEN 'CL Increased'
                WHEN new_cl < old_cl THEN 'CL Decreased'
                ELSE 'No CL Increase'
            END AS cl_category
        FROM `prj-prod-dataplatform.tendo_mart.cl_plus_credit_limit_changes`
        WHERE is_existing_user = TRUE
          AND cl_plus_start_date IS NOT NULL
    )
    SELECT
        cl_category                          AS CLCategory,
        COUNT(DISTINCT user_id)              AS UserCount,
        SUM(old_cl)                          AS SumOldCL,
        SUM(new_cl)                          AS SumNewCL,
        SUM(cl_delta)                        AS CLDelta,
        SAFE_DIVIDE(SUM(cl_delta), NULLIF(SUM(old_cl), 0)) AS PctCLIncrease
    FROM cl_deltas
    GROUP BY 1
    ORDER BY 1
""").to_dataframe()

dt_cl_plus["PctCLIncrease_Pct"] = (dt_cl_plus["PctCLIncrease"] * 100).round(1)
dt_cl_plus["RunMonth"] = datetime.now().strftime("%Y-%m")
save_csv(dt_cl_plus, "page09_cl_plus_exposure.csv")


# =============================================================================
# PAGE 10 — NEW BORROWER INCREMENTAL EXPOSURE
# =============================================================================
# Slide: CL by onboarding cohort + user type mix (new/old × CL+/no CL+)
# Two sub-tables stitched into one wide CSV with a TableType column
# =============================================================================

log("=== PAGE 10: New Borrower Exposure ===")

# Sub-table A: cohort credit limits
dt_cohort_cl = client.query("""
    SELECT
        CASE
            WHEN onboarding_date < DATE '2025-01-01' THEN 'Onboarded before'
            WHEN onboarding_date BETWEEN DATE '2025-01-01' AND DATE '2025-03-31' THEN 'Onboarded Q1 2025'
            WHEN onboarding_date BETWEEN DATE '2025-04-01' AND DATE '2025-06-30' THEN 'Onboarded Q2 2025'
            WHEN onboarding_date BETWEEN DATE '2025-07-01' AND DATE '2025-09-30' THEN 'Onboarded Q3 2025'
            WHEN onboarding_date BETWEEN DATE '2025-10-01' AND DATE '2025-12-31' THEN 'Onboarded Q4 2025'
            ELSE 'Onboarded 2026'
        END                                 AS OnboardingCohort,
        COUNT(DISTINCT user_id)             AS UserCount,
        SUM(credit_limit)                   AS SumNewCL
    FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table_api`
    GROUP BY 1
    ORDER BY MIN(onboarding_date)
""").to_dataframe()
dt_cohort_cl["TableType"] = "CohortCL"
dt_cohort_cl["RunMonth"]  = datetime.now().strftime("%Y-%m")

# Sub-table B: user type × CL+ mix
dt_user_mix = client.query("""
    SELECT
        CASE
            WHEN is_new_user = TRUE  AND has_cl_plus = TRUE  THEN 'new_user_w_cl+'
            WHEN is_new_user = TRUE  AND has_cl_plus = FALSE THEN 'new_user_w_no_cl+'
            WHEN is_new_user = FALSE AND has_cl_plus = TRUE  THEN 'old_user_w_cl+'
            ELSE 'old_user_w_no_cl+'
        END                             AS UserType,
        COUNT(DISTINCT user_id)         AS UserCount,
        SAFE_DIVIDE(COUNT(DISTINCT user_id),
            SUM(COUNT(DISTINCT user_id)) OVER ()) AS Pct
    FROM `prj-prod-dataplatform.tendo_mart.tendo_user_cl_plus_summary`
    GROUP BY 1
    ORDER BY 1
""").to_dataframe()
dt_user_mix["TableType"] = "UserMix"
dt_user_mix["RunMonth"]  = datetime.now().strftime("%Y-%m")
dt_user_mix["Pct_Pct"]  = (dt_user_mix["Pct"] * 100).round(1)

page10 = pd.concat([dt_cohort_cl, dt_user_mix], ignore_index=True)
save_csv(page10, "page10_new_borrower_exposure.csv")


# =============================================================================
# PAGE 11 — UTILIZATION TREND (Jul–Dec 2025)
# =============================================================================
# Slide: End-of-month Credit Limit, Unpaid Principal, Utilization % per month
# =============================================================================

log("=== PAGE 11: Utilization Trend ===")

dt_utilization = client.query("""
    SELECT
        FORMAT_DATE('%Y-%m', report_month_end) AS ReportMonth,
        SUM(credit_limit)                      AS CreditLimit,
        SUM(unpaid_principal_balance)          AS UnpaidPrincipal,
        SAFE_DIVIDE(
            SUM(unpaid_principal_balance),
            NULLIF(SUM(credit_limit), 0)
        )                                      AS UtilizationRate
    FROM `prj-prod-dataplatform.tendo_mart.portfolio_monthly_snapshot`
    WHERE report_month_end BETWEEN DATE '2025-07-01' AND CURRENT_DATE()
    GROUP BY 1
    ORDER BY 1
""").to_dataframe()

dt_utilization["UtilizationRate_Pct"] = (dt_utilization["UtilizationRate"] * 100).round(1)
dt_utilization["RunMonth"] = datetime.now().strftime("%Y-%m")
save_csv(dt_utilization, "page11_utilization_trend.csv")


# =============================================================================
# DONE
# =============================================================================

log("=================================================================")
log(f"All 10 CSV files written to: {OUTPUT_DIR}")
log("Next steps:")
log("  1. Open Power BI Desktop")
log("  2. For first-time setup: Get Data > CSV > select each file")
log("  3. For monthly refresh: click Refresh on the dataset")
log("=================================================================")
