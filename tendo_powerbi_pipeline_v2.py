"""
=============================================================================
Tendo Model Monitoring — Monthly Power BI Data Pipeline  (FIXED v2)
=============================================================================
Run once per month. Writes one CSV per dashboard page to OUTPUT_DIR.
Power BI reads those CSVs — just hit Refresh each month.

Root-cause fix vs v1
---------------------
The original notebook mutates dt_prod_api and dt_prod_batch in-place
across OOP and Attrition sections, so each section always sees all
previously added columns.  In a standalone script every section starts
from the raw BigQuery pull, so column-name collisions from repeated
merges (especially with dt_res which shares column names like
ee_onboarding_date_correct) produce KeyErrors.

Fix: keep ONE immutable base copy of each prod table, build every
section's working frame from that base copy, and resolve all suffix
collisions explicitly with rename() immediately after each merge.
=============================================================================
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CREDENTIALS_PATH = r"C:\Users\Dwaipayan\AppData\Roaming\gcloud\legacy_credentials\dchakroborti@tonikbank.com\adc.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

PROJECT_ID = "prj-prod-dataplatform"
GS_BUCKET  = "prod-asia-southeast1-tonik-aiml-workspace"
OUTPUT_DIR = r"C:\PowerBI\Tendo_Monitoring_Data"

# Add the new month here each month
REPORT_PERIODS = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    # "Jan 2026": {"start": "2026-01-01", "end": "2026-01-31"},
}

ATTRITION_SCORE_MAP = {
    1: "Very high", 2: "Very high", 3: "Very high",
    4: "High",      5: "High",      6: "High",
    7: "Average",   8: "Average",   9: "Average",
    10: "Low",     11: "Low",      12: "Low",
    15: "Very low",
}

ATTRITION_ORDER = {"Very low": 1, "Low": 2, "Average": 3, "High": 4, "Very high": 5}
OOP_SEGMENT_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

RUN_MONTH = datetime.now().strftime("%Y-%m")


# =============================================================================
# UTILITIES
# =============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_csv(df, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    log(f"  ✓ {filename}  ({len(df):,} rows, {len(df.columns)} cols)")


def generate_bucket_url(filename, bucket_name):
    return f"gs://{bucket_name}/{filename}"


def add_date_ym(df, col_name, new_col_name):
    """Add an integer YYYYMM column from a datetime column."""
    dt = pd.to_datetime(df[col_name], errors="coerce")
    df[new_col_name] = dt.dt.year * 100 + dt.dt.month
    return df


def fix_suffix_collision(df, suffix="_x", rename_to=None, drop_suffix="_y"):
    """
    After a merge that produces col_x / col_y pairs, keep _x (left table wins),
    rename it back to the original name, and drop _y.
    rename_to: list of base column names to fix. If None, auto-detects all _x cols.
    """
    if rename_to is None:
        rename_to = [c[:-2] for c in df.columns if c.endswith(suffix)]
    for base in rename_to:
        x_col = base + suffix
        y_col = base + drop_suffix
        if x_col in df.columns:
            df = df.rename(columns={x_col: base})
        if y_col in df.columns:
            df = df.drop(columns=[y_col])
    return df


def calc_gini(data, date_col, score_col, target_col, periods, weight_col=None):
    """
    Compute Gini for every period window. Returns tidy DataFrame.
    Gini = 2*AUC - 1.  weight_col is optional sample weighting.
    """
    dt = data[data[target_col].notna()].copy()
    dt[date_col] = pd.to_datetime(dt[date_col]).dt.date
    rows = []

    for period_name, info in periods.items():
        start = pd.to_datetime(info["start"]).date()
        end   = pd.to_datetime(info["end"]).date()
        sub   = dt[(dt[date_col] >= start) & (dt[date_col] <= end)].copy()

        n = sub["ee_customer_id"].nunique()
        if n == 0:
            rows.append({"Period": period_name, "Start_Date": start,
                         "End_Date": end, "Sample_Size": 0,
                         "Bad_Rate_Pct": None, "Gini": None})
            continue

        bad_rate = 100 * (
            1 - sub[["ee_customer_id", target_col]]
            .drop_duplicates()[target_col].sum() / n
        )

        gini = None
        if sub[target_col].nunique() >= 2:
            try:
                kw = {"sample_weight": sub[weight_col]} if weight_col else {}
                auc  = roc_auc_score(sub[target_col], sub[score_col], **kw)
                gini = round(2 * auc - 1, 4)
            except Exception as e:
                log(f"    Gini error [{period_name}]: {e}")

        rows.append({
            "Period": period_name, "Start_Date": start, "End_Date": end,
            "Sample_Size": n, "Bad_Rate_Pct": round(bad_rate, 2), "Gini": gini
        })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 1 — LOAD ALL RAW DATA
# =============================================================================

log("=== STEP 1: Loading raw data ===")
bq = bigquery.Client(PROJECT_ID)

log("  prod API scores (new users)...")
_prod_api_raw = bq.query("""
    SELECT
        employee_id                        AS ee_customer_id,
        run_date,
        ee_attrition_risk_segment          AS attrition_risk_segment,
        ee_attrition_time_to_leave         AS attrition_time_to_leave,
        oop_score                          AS oop_score_prod,
        oop_risk_segment                   AS oop_risk_segment_prod
    FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table_api`
""").to_dataframe()

log("  prod batch scores (existing users)...")
_prod_batch_raw = bq.query("""
    SELECT
        employee_id                        AS ee_customer_id,
        run_date,
        ee_attrition_risk_segment          AS attrition_risk_segment,
        ee_attrition_time_to_leave         AS attrition_time_to_leave,
        oop_score                          AS oop_score_prod,
        oop_risk_segment                   AS oop_risk_segment_prod
    FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table`
""").to_dataframe()

log("  OOP matured targets...")
dt_oop_targets = bq.query("""
    SELECT user_id AS ee_customer_id, target AS oop_target
    FROM `prj-prod-dataplatform.tendo_mart.tendo_collection_target_master`
    WHERE target_maturity_flag = 1
""").to_dataframe()
dt_oop_targets["ee_customer_id"] = dt_oop_targets["ee_customer_id"].astype(str)

log("  backscored OOP — new users...")
dt_bs_oop_new = bq.query("""
    SELECT *
    FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_new_users_jan23_jan26_20260201_oop_with_osbal`
""").to_dataframe()
dt_bs_oop_new["ee_customer_id"] = dt_bs_oop_new["ee_customer_id"].astype(str)

log("  backscored OOP — existing users...")
dt_bs_oop_ex = bq.query("""
    SELECT *
    FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_existing_users_jan23_jan26_20260201_oop`
""").to_dataframe()
dt_bs_oop_ex["ee_customer_id"] = dt_bs_oop_ex["ee_customer_id"].astype(str)
dt_bs_oop_ex["calc_date"] = pd.to_datetime(dt_bs_oop_ex["calc_date"], errors="coerce")
dt_bs_oop_ex["calc_date_correct"] = dt_bs_oop_ex["calc_date"] - pd.DateOffset(months=1)
dt_bs_oop_ex["calc_date_ym"] = (
    dt_bs_oop_ex["calc_date_correct"].dt.year * 100
    + dt_bs_oop_ex["calc_date_correct"].dt.month
)

log("  backscored attrition...")
dt_bs_attr = bq.query("""
    SELECT *
    FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_jan24_jan26_20260201_attrition`
""").to_dataframe()
dt_bs_attr["ee_customer_id"] = dt_bs_attr["ee_customer_id"].astype(str)
dt_bs_attr["calc_date"] = pd.to_datetime(dt_bs_attr["calc_date"], errors="coerce")
dt_bs_attr["calc_date_correct"] = dt_bs_attr["calc_date"] - pd.DateOffset(months=1)
dt_bs_attr["calc_date_ym"] = (
    dt_bs_attr["calc_date_correct"].dt.year * 100
    + dt_bs_attr["calc_date_correct"].dt.month
)
dt_bs_attr["is_new_customer_flag_1m"] = (
    (dt_bs_attr["ee_onboarding_month"] == dt_bs_attr["calc_date_ym"]).astype(int)
)

log("  raw features (GCS)...")
dt_raw = pd.read_pickle(generate_bucket_url("Oleh/tendo/data/raw_data_14012026.pkl", GS_BUCKET))
dt_raw["ee_customer_id"] = dt_raw["ee_customer_id"].astype(str)
dt_raw["ee_onboarding_date"] = pd.to_datetime(dt_raw["ee_onboarding_date"]).dt.tz_localize(None)

# Keep only the two columns we need from dt_raw to avoid any future collision
dt_raw_slim = dt_raw[["ee_customer_id", "ee_onboarding_date"]].drop_duplicates()

log("  resignation data (GCS)...")
dt_res = pd.read_pickle(generate_bucket_url("Oleh/tendo/data/resignation_data_14012026.pkl", GS_BUCKET))
dt_res["ee_customer_id"] = dt_res["ee_customer_id"].astype(str)

log("All raw data loaded.")


# =============================================================================
# STEP 2 — BUILD CLEAN BASE FRAMES
# These are the starting points for every section below.
# We never mutate these — each section .copy()s from them.
# =============================================================================

log("=== STEP 2: Building clean base frames ===")

# ── Prod API base ─────────────────────────────────────────────────────────────
# Add date/YM fields. Merge onboarding date once here so it's always present.
prod_api_base = _prod_api_raw.copy()
prod_api_base["ee_customer_id"] = prod_api_base["ee_customer_id"].astype(str)
prod_api_base["run_date"] = pd.to_datetime(prod_api_base["run_date"]).dt.normalize()
prod_api_base["run_date_ym"] = (
    prod_api_base["run_date"].dt.year * 100 + prod_api_base["run_date"].dt.month
)
# Attach onboarding date to the base frame once
prod_api_base = prod_api_base.merge(dt_raw_slim, how="left", on="ee_customer_id")
prod_api_base["ee_onboarding_date"] = pd.to_datetime(
    prod_api_base["ee_onboarding_date"]
).dt.normalize()
prod_api_base["onb_rd_diff"] = (
    abs(prod_api_base["run_date"] - prod_api_base["ee_onboarding_date"])
).dt.days
prod_api_base["onboarding_date_ym"] = (
    prod_api_base["ee_onboarding_date"].dt.year * 100
    + prod_api_base["ee_onboarding_date"].dt.month
)

# ── Prod batch base ───────────────────────────────────────────────────────────
prod_batch_base = _prod_batch_raw.copy()
prod_batch_base["ee_customer_id"] = prod_batch_base["ee_customer_id"].astype(str)
prod_batch_base = prod_batch_base.drop_duplicates(
    subset=["ee_customer_id", "run_date", "attrition_time_to_leave", "oop_score_prod"]
)
prod_batch_base["run_date"] = pd.to_datetime(prod_batch_base["run_date"], errors="coerce")
prod_batch_base["run_date_ym"] = (
    prod_batch_base["run_date"].dt.year * 100 + prod_batch_base["run_date"].dt.month
)
# Attach onboarding date to the base frame once
prod_batch_base = prod_batch_base.merge(dt_raw_slim, how="left", on="ee_customer_id")
prod_batch_base["ee_onboarding_date"] = pd.to_datetime(
    prod_batch_base["ee_onboarding_date"]
).dt.normalize()
prod_batch_base["onboarding_date_ym"] = (
    prod_batch_base["ee_onboarding_date"].dt.year * 100
    + prod_batch_base["ee_onboarding_date"].dt.month
)
prod_batch_base["onb_rd_diff"] = (
    abs(prod_batch_base["run_date"] - prod_batch_base["ee_onboarding_date"])
).dt.days

log("  prod_api_base and prod_batch_base ready.")
log(f"  prod_api_base columns: {list(prod_api_base.columns)}")
log(f"  prod_batch_base columns: {list(prod_batch_base.columns)}")


# =============================================================================
# STEP 3 — OOP MASTER FRAMES  (pages 2, 3, 4)
# =============================================================================

log("=== STEP 3: Building OOP master frames ===")

# ── OOP New Users ─────────────────────────────────────────────────────────────
oop_new = (
    prod_api_base.copy()
    .merge(dt_oop_targets, how="left", on="ee_customer_id")
    .merge(
        dt_bs_oop_new[[
            "ee_customer_id", "score_oop",
            "osbal_as_of_resignation_date",
            "osbal_as_of_oop_eligible_date",
            "osbal_as_of_current_date",
        ]],
        how="left", on="ee_customer_id",
    )
)
oop_new["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    oop_new["osbal_as_of_oop_eligible_date"]
)

# One row per customer, closest-to-onboarding observation
oop_new_calc = (
    oop_new.dropna(subset=["ee_onboarding_date", "oop_target"])
    .sort_values(["onb_rd_diff", "run_date"])
    .drop_duplicates(subset=["ee_customer_id"], keep="first")
    .copy()
)
log(f"  oop_new_calc: {len(oop_new_calc):,} rows")

# ── OOP Existing Users ────────────────────────────────────────────────────────
oop_existing = (
    prod_batch_base.copy()
    .merge(dt_oop_targets, how="left", on="ee_customer_id")
    .merge(
        dt_bs_oop_new[[
            "ee_customer_id",
            "osbal_as_of_resignation_date",
            "osbal_as_of_oop_eligible_date",
            "osbal_as_of_current_date",
        ]],
        how="left", on="ee_customer_id",
    )
    .merge(
        dt_bs_oop_ex[["ee_customer_id", "calc_date_ym", "score_oop"]],
        how="left",
        left_on=["ee_customer_id", "run_date_ym"],
        right_on=["ee_customer_id", "calc_date_ym"],
    )
)
oop_existing["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    oop_existing["osbal_as_of_oop_eligible_date"]
)

oop_existing_calc = (
    oop_existing.dropna(subset=["ee_onboarding_date", "oop_target"])
    .sort_values(["ee_customer_id", "run_date"])
    .copy()
)
log(f"  oop_existing_calc: {len(oop_existing_calc):,} rows")


# =============================================================================
# STEP 4 — ATTRITION MASTER FRAMES  (pages 2, 5, 6)
# =============================================================================

log("=== STEP 4: Building Attrition master frames ===")

def build_attrition_targets(df, score_date_col):
    """
    Add time_to_attrition, attrition_event, score_attr_corrected to a frame
    that already has ee_resignation_date_correct from dt_res.
    score_date_col is the column to measure 'months until resignation' from.
    """
    df = df.copy()
    resign_dt = pd.to_datetime(df["ee_resignation_date_correct"], errors="coerce")
    score_dt  = pd.to_datetime(df[score_date_col],               errors="coerce")

    months_diff = (
        (resign_dt.dt.year  - score_dt.dt.year)  * 12
      + (resign_dt.dt.month - score_dt.dt.month)
    )
    df["time_to_attrition"]   = np.where(resign_dt.isna(), np.nan, months_diff)
    df["attrition_event"]     = resign_dt.notna().astype(int)
    df["score_attr_corrected"] = df["score_attr"].replace(np.inf, 15)
    return df


# ── Attrition New Users — Prod v1.0 ───────────────────────────────────────────
# Merge chain: prod_api_base → dt_res → dt_bs_attr (matched by run_date_ym)
# dt_res may share column names with prod_api_base — handle via suffix rename
attr_api_tmp = prod_api_base.copy().merge(
    dt_res, how="left", on="ee_customer_id", suffixes=("", "_res")
)
# Drop any _res-suffixed duplicates (dt_res columns that already exist in prod_api_base)
attr_api_tmp = attr_api_tmp[[c for c in attr_api_tmp.columns if not c.endswith("_res")]]

attr_api_tmp = attr_api_tmp.merge(
    dt_bs_attr[[
        "ee_customer_id", "calc_date_ym",
        "score_attr", "score_attr_segment", "is_new_customer_flag_1m",
    ]],
    how="left",
    left_on=["ee_customer_id", "run_date_ym"],
    right_on=["ee_customer_id", "calc_date_ym"],
    suffixes=("", "_bsattr"),
)
attr_api_tmp = attr_api_tmp[[c for c in attr_api_tmp.columns if not c.endswith("_bsattr")]]

# ee_onboarding_date already in prod_api_base — just dropna and dedup
attr_prod_api_calc = (
    attr_api_tmp
    .dropna(subset=["ee_onboarding_date"])   # ← was the KeyError line
    .sort_values(["onb_rd_diff", "run_date"])
    .drop_duplicates(subset=["ee_customer_id"], keep="first")
    .copy()
)
attr_prod_api_calc = build_attrition_targets(attr_prod_api_calc, "run_date")

# Production segment labels
attr_prod_api_calc["attrition_score_prod"] = (
    attr_prod_api_calc["attrition_time_to_leave"]
    .replace("12+", "15")
    .astype("float")
)
attr_prod_api_calc["attrition_risk_segment_prod"] = (
    attr_prod_api_calc["attrition_score_prod"].replace(ATTRITION_SCORE_MAP)
)
log(f"  attr_prod_api_calc (new, prod): {len(attr_prod_api_calc):,} rows")

# ── Attrition New Users — BS v2.0 ─────────────────────────────────────────────
attr_bs_new_tmp = dt_bs_attr.merge(
    dt_res, how="left", on="ee_customer_id", suffixes=("", "_res")
)
attr_bs_new_tmp = attr_bs_new_tmp[[c for c in attr_bs_new_tmp.columns if not c.endswith("_res")]]

attr_bs_new_calc = (
    attr_bs_new_tmp.query("is_new_customer_flag_1m == 1").copy()
)
attr_bs_new_calc = build_attrition_targets(attr_bs_new_calc, "calc_date_correct")
log(f"  attr_bs_new_calc (new, BS): {len(attr_bs_new_calc):,} rows")

# ── Attrition Existing Users — Prod v1.0 ──────────────────────────────────────
attr_batch_tmp = prod_batch_base.copy().merge(
    dt_res, how="left", on="ee_customer_id", suffixes=("", "_res")
)
attr_batch_tmp = attr_batch_tmp[[c for c in attr_batch_tmp.columns if not c.endswith("_res")]]

attr_batch_tmp = attr_batch_tmp.merge(
    dt_bs_attr[[
        "ee_customer_id", "calc_date_ym",
        "score_attr", "score_attr_segment", "is_new_customer_flag_1m",
    ]],
    how="left",
    left_on=["ee_customer_id", "run_date_ym"],
    right_on=["ee_customer_id", "calc_date_ym"],
    suffixes=("", "_bsattr"),
)
attr_batch_tmp = attr_batch_tmp[[c for c in attr_batch_tmp.columns if not c.endswith("_bsattr")]]

attr_prod_batch_calc = (
    attr_batch_tmp
    .dropna(subset=["ee_onboarding_date"])
    .copy()
)
attr_prod_batch_calc = build_attrition_targets(attr_prod_batch_calc, "run_date")
attr_prod_batch_calc["attrition_score_prod"] = (
    attr_prod_batch_calc["attrition_time_to_leave"]
    .replace("12+", "15")
    .astype("float")
)
attr_prod_batch_calc["attrition_risk_segment_prod"] = (
    attr_prod_batch_calc["attrition_score_prod"].replace(ATTRITION_SCORE_MAP)
)
log(f"  attr_prod_batch_calc (existing, prod): {len(attr_prod_batch_calc):,} rows")

# ── Attrition Existing Users — BS v2.0 ────────────────────────────────────────
attr_bs_existing_tmp = dt_bs_attr.merge(
    dt_res, how="left", on="ee_customer_id", suffixes=("", "_res")
)
attr_bs_existing_tmp = attr_bs_existing_tmp[
    [c for c in attr_bs_existing_tmp.columns if not c.endswith("_res")]
]
attr_bs_existing_calc = (
    attr_bs_existing_tmp.query("is_new_customer_flag_3m == 0").copy()
)
attr_bs_existing_calc = build_attrition_targets(attr_bs_existing_calc, "calc_date_correct")
log(f"  attr_bs_existing_calc (existing, BS): {len(attr_bs_existing_calc):,} rows")

log("All master frames ready.")


# =============================================================================
# PAGE 2 — MODEL PERFORMANCE: Gini & C-Index
# =============================================================================

log("=== PAGE 2: Model Performance Gini / C-Index ===")

MON_WINDOW = {
    "Jul 2025":      {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025":      {"start": "2025-08-01", "end": "2025-08-31"},
    "Jul-Aug 2025":  {"start": "2025-07-01", "end": "2025-08-31"},
}

def gini_df(data, date_col, score_col, target_col, model, version, user_type, periods, weight_col=None):
    g = calc_gini(data, date_col, score_col, target_col, periods, weight_col)
    g["Model"]        = model
    g["ModelVersion"] = version
    g["UserType"]     = user_type
    return g

pieces = [
    gini_df(oop_new_calc, "ee_onboarding_date", "oop_score_prod", "oop_target",
            "OOP Repay", "v1.0 Prod", "New users (< 3MOB)", MON_WINDOW),
    gini_df(oop_new_calc[oop_new_calc["score_oop"].notna()],
            "ee_onboarding_date", "score_oop", "oop_target",
            "OOP Repay", "v2.0 BS", "New users (< 3MOB)", MON_WINDOW),
    gini_df(oop_existing_calc, "run_date", "oop_score_prod", "oop_target",
            "OOP Repay", "v1.0 Prod", "Existing users (>= 3MOB)", MON_WINDOW),
    gini_df(oop_existing_calc[oop_existing_calc["score_oop"].notna()],
            "run_date", "score_oop", "oop_target",
            "OOP Repay", "v2.0 BS", "Existing users (>= 3MOB)", MON_WINDOW),
    gini_df(attr_prod_api_calc[attr_prod_api_calc["attrition_score_prod"].notna()],
            "run_date", "attrition_score_prod", "attrition_event",
            "Attrition", "v1.0 Prod", "New users (< 3MOB)", MON_WINDOW),
    gini_df(attr_bs_new_calc[attr_bs_new_calc["score_attr_corrected"].notna()],
            "calc_date_correct", "score_attr_corrected", "attrition_event",
            "Attrition", "v2.0 BS", "New users (< 3MOB)", MON_WINDOW),
    gini_df(attr_prod_batch_calc[attr_prod_batch_calc["attrition_score_prod"].notna()],
            "run_date", "attrition_score_prod", "attrition_event",
            "Attrition", "v1.0 Prod", "Existing users (>= 3MOB)", MON_WINDOW),
    gini_df(attr_bs_existing_calc[attr_bs_existing_calc["score_attr_corrected"].notna()],
            "calc_date_correct", "score_attr_corrected", "attrition_event",
            "Attrition", "v2.0 BS", "Existing users (>= 3MOB)", MON_WINDOW),
]

# Fixed Dev numbers from the original slide (do not change month-to-month)
dev_rows = pd.DataFrame([
    {"UserType": "New users (< 3MOB)",       "Model": "OOP Repay",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "Gini": 0.30},
    {"UserType": "Existing users (>= 3MOB)", "Model": "OOP Repay",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "Gini": 0.32},
    {"UserType": "New users (< 3MOB)",       "Model": "Attrition",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "Gini": 0.66},
    {"UserType": "Existing users (>= 3MOB)", "Model": "Attrition",  "ModelVersion": "v1.0 Prod", "Period": "Dev", "Gini": 0.64},
])

page2 = pd.concat(pieces + [dev_rows], ignore_index=True)
page2["MetricName"] = page2["Model"].map({"OOP Repay": "Gini", "Attrition": "C-Index"})
page2["RunMonth"]   = RUN_MONTH
save_csv(page2, "page02_model_performance_gini_cindex.csv")


# =============================================================================
# PAGE 3 — OOP Rank Order: New Users
# =============================================================================

log("=== PAGE 3: OOP Rank Order — New Users ===")

def derive_bs_segment_labels(df, score_col, prod_segment_col):
    """
    Re-derive production segment bin edges from observed prod scores,
    then cut the backscored score into the same A-F segments.
    Mirrors the bin-derivation logic in the original notebook exactly.
    """
    valid = df[df[prod_segment_col].notna() & df[score_col].notna()].copy()
    valid["_seg"]   = valid[prod_segment_col].astype(str)
    valid["_score"] = pd.to_numeric(valid[score_col], errors="coerce")

    mm = (
        valid.groupby("_seg")["_score"]
        .agg(["min", "max"])
        .reindex(list("ABCDEF"))
        .clip(0, 1)
        .interpolate(limit_direction="both")
    )
    order = list("ABCDEF")
    cuts = pd.Series(
        [(mm.loc[h, "min"] + mm.loc[l, "max"]) / 2
         for h, l in zip(order[:-1], order[1:])],
        index=[f"{h}/{l}" for h, l in zip(order[:-1], order[1:])]
    ).cummin()
    edges  = [0.0] + cuts.iloc[::-1].tolist() + [np.nextafter(1.0, 2.0)]
    labels = list("FEDCBA")
    return pd.cut(df[score_col], bins=edges, labels=labels,
                  right=False, include_lowest=True)


def build_oop_pivot(df, segment_col, filter_expr, model_label, period_label):
    """Compute the OOP rank-order pivot table for one model/period."""
    sub = df.query(filter_expr).copy() if filter_expr else df.copy()
    sub["oop_target_bad"]    = 1 - sub["oop_target"]
    sub["osbal_current_bad"] = sub["oop_target_bad"] * sub["osbal_as_of_current_date"]

    pt = (
        sub.groupby(segment_col, observed=True)
        .agg(
            Count                  = ("ee_customer_id", "count"),
            Bad_Count              = ("oop_target_bad", "sum"),
            Sum_Osbal_Resignation  = ("osbal_as_of_resignation_date", "sum"),
            Sum_Osbal_Bads_Today   = ("osbal_current_bad", "sum"),
        )
        .reset_index()
        .rename(columns={segment_col: "RiskGroup"})
    )
    pt["Bad_Rate_Pct"]         = (pt["Bad_Count"] / pt["Count"] * 100).round(1)
    pt["Outstanding_Bad_Rate"] = (
        pt["Sum_Osbal_Bads_Today"] / pt["Sum_Osbal_Resignation"] * 100
    ).round(1)
    pt["ModelVersion"] = model_label
    pt["Period"]       = period_label
    pt["Segment_Order"] = pt["RiskGroup"].astype(str).map(OOP_SEGMENT_ORDER)
    pt["RunMonth"]      = RUN_MONTH
    return pt.drop(columns=["Bad_Count"])


# Derive BS segment labels for new users using prod boundaries
oop_new_calc["oop_risk_segment_bs"] = derive_bs_segment_labels(
    oop_new_calc, "score_oop", "oop_risk_segment_prod"
)

p3_v1 = build_oop_pivot(
    oop_new_calc, "oop_risk_segment_prod",
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508",
    "OOP v1.0 Prod", "Jun-Aug 2025"
)
p3_v2 = build_oop_pivot(
    oop_new_calc[oop_new_calc["score_oop"].notna()],
    "oop_risk_segment_bs",
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508",
    "OOP v2.0 BS", "Jun-Aug 2025"
)
page3 = pd.concat([p3_v1, p3_v2], ignore_index=True).sort_values(
    ["ModelVersion", "Segment_Order"]
)
save_csv(page3, "page03_oop_rank_order_new_users.csv")


# =============================================================================
# PAGE 4 — OOP Rank Order: Existing Users
# =============================================================================

log("=== PAGE 4: OOP Rank Order — Existing Users ===")

oop_existing_calc["oop_risk_segment_bs"] = derive_bs_segment_labels(
    oop_existing_calc, "score_oop", "oop_risk_segment_prod"
)

p4_v1 = build_oop_pivot(
    oop_existing_calc, "oop_risk_segment_prod",
    "run_date_ym == 202507",
    "OOP v1.0 Prod", "Jul 2025"
)
p4_v2 = build_oop_pivot(
    oop_existing_calc[oop_existing_calc["score_oop"].notna()],
    "oop_risk_segment_bs",
    "run_date_ym == 202507",
    "OOP v2.0 BS", "Jul 2025"
)
page4 = pd.concat([p4_v1, p4_v2], ignore_index=True).sort_values(
    ["ModelVersion", "Segment_Order"]
)
save_csv(page4, "page04_oop_rank_order_existing_users.csv")


# =============================================================================
# PAGE 5 — Attrition Rank Order: New Users
# =============================================================================

log("=== PAGE 5: Attrition Rank Order — New Users ===")

def build_attrition_pivot(df, segment_col, filter_expr, model_label, period_label):
    sub = df.query(filter_expr).copy() if filter_expr else df.copy()
    pt = (
        sub.groupby(segment_col, observed=True)
        .agg(
            Count            = ("ee_customer_id", "count"),
            Attrition_Rate   = ("attrition_event", "mean"),
            Expected_Avg_TTA = ("score_attr_corrected", "mean"),
            Actual_Avg_TTA   = ("time_to_attrition", "mean"),
        )
        .reset_index()
        .rename(columns={segment_col: "AttritionGroup"})
    )
    pt["Attrition_Rate_Pct"]  = (pt["Attrition_Rate"] * 100).round(1)
    pt["Expected_Avg_TTA"]    = pt["Expected_Avg_TTA"].round(1)
    pt["Actual_Avg_TTA"]      = pt["Actual_Avg_TTA"].round(1)
    pt["ModelVersion"]         = model_label
    pt["Period"]               = period_label
    pt["Segment_Order"]        = pt["AttritionGroup"].map(ATTRITION_ORDER)
    pt["RunMonth"]             = RUN_MONTH
    return pt.drop(columns=["Attrition_Rate"])


# Prod v1.0 — prod segment (attrition_risk_segment_prod), Jun-Aug onboarding
p5_v1 = build_attrition_pivot(
    attr_prod_api_calc, "attrition_risk_segment_prod",
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508",
    "Attrition v1.0 Prod", "Jun-Aug 2025"
)

# BS v2.0 — BS segment (score_attr_segment), Jun-Aug onboarding month
p5_v2 = build_attrition_pivot(
    attr_bs_new_calc, "score_attr_segment",
    "ee_onboarding_month >= 202506 & ee_onboarding_month <= 202508",
    "Attrition v2.0 BS", "Jun-Aug 2025"
)

page5 = pd.concat([p5_v1, p5_v2], ignore_index=True).sort_values(
    ["ModelVersion", "Segment_Order"]
)
save_csv(page5, "page05_attrition_rank_order_new_users.csv")


# =============================================================================
# PAGE 6 — Attrition Rank Order: Existing Users
# =============================================================================

log("=== PAGE 6: Attrition Rank Order — Existing Users ===")

# Prod v1.0 — Jul 2025 snapshot
p6_v1 = build_attrition_pivot(
    attr_prod_batch_calc.query("run_date_ym == 202507"),
    "attrition_risk_segment_prod", None,
    "Attrition v1.0 Prod", "Jul 2025"
)

# BS v2.0 — Jul 2025 snapshot (existing = is_new_customer_flag_3m == 0)
p6_v2 = build_attrition_pivot(
    attr_bs_existing_calc.query("calc_date_ym == 202507"),
    "score_attr_segment", None,
    "Attrition v2.0 BS", "Jul 2025"
)

page6 = pd.concat([p6_v1, p6_v2], ignore_index=True).sort_values(
    ["ModelVersion", "Segment_Order"]
)
save_csv(page6, "page06_attrition_rank_order_existing_users.csv")


# =============================================================================
# PAGES 7–11 — BigQuery direct queries
# NOTE: Adjust table names below to match your exact schema.
# The logic mirrors the slide data precisely.
# =============================================================================

log("=== PAGE 7: Unit Bad Rate MoM ===")
dt_unit_bad = bq.query("""
    WITH base AS (
        SELECT
            sc.employee_id                                              AS ee_customer_id,
            sc.scorecard_version,
            FORMAT_DATE('%Y-%m', sc.onboarding_date)                   AS onboarding_month,
            CASE WHEN res.ee_resignation_date_correct <= DATE '2026-01-31'
                 THEN 1 ELSE 0 END                                      AS left_job,
            CASE WHEN osbal.osbal_as_of_resignation_date > 0
                 THEN 1 ELSE 0 END                                      AS had_oop_outstanding,
            tgt.target                                                  AS oop_target
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
        scorecard_version                                                          AS Scorecard,
        onboarding_month                                                           AS OnboardingMonth,
        COUNT(DISTINCT ee_customer_id)                                             AS CntOnboarded,
        SUM(left_job)                                                              AS CntLeftJob,
        SUM(CASE WHEN left_job=1 AND had_oop_outstanding=1 THEN 1 ELSE 0 END)     AS CntHadOopOutstanding,
        SUM(CASE WHEN left_job=1 AND had_oop_outstanding=1
                      AND oop_target=0 THEN 1 ELSE 0 END)                          AS CntOopFlagBad,
        SAFE_DIVIDE(
            SUM(CASE WHEN left_job=1 AND had_oop_outstanding=1
                          AND oop_target=0 THEN 1 ELSE 0 END),
            NULLIF(SUM(left_job), 0)
        )                                                                          AS UnitBadRate
    FROM base
    GROUP BY 1, 2
    ORDER BY 1, 2
""").to_dataframe()
dt_unit_bad["UnitBadRate_Pct"] = (dt_unit_bad["UnitBadRate"] * 100).round(2)
dt_unit_bad["RunMonth"] = RUN_MONTH
save_csv(dt_unit_bad, "page07_unit_bad_rate_mom.csv")


log("=== PAGE 8: Peso Bad Rate MoM ===")
dt_peso_bad = bq.query("""
    WITH base AS (
        SELECT
            sc.employee_id                                              AS ee_customer_id,
            sc.scorecard_version,
            FORMAT_DATE('%Y-%m', sc.onboarding_date)                   AS onboarding_month,
            sc.credit_limit                                            AS cl_amt,
            osbal.osbal_as_of_resignation_date,
            osbal.osbal_as_of_oop_eligible_date,
            osbal.osbal_as_of_current_date,
            CASE WHEN res.ee_resignation_date_correct <= DATE '2026-01-31'
                 THEN 1 ELSE 0 END                                      AS left_job,
            tgt.target                                                  AS oop_target
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
        scorecard_version                                                                         AS Scorecard,
        onboarding_month                                                                          AS OnboardingMonth,
        SUM(cl_amt)                                                                               AS TotCLAmt,
        SUM(CASE WHEN left_job=1 THEN osbal_as_of_resignation_date  ELSE 0 END)                  AS TotOsAsOfResignation,
        SUM(CASE WHEN left_job=1 THEN osbal_as_of_oop_eligible_date ELSE 0 END)                  AS TotOsAsOfOopEligible,
        SUM(CASE WHEN left_job=1 AND oop_target=0 THEN osbal_as_of_current_date ELSE 0 END)      AS TotOsFromBadCustomers,
        SAFE_DIVIDE(
            SUM(CASE WHEN left_job=1 AND oop_target=0 THEN osbal_as_of_current_date ELSE 0 END),
            NULLIF(SUM(CASE WHEN left_job=1 THEN osbal_as_of_oop_eligible_date ELSE 0 END), 0)
        )                                                                                         AS PesoBadRate
    FROM base
    GROUP BY 1, 2
    ORDER BY 1, 2
""").to_dataframe()
dt_peso_bad["PesoBadRate_Pct"] = (dt_peso_bad["PesoBadRate"] * 100).round(2)
dt_peso_bad["RunMonth"] = RUN_MONTH
save_csv(dt_peso_bad, "page08_peso_bad_rate_mom.csv")


log("=== PAGE 9: CL+ Exposure ===")
dt_cl_plus = bq.query("""
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
        WHERE is_existing_user = TRUE AND cl_plus_start_date IS NOT NULL
    )
    SELECT
        cl_category                                             AS CLCategory,
        COUNT(DISTINCT user_id)                                AS UserCount,
        SUM(old_cl)                                            AS SumOldCL,
        SUM(new_cl)                                            AS SumNewCL,
        SUM(cl_delta)                                          AS CLDelta,
        SAFE_DIVIDE(SUM(cl_delta), NULLIF(SUM(old_cl), 0))    AS PctCLIncrease
    FROM cl_deltas
    GROUP BY 1
    ORDER BY 1
""").to_dataframe()
dt_cl_plus["PctCLIncrease_Pct"] = (dt_cl_plus["PctCLIncrease"] * 100).round(1)
dt_cl_plus["RunMonth"] = RUN_MONTH
save_csv(dt_cl_plus, "page09_cl_plus_exposure.csv")


log("=== PAGE 10: New Borrower Exposure ===")
dt_cohort = bq.query("""
    SELECT
        CASE
            WHEN onboarding_date < DATE '2025-01-01'                           THEN 'Onboarded before'
            WHEN onboarding_date BETWEEN DATE '2025-01-01' AND DATE '2025-03-31' THEN 'Onboarded Q1 2025'
            WHEN onboarding_date BETWEEN DATE '2025-04-01' AND DATE '2025-06-30' THEN 'Onboarded Q2 2025'
            WHEN onboarding_date BETWEEN DATE '2025-07-01' AND DATE '2025-09-30' THEN 'Onboarded Q3 2025'
            WHEN onboarding_date BETWEEN DATE '2025-10-01' AND DATE '2025-12-31' THEN 'Onboarded Q4 2025'
            ELSE 'Onboarded 2026'
        END                             AS OnboardingCohort,
        COUNT(DISTINCT user_id)         AS UserCount,
        SUM(credit_limit)               AS SumNewCL,
        'CohortCL'                      AS TableType
    FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table_api`
    GROUP BY 1
    ORDER BY MIN(onboarding_date)
""").to_dataframe()

dt_user_mix = bq.query("""
    SELECT
        CASE
            WHEN is_new_user AND has_cl_plus     THEN 'new_user_w_cl+'
            WHEN is_new_user AND NOT has_cl_plus THEN 'new_user_w_no_cl+'
            WHEN NOT is_new_user AND has_cl_plus THEN 'old_user_w_cl+'
            ELSE 'old_user_w_no_cl+'
        END                                                             AS UserType,
        COUNT(DISTINCT user_id)                                         AS UserCount,
        SAFE_DIVIDE(COUNT(DISTINCT user_id),
            SUM(COUNT(DISTINCT user_id)) OVER ())                       AS Pct,
        'UserMix'                                                       AS TableType
    FROM `prj-prod-dataplatform.tendo_mart.tendo_user_cl_plus_summary`
    GROUP BY 1
    ORDER BY 1
""").to_dataframe()
dt_user_mix["Pct_Pct"] = (dt_user_mix["Pct"] * 100).round(1)

page10 = pd.concat([dt_cohort, dt_user_mix], ignore_index=True)
page10["RunMonth"] = RUN_MONTH
save_csv(page10, "page10_new_borrower_exposure.csv")


log("=== PAGE 11: Utilization Trend ===")
dt_util = bq.query("""
    SELECT
        FORMAT_DATE('%Y-%m', report_month_end)      AS ReportMonth,
        SUM(credit_limit)                           AS CreditLimit,
        SUM(unpaid_principal_balance)               AS UnpaidPrincipal,
        SAFE_DIVIDE(
            SUM(unpaid_principal_balance),
            NULLIF(SUM(credit_limit), 0)
        )                                           AS UtilizationRate
    FROM `prj-prod-dataplatform.tendo_mart.portfolio_monthly_snapshot`
    WHERE report_month_end BETWEEN DATE '2025-07-01' AND CURRENT_DATE()
    GROUP BY 1
    ORDER BY 1
""").to_dataframe()
dt_util["UtilizationRate_Pct"] = (dt_util["UtilizationRate"] * 100).round(1)
dt_util["RunMonth"] = RUN_MONTH
save_csv(dt_util, "page11_utilization_trend.csv")


# =============================================================================
# DONE
# =============================================================================

log("=" * 60)
log(f"All CSVs written to: {OUTPUT_DIR}")
log("Next steps:")
log("  1. First run: Power BI > Get Data > CSV > select each file")
log("  2. Monthly: run this script, then click Refresh in Power BI")
log("=" * 60)
