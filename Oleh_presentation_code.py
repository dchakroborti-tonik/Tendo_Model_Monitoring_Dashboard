# %% [markdown]
# https://tonikph-my.sharepoint.com/:p:/g/personal/bbanik_tonikbank_com/IQAvsct_VJYKRoYRQISh8jy1AWHNdHcuDqk9bIfwQjpiW6M?e=wvYx82

# %% [markdown]
# # Import packages

# %%

import io
import os
import pickle
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from typing import Union

import duckdb as dd
import gcsfs
import joblib
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# # Jupyter Notebook Loading Header
#
# This is a custom loading header for Jupyter Notebooks in Visual Studio Code.
# It includes common imports and settings to get you started quickly.
# %% [markdown]
## Import Libraries
import pandas as pd
import seaborn as sns
from google.cloud import bigquery, storage
from sklearn.metrics import roc_auc_score

path = r"C:\Users\Dwaipayan\AppData\Roaming\gcloud\legacy_credentials\dchakroborti@tonikbank.com\adc.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
client = bigquery.Client(project="prj-prod-dataplatform")
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"
# %% [markdown]
## Configure Settings
# Set options or configurations as needed
pd.set_option("display.max_columns", None)
pd.set_option("Display.max_rows", 100)
sns.set_style("whitegrid")

import os
import pickle

import numpy as np

# %%
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import roc_auc_score

# %% [markdown]
# # Settings

# %%
pd.set_option("display.max_columns", 100)

# %%
GS_BUCKET = "prod-asia-southeast1-tonik-aiml-workspace"
PROJECT_ID = "prj-prod-dataplatform"
RANDOM_SEED = 2024

# %% [markdown]
# # Functions


# %%
def define_cat_features(columns, cat_vars):
    return list(set(cat_vars).intersection(columns))


def generate_bucket_url(filename, bucket_name):

    return f"gs://{bucket_name}/{filename}"


def save_to_gcs(data, filename, bucket_name):
    """
    Save data to Google Cloud Storage bucket.

    :param data: The data to save. Can be a string, bytes, or a file-like object.
    :param filename: The name of the file to save in the bucket.
    :param bucket_name: The name of the GCS bucket. Defaults to 'ABC'.
    :return: The public URL of the saved file.
    """
    # Create a client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Create a blob (file) in the bucket
    blob = bucket.blob(filename)

    # Determine the content type and upload accordingly
    if isinstance(data, str):
        blob.upload_from_string(data)
    elif isinstance(data, bytes):
        blob.upload_from_string(data, content_type="application/octet-stream")
    elif hasattr(data, "read"):  # File-like object
        blob.upload_from_file(data)
    else:
        raise ValueError(
            "Unsupported data type. Please provide a string, bytes, or file-like object."
        )

    print(f"File {filename} uploaded to {bucket_name}.")
    return blob.public_url


def load_artifact_from_gcs(filename, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)

    # Download model
    artifact_bytes = blob.download_as_bytes()
    artifact = pickle.loads(artifact_bytes)
    print(f"Model loaded from gs://{bucket_name}/{filename}")

    return artifact


def load_from_gcs(filename, bucket_name, output_type="bytes"):
    """
    Load data from Google Cloud Storage bucket with flexible output formats.

    :param filename: The name of the file to load from the bucket.
    :param bucket_name: The name of the GCS bucket.
    :param output_type: The desired output format. Can be 'bytes', 'string', 'pickle', or 'file'.
                       'bytes': Returns raw bytes
                       'string': Returns decoded string
                       'pickle': Returns unpickled Python object
                       'file': Returns a file-like object for streaming
    :return: The loaded data in the specified format.
    """
    import io
    import pickle

    # Create a client
    client = storage.Client()

    # Get the bucket and blob
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)

    # Handle different output types
    if output_type == "bytes":
        return blob.download_as_bytes()

    elif output_type == "string":
        return blob.download_as_string().decode("utf-8")

    elif output_type == "pickle":
        pickled_data = blob.download_as_bytes()
        return pickle.loads(pickled_data)

    elif output_type == "file":
        file_obj = io.BytesIO()
        blob.download_to_file(file_obj)
        file_obj.seek(0)  # Reset file pointer to beginning
        return file_obj

    else:
        raise ValueError(
            "Unsupported output_type. Must be one of: 'bytes', 'string', 'pickle', 'file'"
        )


def load_artifacts_logreg(exp_number):

    model_filename = f"Oleh/tendo/experiments/{exp_number}/model.pkl"
    model = load_artifact_from_gcs(model_filename, GS_BUCKET)

    feature_filename = f"Oleh/tendo/experiments/{exp_number}/features.pkl"
    features = load_artifact_from_gcs(feature_filename, GS_BUCKET)

    scaler_filename = f"Oleh/tendo/experiments/{exp_number}/scaler.pkl"
    scaler = load_artifact_from_gcs(scaler_filename, GS_BUCKET)

    return model, features, scaler


def save_artifact_to_gcs(artifact, filename, bucket_name):
    """Saves the Cox model to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)

    # Serialize artifact
    artifact_bytes = pickle.dumps(artifact)

    # Upload to GCS
    blob.upload_from_string(artifact_bytes)
    print(f"Artifact saved to gs://{bucket_name}/{filename}")


from typing import Dict

# %%
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import roc_auc_score


def calculate_gini_for_table(
    data: pd.DataFrame,
    date_column: str,
    score_column: str,
    target_column: str,
    data_periods_dict: Dict,
    weights_col: str = None,
):
    """
    Calculate Gini coefficient for different time periods.

    Args:
        project_id: BigQuery project ID
        table_name: Full table name (dataset.table)
        date_column: Name of the date column
        score_column: Name of the score column
        target_column: Name of the target column
        target_maturity_column: Name of the target maturity column
        data_periods_dict: Dictionary with periods, e.g.:
            {'Train': {'start': '2024-01-01', 'end': '2025-01-31'},
             'Test': {'start': '2025-02-01', 'end': '2025-12-31'}}

    Returns:
        pandas.DataFrame: Table with Gini coefficients for each period
    """
    dt = data[data[target_column].notna()].copy()

    # Convert date column to datetime and extract just the date part
    dt[date_column] = pd.to_datetime(dt[date_column]).dt.date

    # Initialize results
    gini_results = []

    print("Gini Coefficient Results:")
    print("=" * 50)

    # Calculate Gini for each period
    for period_name, period_info in data_periods_dict.items():
        start_date = pd.to_datetime(period_info["start"]).date()
        end_date = pd.to_datetime(period_info["end"]).date()

        # Filter data for the current period
        period_mask = (dt[date_column] >= start_date) & (dt[date_column] <= end_date)
        period_data = dt[period_mask].copy()

        sample_size = period_data["ee_customer_id"].nunique()
        bad_rate = 100 * (
            1
            - period_data[["ee_customer_id", target_column]]
            .drop_duplicates()[target_column]
            .sum()
            / sample_size
        )

        if len(period_data) == 0:
            print(
                f"{period_name}: No data available for period {start_date} to {end_date}"
            )
            gini_results.append(
                {
                    "Period": period_name,
                    "Start_Date": start_date,
                    "End_Date": end_date,
                    "Sample_Size": 0,
                    "Bad Rate": np.nan,
                    "Gini_Coefficient": None,
                }
            )
            continue

        # Check if we have both classes (0 and 1) in target
        unique_targets = period_data[target_column].unique()
        if len(unique_targets) < 2:
            print(
                f"{period_name}: Only one class present in target variable. Cannot calculate Gini."
            )
            gini_results.append(
                {
                    "Period": period_name,
                    "Start_Date": start_date,
                    "End_Date": end_date,
                    "Sample_Size": sample_size,
                    "Bad Rate": bad_rate,
                    "Gini_Coefficient": None,
                }
            )
            continue

        # Calculate Gini coefficient
        try:
            if weights_col:
                auc = roc_auc_score(
                    period_data[target_column],
                    period_data[score_column],
                    sample_weight=period_data[weights_col],
                )
                gini = 2 * auc - 1
            else:
                auc = roc_auc_score(
                    period_data[target_column], period_data[score_column]
                )
                gini = 2 * auc - 1

            print(
                f"{period_name}: {round(gini, 4)} (Sample size: {len(period_data):,})"
            )

            gini_results.append(
                {
                    "Period": period_name,
                    "Start_Date": start_date,
                    "End_Date": end_date,
                    "Sample_Size": sample_size,
                    "Bad Rate": bad_rate,
                    "Gini_Coefficient": round(gini, 4),
                }
            )

        except Exception as e:
            print(f"{period_name}: Error calculating Gini - {str(e)}")
            gini_results.append(
                {
                    "Period": period_name,
                    "Start_Date": start_date,
                    "End_Date": end_date,
                    "Sample_Size": sample_size,
                    "Bad Rate": bad_rate,
                    "Gini_Coefficient": None,
                }
            )

    # Create results DataFrame
    results_df = pd.DataFrame(gini_results)

    print("\n" + "=" * 50)
    print("Summary Table:")
    print(results_df.to_string(index=False))

    return results_df


# %% [markdown]
# # Data loading

# %%
client = bigquery.Client(PROJECT_ID)

# %%
# PROD API
sql_query_prod_api = """
SELECT
  employee_id as ee_customer_id,
  run_date,
  ee_attrition_risk_segment as attrition_risk_segment,
  ee_attrition_time_to_leave as attrition_time_to_leave,
  oop_score as oop_score_prod,
  oop_risk_segment as oop_risk_segment_prod
FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table_api`
"""

dt_prod_api = client.query(sql_query_prod_api).to_dataframe()

# %%
dt_prod_api.head(2)

# %%
# PROD BATCH
sql_query_prod_batch = """
SELECT
  employee_id as ee_customer_id,
  run_date,
  ee_attrition_risk_segment as attrition_risk_segment,
  ee_attrition_time_to_leave as attrition_time_to_leave,
  oop_score as oop_score_prod,
  oop_risk_segment as oop_risk_segment_prod
FROM `prj-prod-dataplatform.tendo_mart.tendo_scorecard_master_table`
"""

dt_prod_batch = client.query(sql_query_prod_batch).to_dataframe()

# %%
dt_prod_batch.head(2)

# %%


# %%
# OOP Latest targets
sql_query_oop_targets = """
SELECT
  user_id as ee_customer_id,
  target as oop_target
FROM `prj-prod-dataplatform.tendo_mart.tendo_collection_target_master`
WHERE target_maturity_flag = 1
"""

dt_oop_targets = client.query(sql_query_oop_targets).to_dataframe()

# %%
dt_oop_targets.head(2)

# %%
dt_raw = pd.read_pickle(
    generate_bucket_url("Oleh/tendo/data/raw_data_14012026.pkl", GS_BUCKET)
)

# %%
dt_raw.head(2)

# %%
dt_raw["ee_customer_id"] = dt_raw["ee_customer_id"].astype("str")
dt_raw["ee_onboarding_date"] = pd.to_datetime(
    dt_raw["ee_onboarding_date"]
).dt.tz_localize(None)

# %% [markdown]
# # OOP

# %%
# BS OOP new
sql_query_oop_new = """
SELECT *
FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_new_users_jan23_jan26_20260201_oop_with_osbal`
"""

dt_bs_oop_new = client.query(sql_query_oop_new).to_dataframe()

# BS OOP existing
sql_query_oop_existing = """
SELECT *
FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_existing_users_jan23_jan26_20260201_oop`
"""

dt_bs_oop_ex = client.query(sql_query_oop_existing).to_dataframe()

# %% [markdown]
# `Don't have the osbal table calculation. Need to ask Oleh or understand the process of understanding how to calculate the outstanding balance data`

# %% [markdown]
# Biswa  Oleh, for now just use the scorecard v2.2 as of January Employment date fix for backscoring
#
# I understand, but it is a lot of work, starting from requesting to update dev tables, generating data and backscore it.
#
# I am also thinking that it would be a good idea to score every customer in prod, but to use outputs only for customers eligible for sc2.0. So in that case we will not need to have backscored scores at all.
#
# Biswa
# Oleh, for now just use the scorecard v2.2 as of January Employment date fix for backscoring
#
# I can generate those backscored tables by the EOW
#
# Hi Udhayanan Agasthiappan
# Please update
#
# prj-prod-dataplatform.worktable_data_analysis.tendo_scorecard_features_data_23-02-2026
#
# prj-prod-dataplatform.worktable_data_analysis.tendo_scorecard_loan_repayment_data_23-02-2026

# %%
dt_bs_oop_new["ee_customer_id"] = dt_bs_oop_new["ee_customer_id"].astype("str")
dt_bs_oop_ex["ee_customer_id"] = dt_bs_oop_ex["ee_customer_id"].astype("str")

# %%
dt_bs_oop_new.head(2)

# %%
dt_bs_oop_ex.head(2)

# %%
dt_bs_oop_ex["calc_date"] = pd.to_datetime(dt_bs_oop_ex["calc_date"], errors="coerce")
dt_bs_oop_ex["calc_date_correct"] = pd.to_datetime(
    dt_bs_oop_ex["calc_date"], errors="coerce"
) - pd.DateOffset(months=1)
dt_bs_oop_ex["calc_date_ym"] = (
    dt_bs_oop_ex["calc_date_correct"].dt.year * 100
    + dt_bs_oop_ex["calc_date_correct"].dt.month
)

# %%
dt_bs_oop_ex.head(2)

# %% [markdown]
# ## Gini

# %% [markdown]
# ### prod new

# %%
dt_prod_api = (
    dt_prod_api.merge(
        dt_raw[["ee_customer_id", "ee_onboarding_date"]].drop_duplicates(),
        how="left",
        on="ee_customer_id",
    )
    .merge(dt_oop_targets, how="left", on="ee_customer_id")
    .merge(
        dt_bs_oop_new[
            [
                "ee_customer_id",
                "score_oop",
                "osbal_as_of_resignation_date",
                "osbal_as_of_oop_eligible_date",
                "osbal_as_of_current_date",
            ]
        ],
        how="left",
        on="ee_customer_id",
    )
)

# %%
dt_prod_api["onb_rd_diff"] = (
    abs(
        pd.to_datetime(dt_prod_api["run_date"]).dt.normalize()
        - pd.to_datetime(dt_prod_api["ee_onboarding_date"]).dt.normalize()
    )
).dt.days
dt_prod_api["ee_onboarding_date"] = pd.to_datetime(
    dt_prod_api["ee_onboarding_date"]
).dt.normalize()
dt_prod_api["run_date"] = pd.to_datetime(dt_prod_api["run_date"]).dt.normalize()
dt_prod_api["onboarding_date_ym"] = (
    dt_prod_api["ee_onboarding_date"].dt.year * 100
    + dt_prod_api["ee_onboarding_date"].dt.month
)
dt_prod_api["run_date_ym"] = (
    dt_prod_api["run_date"].dt.year * 100 + dt_prod_api["run_date"].dt.month
)

# %%
dt_prod_api_calc = (
    dt_prod_api.dropna(subset=["ee_onboarding_date", "oop_target"])
    .sort_values(["onb_rd_diff", "run_date"])
    .drop_duplicates(subset=["ee_customer_id"], keep="first")
)

# %%
dt_prod_api_calc.head(1)

# %%
print("OOP New Users Prod")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_api_calc,
    date_column="ee_onboarding_date",
    score_column="oop_score_prod",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
)

# %%
print("OOP New Users Prod BS")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_api_calc[dt_prod_api_calc["score_oop"].notna()],
    date_column="ee_onboarding_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
)

# %%
print("OOP New Users Prod, weighted as-is")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_api_calc[dt_prod_api_calc["osbal_as_of_oop_eligible_date"].notna()],
    date_column="ee_onboarding_date",
    score_column="oop_score_prod",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date",
)

# %%
dt_prod_api_calc["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    dt_prod_api_calc["osbal_as_of_oop_eligible_date"]
)

# %%
dt_prod_api_calc.head(2)

# %%
print("OOP New Users Prod, weighted log")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    "Jun 2025 - Dec 2025": {"start": "2025-06-01", "end": "2025-12-31"},
}

calculate_gini_for_table(
    data=dt_prod_api_calc[
        dt_prod_api_calc["osbal_as_of_oop_eligible_date_log"].notna()
    ],
    date_column="ee_onboarding_date",
    score_column="oop_score_prod",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date_log",
)

# %%
print("OOP New Users Prod BS, weighted as-is")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_api_calc[
        (dt_prod_api_calc["score_oop"].notna())
        & dt_prod_api_calc["osbal_as_of_oop_eligible_date"].notna()
    ],
    date_column="ee_onboarding_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date",
)

# %%
print("OOP New Users Prod BS, weighted log transformed")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    "Jun 2025 - Dec 2025": {"start": "2025-06-01", "end": "2025-12-31"},
}

calculate_gini_for_table(
    data=dt_prod_api_calc[
        (dt_prod_api_calc["score_oop"].notna())
        & dt_prod_api_calc["osbal_as_of_oop_eligible_date_log"].notna()
    ],
    date_column="ee_onboarding_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date_log",
)

# %% [markdown]
# ### prod existing

# %%
dt_prod_batch.shape

# %%
dt_prod_batch["ee_customer_id"].nunique()

# %%
dt_prod_batch = dt_prod_batch.drop_duplicates(
    subset=["ee_customer_id", "run_date", "attrition_time_to_leave", "oop_score_prod"]
)

# %%
dt_prod_batch.shape

# %%
dt_prod_batch["run_date"] = pd.to_datetime(dt_prod_batch["run_date"], errors="coerce")
dt_prod_batch["run_date_ym"] = (
    dt_prod_batch["run_date"].dt.year * 100 + dt_prod_batch["run_date"].dt.month
)

# %%
dt_prod_batch.head(2)

# %%
dt_bs_oop_ex.tail(2)

# %%
dt_prod_batch = (
    dt_prod_batch.merge(
        dt_raw[["ee_customer_id", "ee_onboarding_date"]].drop_duplicates(),
        how="left",
        on="ee_customer_id",
    )
    .merge(dt_oop_targets, how="left", on="ee_customer_id")
    .merge(
        dt_bs_oop_new[
            [
                "ee_customer_id",
                "osbal_as_of_resignation_date",
                "osbal_as_of_oop_eligible_date",
                "osbal_as_of_current_date",
            ]
        ],
        how="left",
        on="ee_customer_id",
    )
    .merge(
        dt_bs_oop_ex[["ee_customer_id", "calc_date_ym", "score_oop"]],
        how="left",
        left_on=["ee_customer_id", "run_date_ym"],
        right_on=["ee_customer_id", "calc_date_ym"],
    )
)

# %%
dt_prod_batch.columns

# %%
dt_prod_batch.rename(
    columns={
        "ee_onboarding_date_x": "ee_onboarding_date",
    },
    inplace=True,
)

# %%
dt_prod_batch["ee_onboarding_date"] = pd.to_datetime(
    dt_prod_batch["ee_onboarding_date"]
).dt.normalize()
dt_prod_batch["onboarding_date_ym"] = (
    dt_prod_batch["ee_onboarding_date"].dt.year * 100
    + dt_prod_batch["ee_onboarding_date"].dt.month
)
dt_prod_batch["onb_rd_diff"] = (
    abs(dt_prod_batch["run_date"] - dt_prod_batch["ee_onboarding_date"])
).dt.days

# %%
dt_prod_batch_calc = dt_prod_batch.dropna(
    subset=["ee_onboarding_date", "oop_target"]
).sort_values(["ee_customer_id", "run_date"])

# %%
dt_prod_batch_calc.head(1)

# %%
print("OOP Existing Users Prod")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_batch_calc,
    date_column="run_date",
    score_column="oop_score_prod",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
)

# %%
print("OOP Existing Users Prod BS")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_batch_calc[dt_prod_batch_calc["score_oop"].notna()],
    date_column="run_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
)

# %%
print("OOP Existing Users Prod, weighted as-is")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_batch_calc[
        dt_prod_batch_calc["osbal_as_of_oop_eligible_date"].notna()
    ],
    date_column="run_date",
    score_column="oop_score_prod",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date",
)

# %%
dt_prod_batch_calc["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    dt_prod_batch_calc["osbal_as_of_oop_eligible_date"]
)

# %%
print("OOP Existing Users Prod, weighted log transformed")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    "Jun 2025 - Dec 2025": {"start": "2025-06-01", "end": "2025-12-31"},
}

calculate_gini_for_table(
    data=dt_prod_batch_calc[
        dt_prod_batch_calc["osbal_as_of_oop_eligible_date_log"].notna()
    ],
    date_column="run_date",
    score_column="oop_score_prod",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date_log",
)

# %%
print("OOP Existing Users Prod BS, weighted as-is")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_prod_batch_calc[
        (dt_prod_batch_calc["score_oop"].notna())
        & (dt_prod_batch_calc["osbal_as_of_oop_eligible_date"].notna())
    ],
    date_column="run_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date",
)

# %%
print("OOP Existing Users Prod BSm weighted log transformed")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    "Jun 2025 - Dec 2025": {"start": "2025-06-01", "end": "2025-12-31"},
}

calculate_gini_for_table(
    data=dt_prod_batch_calc[
        (dt_prod_batch_calc["score_oop"].notna())
        & (dt_prod_batch_calc["osbal_as_of_oop_eligible_date_log"].notna())
    ],
    date_column="run_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date_log",
)

# %% [markdown]
# ### bs new

# %%
dt_bs_oop_new.head(1)

# %%
dt_bs_oop_new.shape

# %%
dt_bs_oop_new["ee_customer_id"].nunique()

# %%
dt_bs_oop_new = dt_bs_oop_new.merge(
    dt_raw[["ee_customer_id", "ee_onboarding_date"]].drop_duplicates(),
    how="left",
    on="ee_customer_id",
).merge(dt_oop_targets, how="left", on="ee_customer_id")

# %%
dt_bs_oop_new["ee_onboarding_date"] = pd.to_datetime(
    dt_bs_oop_new["ee_onboarding_date"]
).dt.normalize()
dt_bs_oop_new["onboarding_date_ym"] = (
    dt_bs_oop_new["ee_onboarding_date"].dt.year * 100
    + dt_bs_oop_new["ee_onboarding_date"].dt.month
)

# %%
dt_bs_oop_new_calc = dt_bs_oop_new.dropna(subset=["oop_target"])

# %%
dt_bs_oop_new_calc.head(1)

# %%
print("OOP New Users BS")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_bs_oop_new_calc,
    date_column="ee_onboarding_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
)

# %%
dt_bs_oop_new_calc["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    dt_bs_oop_new_calc["osbal_as_of_oop_eligible_date"]
)

# %%
print("OOP Existing Users BS, weighted as-is")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_bs_oop_new_calc[
        dt_bs_oop_new_calc["osbal_as_of_oop_eligible_date"].notna()
    ],
    date_column="ee_onboarding_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date",
)

# %%
print("OOP Existing Users BS, weighted log")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    "Jun 2025 - Dec 2025": {"start": "2025-06-01", "end": "2025-12-31"},
}

calculate_gini_for_table(
    data=dt_bs_oop_new_calc[
        dt_bs_oop_new_calc["osbal_as_of_oop_eligible_date_log"].notna()
    ],
    date_column="ee_onboarding_date",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date_log",
)

# %% [markdown]
# ### bs existing

# %%
dt_bs_oop_ex.head(1)

# %%
dt_bs_oop_ex.shape

# %%
dt_bs_oop_ex["ee_customer_id"].nunique()

# %%
dt_bs_oop_ex = (
    dt_bs_oop_ex.merge(
        dt_raw[["ee_customer_id", "ee_onboarding_date"]].drop_duplicates(),
        how="left",
        on="ee_customer_id",
    )
    .merge(dt_oop_targets, how="left", on="ee_customer_id")
    .merge(
        dt_bs_oop_new[
            [
                "ee_customer_id",
                "osbal_as_of_resignation_date",
                "osbal_as_of_oop_eligible_date",
                "osbal_as_of_current_date",
            ]
        ],
        how="left",
        on="ee_customer_id",
    )
)

# %%
dt_bs_oop_ex.shape

# %%
dt_bs_oop_ex.head(1)

# %%
dt_bs_oop_ex_calc = dt_bs_oop_ex.dropna(subset=["oop_target"])

# %%
dt_bs_oop_ex_calc.head(1)

# %%
print("OOP Existing Users BS")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_bs_oop_ex_calc,
    date_column="calc_date_correct",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
)

# %%
dt_bs_oop_ex_calc["osbal_as_of_oop_eligible_date_log"] = np.log1p(
    dt_bs_oop_ex_calc["osbal_as_of_oop_eligible_date"]
)

# %%
print("OOP Existing Users BS, weighted as-is")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Jun 2025 - Sep 2025": {"start": "2025-06-01", "end": "2025-09-30"},
}

calculate_gini_for_table(
    data=dt_bs_oop_ex_calc[dt_bs_oop_ex_calc["osbal_as_of_oop_eligible_date"].notna()],
    date_column="calc_date_correct",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date",
)

# %%
print("OOP Existing Users BS, weighted log transform")
print("\n" + "=" * 50)

data_periods_dict = {
    "Jun 2025": {"start": "2025-06-01", "end": "2025-06-30"},
    "Jul 2025": {"start": "2025-07-01", "end": "2025-07-31"},
    "Aug 2025": {"start": "2025-08-01", "end": "2025-08-31"},
    "Sep 2025": {"start": "2025-09-01", "end": "2025-09-30"},
    "Oct 2025": {"start": "2025-10-01", "end": "2025-10-31"},
    "Nov 2025": {"start": "2025-11-01", "end": "2025-11-30"},
    "Dec 2025": {"start": "2025-12-01", "end": "2025-12-31"},
    "Jun 2025 - Dec 2025": {"start": "2025-06-01", "end": "2025-12-31"},
}

calculate_gini_for_table(
    data=dt_bs_oop_ex_calc[
        dt_bs_oop_ex_calc["osbal_as_of_oop_eligible_date_log"].notna()
    ],
    date_column="calc_date_correct",
    score_column="score_oop",
    target_column="oop_target",
    data_periods_dict=data_periods_dict,
    weights_col="osbal_as_of_oop_eligible_date_log",
)

# %% [markdown]
# ## Distribution metrics

# %% [markdown]
# ### prod new

# %%
# finding oop score ranges
# 1) observed min/max per segment from production
mm = (
    dt_prod_api.query('run_date >= "2025-10-15"')
    .assign(
        oop_risk_segment_prod=lambda d: d["oop_risk_segment_prod"].astype(str),
        oop_score_prod=lambda d: pd.to_numeric(d["oop_score_prod"], errors="coerce"),
    )
    .pivot_table(
        index="oop_risk_segment_prod", values="oop_score_prod", aggfunc=["min", "max"]
    )
)

# flatten columns: ('min','oop_score_prod') -> 'min', same for 'max'
mm = mm.droplevel(1, axis=1).rename(columns={"min": "min_score", "max": "max_score"})

# enforce segment order: A highest scores ... F lowest
order = list("ABCDEF")
mm = mm.reindex(order)

# clamp to [0,1] (safe even if scores are slightly outside)
mm["min_score"] = mm["min_score"].clip(0, 1)
mm["max_score"] = mm["max_score"].clip(0, 1)

# if any segments missing in prod sample, fill min/max by interpolation between neighbors
mm[["min_score", "max_score"]] = mm[["min_score", "max_score"]].interpolate(
    limit_direction="both"
)

# 2) compute cutpoints between adjacent segments:
# boundary between (A,B) = midpoint between min(A) and max(B), etc.
cut_idx = [f"{hi}/{lo}" for hi, lo in zip(order[:-1], order[1:])]
cuts = pd.Series(
    [
        (mm.loc[hi, "min_score"] + mm.loc[lo, "max_score"]) / 2
        for hi, lo in zip(order[:-1], order[1:])
    ],
    index=cut_idx,
)

# enforce monotonicity (A/B cutoff should be >= B/C cutoff >= ... >= E/F cutoff)
cuts = cuts.cummin()

# 3) build bins for pd.cut (ascending edges) + labels (F..A)
# edges: [0, cut_EF, cut_DE, ..., cut_AB, 1]
edges_new = [0.0] + cuts.iloc[::-1].tolist() + [np.nextafter(1.0, 2.0)]
labels_new = list("FEDCBA")

# Optional: a readable cutoff table
cutoff_table = pd.DataFrame(
    {
        "segment": labels_new,
        "min_inclusive": edges_new[:-1],
        "max_exclusive": edges_new[1:],
    }
)
cutoff_table.loc[cutoff_table["segment"] == "A", "max_exclusive"] = 1.0

# %%
dt_prod_api_calc["oop_risk_segment_bs"] = pd.cut(
    dt_prod_api_calc["score_oop"],
    bins=edges_new,
    labels=labels_new,
    right=False,  # intervals are [a, b) so boundary goes to the higher segment
    include_lowest=True,
)

# %%
# New prod users, prod score, June-Aug
df = dt_prod_api_calc.query(
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508"
).copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_prod",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt

# %%
# New prod users, BS score, June-Aug
df = dt_prod_api_calc.query(
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508"
).copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %% [markdown]
# ### bs new

# %%
dt_bs_oop_new_calc["oop_risk_segment_bs"] = pd.cut(
    dt_bs_oop_new_calc["score_oop"],
    bins=edges_new,
    labels=labels_new,
    right=False,  # intervals are [a, b) so boundary goes to the higher segment
    include_lowest=True,
)

# %%
# New BS users, BS score, June-Aug
df = dt_bs_oop_new_calc.query(
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508"
).copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %% [markdown]
# ### prod existing

# %%
# finding oop score ranges
# 1) observed min/max per segment from production
mm = (
    dt_prod_batch.query('run_date >= "2025-10-15"')
    .assign(
        oop_risk_segment_prod=lambda d: d["oop_risk_segment_prod"].astype(str),
        oop_score_prod=lambda d: pd.to_numeric(d["oop_score_prod"], errors="coerce"),
    )
    .pivot_table(
        index="oop_risk_segment_prod", values="oop_score_prod", aggfunc=["min", "max"]
    )
)

# flatten columns: ('min','oop_score_prod') -> 'min', same for 'max'
mm = mm.droplevel(1, axis=1).rename(columns={"min": "min_score", "max": "max_score"})

# enforce segment order: A highest scores ... F lowest
order = list("ABCDEF")
mm = mm.reindex(order)

# clamp to [0,1] (safe even if scores are slightly outside)
mm["min_score"] = mm["min_score"].clip(0, 1)
mm["max_score"] = mm["max_score"].clip(0, 1)

# if any segments missing in prod sample, fill min/max by interpolation between neighbors
mm[["min_score", "max_score"]] = mm[["min_score", "max_score"]].interpolate(
    limit_direction="both"
)

# 2) compute cutpoints between adjacent segments:
# boundary between (A,B) = midpoint between min(A) and max(B), etc.
cut_idx = [f"{hi}/{lo}" for hi, lo in zip(order[:-1], order[1:])]
cuts = pd.Series(
    [
        (mm.loc[hi, "min_score"] + mm.loc[lo, "max_score"]) / 2
        for hi, lo in zip(order[:-1], order[1:])
    ],
    index=cut_idx,
)

# enforce monotonicity (A/B cutoff should be >= B/C cutoff >= ... >= E/F cutoff)
cuts = cuts.cummin()

# 3) build bins for pd.cut (ascending edges) + labels (F..A)
# edges: [0, cut_EF, cut_DE, ..., cut_AB, 1]
edges_ex = [0.0] + cuts.iloc[::-1].tolist() + [np.nextafter(1.0, 2.0)]
labels_ex = list("FEDCBA")

# Optional: a readable cutoff table
cutoff_table = pd.DataFrame(
    {
        "segment": labels_ex,
        "min_inclusive": edges_ex[:-1],
        "max_exclusive": edges_ex[1:],
    }
)
cutoff_table.loc[cutoff_table["segment"] == "A", "max_exclusive"] = 1.0

# %%
dt_prod_batch_calc["oop_risk_segment_bs"] = pd.cut(
    dt_prod_batch_calc["score_oop"],
    bins=edges_ex,
    labels=labels_ex,
    right=False,  # intervals are [a, b) so boundary goes to the higher segment
    include_lowest=True,
)

# %%
# Existing prod users, prod score, June
df = dt_prod_batch_calc.query("run_date_ym == 202506").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_prod",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=True)

# %%
# Existing prod users, prod score, July
df = dt_prod_batch_calc.query("run_date_ym == 202507").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_prod",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=True)

# %%
# Existing prod users, prod score, Aug
df = dt_prod_batch_calc.query("run_date_ym == 202508").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_prod",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=True)

# %%
# Existing prod users, BS score, June
df = dt_prod_batch_calc.query("run_date_ym == 202506").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %%
# Existing prod users, BS score, July
df = dt_prod_batch_calc.query("run_date_ym == 202507").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %%
# Existing prod users, BS score, Aug
df = dt_prod_batch_calc.query("run_date_ym == 202508").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %% [markdown]
# ### bs existing

# %%
dt_bs_oop_ex_calc["oop_risk_segment_bs"] = pd.cut(
    dt_bs_oop_ex_calc["score_oop"],
    bins=edges_ex,
    labels=labels_ex,
    right=False,  # intervals are [a, b) so boundary goes to the higher segment
    include_lowest=True,
)

# %%
# Existing BS users, BS score, June
df = dt_bs_oop_ex_calc.query("calc_date_ym == 202506").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %%
# Existing BS users, BS score, July
df = dt_bs_oop_ex_calc.query("calc_date_ym == 202507").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %%
# Existing BS users, BS score, Aug
df = dt_bs_oop_ex_calc.query("calc_date_ym == 202508").copy()

df["oop_target_bad"] = 1 - df["oop_target"]

df["osbal_current_bad"] = df["oop_target_bad"] * df["osbal_as_of_current_date"]

pt = pd.pivot_table(
    df,
    index="oop_risk_segment_bs",
    values=["oop_target_bad", "osbal_as_of_resignation_date", "osbal_current_bad"],
    aggfunc={
        "oop_target_bad": ["count", "mean"],
        "osbal_as_of_resignation_date": "sum",
        "osbal_current_bad": "sum",
    },
)

pt[("php_weighted_outstanding_bad_rate", "ratio")] = (
    pt[("osbal_current_bad", "sum")] / pt[("osbal_as_of_resignation_date", "sum")]
)

pt.sort_index(ascending=False)

# %% [markdown]
# # Attrition

# %%
# BS Attrition
sql_query_attrition = """
SELECT *
FROM `prj-prod-dataplatform.risk_mart.tendo_backscored_jan24_jan26_20260201_attrition`
"""

dt_bs_attr = client.query(sql_query_attrition).to_dataframe()

# %%
dt_bs_attr["ee_customer_id"] = dt_bs_attr["ee_customer_id"].astype("str")

dt_bs_attr["calc_date"] = pd.to_datetime(dt_bs_attr["calc_date"], errors="coerce")
dt_bs_attr["calc_date_correct"] = pd.to_datetime(
    dt_bs_attr["calc_date"], errors="coerce"
) - pd.DateOffset(months=1)
dt_bs_attr["calc_date_ym"] = (
    dt_bs_attr["calc_date_correct"].dt.year * 100
    + dt_bs_attr["calc_date_correct"].dt.month
)

# %%
new_1m = dt_bs_attr["ee_onboarding_month"] == dt_bs_attr["calc_date_ym"]
dt_bs_attr["is_new_customer_flag_1m"] = new_1m.astype("int")

# %%
dt_bs_attr.head(2)

# %%
dt_res = pd.read_pickle(
    generate_bucket_url("Oleh/tendo/data/resignation_data_14012026.pkl", GS_BUCKET)
)

# %%
dt_res.head(1)

# %% [markdown]
# ## Distribution metrics

# %% [markdown]
# ### prod new

# %%
dt_prod_api.shape

# %%
dt_prod_api.head()

# %%
dt_prod_api_attr = dt_prod_api.merge(dt_res, how="left", on="ee_customer_id").merge(
    dt_bs_attr[
        [
            "ee_customer_id",
            "calc_date_ym",
            "score_attr",
            "score_attr_segment",
            "is_new_customer_flag_1m",
        ]
    ],
    how="left",
    left_on=["ee_customer_id", "run_date_ym"],
    right_on=["ee_customer_id", "calc_date_ym"],
)

# %%
dt_prod_api_calc = (
    dt_prod_api_attr.dropna(subset=["ee_onboarding_date"])
    .sort_values(["onb_rd_diff", "run_date"])
    .drop_duplicates(subset=["ee_customer_id"], keep="first")
)

# %%
months_diff = (
    dt_prod_api_calc["ee_resignation_date_correct"].dt.year
    - dt_prod_api_calc["run_date"].dt.year
) * 12 + (
    dt_prod_api_calc["ee_resignation_date_correct"].dt.month
    - dt_prod_api_calc["run_date"].dt.month
)

dt_prod_api_calc["time_to_attrition"] = np.where(
    dt_prod_api_calc["ee_resignation_date_correct"].isna(), np.nan, months_diff
)
dt_prod_api_calc["attrition_event"] = (
    dt_prod_api_calc["ee_resignation_date_correct"].notna().astype("int")
)

# %%
dt_prod_api_calc["score_attr_corrected"] = dt_prod_api_calc["score_attr"].replace(
    np.inf, 15
)
dt_prod_api_calc["attrition_score_prod"] = (
    dt_prod_api_calc["attrition_time_to_leave"].replace("12+", "15").astype("float")
)

mapping_dict = {
    1: "Very high",
    2: "Very high",
    3: "Very high",
    4: "High",
    5: "High",
    6: "High",
    7: "Average",
    8: "Average",
    9: "Average",
    10: "Low",
    11: "Low",
    12: "Low",
    15: "Very low",
}

dt_prod_api_calc["attrition_risk_segment_prod"] = dt_prod_api_calc[
    "attrition_score_prod"
].replace(mapping_dict)

# %%
# New prod users, prod score, June-Aug
df = dt_prod_api_calc.query(
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508"
).copy()

pt = pd.pivot_table(
    df,
    index="attrition_risk_segment_prod",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "attrition_score_prod",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "attrition_score_prod": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "attrition_score_prod", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# New prod users, BS score, June-Aug
df = dt_prod_api_calc.query(
    "onboarding_date_ym >= 202506 & onboarding_date_ym <= 202508"
).copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %% [markdown]
# ### bs new

# %%
dt_bs_attr.head()

# %%
dt_bs_attr_dev = dt_bs_attr.merge(dt_res, how="left", on="ee_customer_id")

# %%
dt_bs_attr_dev_calc = dt_bs_attr_dev.query("is_new_customer_flag_1m == 1").copy()

# %%
months_diff = (
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].dt.year
    - dt_bs_attr_dev_calc["calc_date_correct"].dt.year
) * 12 + (
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].dt.month
    - dt_bs_attr_dev_calc["calc_date_correct"].dt.month
)

dt_bs_attr_dev_calc["time_to_attrition"] = np.where(
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].isna(), np.nan, months_diff
)
dt_bs_attr_dev_calc["attrition_event"] = (
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].notna().astype("int")
)

# %%
dt_bs_attr_dev_calc["score_attr_corrected"] = dt_bs_attr_dev_calc["score_attr"].replace(
    np.inf, 15
)

# %%
# New BS users, BS score, June-Aug
df = dt_bs_attr_dev_calc.query(
    "ee_onboarding_month >= 202506 & ee_onboarding_month <= 202508"
).copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %% [markdown]
# ### prod existing

# %%
dt_prod_batch.shape

# %%
dt_prod_batch.head()

# %%
dt_prod_batch_attr = dt_prod_batch.merge(dt_res, how="left", on="ee_customer_id").merge(
    dt_bs_attr[
        [
            "ee_customer_id",
            "calc_date_ym",
            "score_attr",
            "score_attr_segment",
            "is_new_customer_flag_1m",
        ]
    ],
    how="left",
    left_on=["ee_customer_id", "run_date_ym"],
    right_on=["ee_customer_id", "calc_date_ym"],
)

# %%
dt_prod_batch_attr.shape

# %%
dt_prod_batch_calc = dt_prod_batch_attr.dropna(subset=["ee_onboarding_date"])

# %%
months_diff = (
    dt_prod_batch_calc["ee_resignation_date_correct"].dt.year
    - dt_prod_batch_calc["run_date"].dt.year
) * 12 + (
    dt_prod_batch_calc["ee_resignation_date_correct"].dt.month
    - dt_prod_batch_calc["run_date"].dt.month
)

dt_prod_batch_calc["time_to_attrition"] = np.where(
    dt_prod_batch_calc["ee_resignation_date_correct"].isna(), np.nan, months_diff
)
dt_prod_batch_calc["attrition_event"] = (
    dt_prod_batch_calc["ee_resignation_date_correct"].notna().astype("int")
)

# %%
dt_prod_batch_calc["score_attr_corrected"] = dt_prod_batch_calc["score_attr"].replace(
    np.inf, 15
)
dt_prod_batch_calc["attrition_score_prod"] = (
    dt_prod_batch_calc["attrition_time_to_leave"].replace("12+", "15").astype("float")
)

mapping_dict = {
    1: "Very high",
    2: "Very high",
    3: "Very high",
    4: "High",
    5: "High",
    6: "High",
    7: "Average",
    8: "Average",
    9: "Average",
    10: "Low",
    11: "Low",
    12: "Low",
    15: "Very low",
}

dt_prod_batch_calc["attrition_risk_segment_prod"] = dt_prod_batch_calc[
    "attrition_score_prod"
].replace(mapping_dict)

# %%
# Existing prod users, prod score, June
df = dt_prod_batch_calc.query("run_date_ym == 202506").copy()

pt = pd.pivot_table(
    df,
    index="attrition_risk_segment_prod",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "attrition_score_prod",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "attrition_score_prod": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "attrition_score_prod", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# Existing prod users, prod score, July
df = dt_prod_batch_calc.query("run_date_ym == 202507").copy()

pt = pd.pivot_table(
    df,
    index="attrition_risk_segment_prod",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "attrition_score_prod",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "attrition_score_prod": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "attrition_score_prod", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# Existing prod users, prod score, August
df = dt_prod_batch_calc.query("run_date_ym == 202508").copy()

pt = pd.pivot_table(
    df,
    index="attrition_risk_segment_prod",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "attrition_score_prod",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "attrition_score_prod": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "attrition_score_prod", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# Existing prod users, BS score, June
df = dt_prod_batch_calc.query("run_date_ym == 202506").copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# Existing prod users, BS score, July
df = dt_prod_batch_calc.query("run_date_ym == 202507").copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# Existing prod users, BS score, August
df = dt_prod_batch_calc.query("run_date_ym == 202508").copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",  # score_attr_segment - BS
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %% [markdown]
# ### bs existing

# %%
dt_bs_attr_dev_calc.head()

# %%
dt_bs_attr_dev_calc.shape

# %%
dt_bs_attr_dev_calc = dt_bs_attr_dev.query("is_new_customer_flag_3m == 0").copy()

# %%
months_diff = (
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].dt.year
    - dt_bs_attr_dev_calc["calc_date_correct"].dt.year
) * 12 + (
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].dt.month
    - dt_bs_attr_dev_calc["calc_date_correct"].dt.month
)

dt_bs_attr_dev_calc["time_to_attrition"] = np.where(
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].isna(), np.nan, months_diff
)
dt_bs_attr_dev_calc["attrition_event"] = (
    dt_bs_attr_dev_calc["ee_resignation_date_correct"].notna().astype("int")
)

# %%
dt_bs_attr_dev_calc["score_attr_corrected"] = dt_bs_attr_dev_calc["score_attr"].replace(
    np.inf, 15
)

# %%
# Existing BS users, BS score, June
df = dt_bs_attr_dev_calc.query("calc_date_ym == 202506").copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# Existing BS users, BS score, July
df = dt_bs_attr_dev_calc.query("calc_date_ym == 202507").copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%
# Existing BS users, BS score, August
df = dt_bs_attr_dev_calc.query("calc_date_ym == 202508").copy()

pt = pd.pivot_table(
    df,
    index="score_attr_segment",
    values=[
        "attrition_event",
        "score_attr_corrected",
        "time_to_attrition",
        "ee_customer_id",
    ],
    aggfunc={
        "attrition_event": "mean",
        "score_attr_corrected": "mean",
        "time_to_attrition": "mean",
        "ee_customer_id": "count",
    },
).rename(columns={"ee_customer_id": "count"})[
    ["attrition_event", "score_attr_corrected", "time_to_attrition", "count"]
]

order = ["Very low", "Low", "Average", "High", "Very high"]

pt.loc[order]

# %%


# %% [markdown]
#

# %%
