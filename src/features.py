"""
features.py
Feature engineering for the Chargeback Evidence Responder.

Takes raw Sparkov-simulated transaction rows and produces:
  1. Real, model-ready numeric/categorical features derived from actual data.
  2. Synthetic dispute-layer fields (clearly flagged) that simulate
     operational metadata a real payments platform would have but this
     public dataset doesn't (delivery status, device/IP match, dispute reason).

Every synthetic column is prefixed `synth_` so it's unmistakable in the
dataframe which fields are real vs. simulated. See data/synthetic_fields.md
for the full documentation of how each is generated.
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ---------------------------------------------------------------------------
# REAL FEATURES (derived entirely from actual dataset columns)
# ---------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between customer home and merchant location."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_real_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse timestamp
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # 0=Mon
    df["is_night"] = df["hour"].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)

    # Customer age at time of transaction
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # Distance between customer home and merchant (km) — strong real fraud signal
    df["distance_km"] = haversine_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # Amount z-score within its category (is this amount unusual for this
    # type of purchase, relative to all transactions of that category?)
    cat_mean = df.groupby("category")["amt"].transform("mean")
    cat_std = df.groupby("category")["amt"].transform("std").replace(0, np.nan)
    df["amt_zscore_in_category"] = ((df["amt"] - cat_mean) / cat_std).fillna(0)

    # City population bucket (proxy for urban vs rural — fraud patterns differ)
    df["city_pop_log"] = np.log1p(df["city_pop"])

    return df


# ---------------------------------------------------------------------------
# SYNTHETIC DISPUTE-LAYER FEATURES (clearly flagged as simulated)
# ---------------------------------------------------------------------------
#
# These fields do NOT exist in the source dataset. They simulate the kind of
# operational metadata a real chargeback system would have (delivery
# confirmation, device/IP fingerprint match, dispute reason code). They are
# generated with a controlled correlation to `is_fraud` so the resulting
# "disputes" are realistic (fraud correlates with delivery/device mismatches)
# without being deterministic giveaways (i.e. the classifier still has to
# work — these aren't just is_fraud in disguise).
# ---------------------------------------------------------------------------

DISPUTE_REASON_CODES = [
    "product_not_received",
    "product_not_as_described",
    "unauthorized_transaction",
    "duplicate_charge",
    "credit_not_processed",
]


def add_synthetic_dispute_fields(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    n = len(df)
    fraud = df["is_fraud"].values

    # Delivery confirmed: legitimate transactions usually have delivery
    # confirmation; fraudulent ones often don't (goods intercepted / never
    # ordered by the real cardholder).
    p_delivered = np.where(fraud == 1, 0.25, 0.92)
    df["synth_delivery_confirmed"] = rng.binomial(1, p_delivered).astype(object)

    # Device/IP match to customer's known devices: fraud far more likely to
    # come from an unrecognized device/IP.
    p_device_match = np.where(fraud == 1, 0.15, 0.90)
    df["synth_device_ip_match"] = rng.binomial(1, p_device_match).astype(bool)

    # Prior dispute count for this customer (fraud victims / friendly-fraud
    # repeat offenders skew slightly higher; kept noisy on purpose).
    base_rate = np.where(fraud == 1, 1.2, 0.3)
    df["synth_prior_dispute_count"] = rng.poisson(base_rate)

    # Dispute reason code: unauthorized_transaction is disproportionately
    # tied to real fraud; other reasons are spread more evenly.
    reason = np.empty(n, dtype=object)
    for i in range(n):
        if fraud[i] == 1:
            probs = [0.15, 0.10, 0.60, 0.10, 0.05]
        else:
            probs = [0.30, 0.25, 0.10, 0.20, 0.15]
        reason[i] = rng.choice(DISPUTE_REASON_CODES, p=probs)
    df["synth_dispute_reason"] = reason

    # Time-to-dispute in days (how long after the transaction the dispute
    # was filed). Fraud tends to surface faster once the cardholder notices.
    df["synth_days_to_dispute"] = np.where(
        fraud == 1,
        rng.gamma(shape=2.0, scale=2.0, size=n),
        rng.gamma(shape=3.0, scale=5.0, size=n),
    ).round(1)

    # Missing-data failure case injection: randomly null out delivery status
    # on a small % of rows to exercise the "escalate to human" guardrail.
    missing_mask = rng.random(n) < 0.03
    df.loc[missing_mask, "synth_delivery_confirmed"] = None

    return df


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "amt",
    "hour",
    "day_of_week",
    "is_night",
    "age",
    "distance_km",
    "amt_zscore_in_category",
    "city_pop_log",
    "synth_device_ip_match",
    "synth_prior_dispute_count",
    "synth_days_to_dispute",
]

# Subset containing ONLY features derived from real dataset columns —
# used for the "real-features-only" comparison to prove the model isn't
# just leaning on synthetic dispute metadata.
REAL_ONLY_FEATURE_COLUMNS = [
    "amt",
    "hour",
    "day_of_week",
    "is_night",
    "age",
    "distance_km",
    "amt_zscore_in_category",
    "city_pop_log",
]


def build_feature_table(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)
    df = add_real_features(df)
    df = add_synthetic_dispute_fields(df)
    return df


if __name__ == "__main__":
    out = build_feature_table("data/raw/fraudTrain_150k.csv")
    print(out[["trans_num", "is_fraud"] + FEATURE_COLUMNS + [
        "synth_delivery_confirmed", "synth_dispute_reason"
    ]].head(10).to_string())
    print("\nShape:", out.shape)
    out.to_csv("data/processed/features_sample.csv", index=False)
    print("Saved to data/processed/features_sample.csv")
