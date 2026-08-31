# Data Provenance: Real vs. Synthetic Fields

This document exists so anyone reviewing this project can see exactly which
fields came from real transaction data and which were generated to simulate
dispute-operations metadata that public fraud datasets don't include.

## Source dataset

**Sparkov Credit Card Transactions Fraud Detection** (Kaggle:
`kartik2112/fraud-detection`) — simulated-but-realistic transaction data
generated with the Sparkov simulator, including real geolocation, merchant,
category, and demographic fields, with a genuine fraud label (`is_fraud`)
at a realistic ~0.5% positive rate.

## REAL fields (used as-is or directly derived from the dataset)

| Field | Source |
|---|---|
| `amt`, `category`, `merchant` | Original dataset |
| `trans_date_trans_time`, `hour`, `day_of_week`, `is_night` | Derived from original timestamp |
| `age` | Derived from original `dob` |
| `distance_km` | Derived from original customer lat/long vs merchant lat/long |
| `amt_zscore_in_category` | Derived: how unusual this amount is vs. all transactions in its category |
| `city_pop_log` | Derived from original `city_pop` |
| `is_fraud` | Original ground-truth label |

None of these required simulation — they're real signal from real
(simulated-transaction-but-realistic) data.

## SYNTHETIC fields (prefixed `synth_`, generated — NOT in the source data)

The source dataset has no dispute-operations metadata (no delivery status,
no device fingerprinting, no dispute reason codes) because it wasn't built
for that purpose. Real chargeback systems *do* have this metadata. We
generate it to make the Evidence Agent's output realistic, with each field
probabilistically correlated to the real `is_fraud` label (not 1:1 — noise
is deliberately included so the classifier still has to do real work):

| Field | How it's generated | Correlation logic |
|---|---|---|
| `synth_delivery_confirmed` | Bernoulli draw | P(delivered\|fraud)=0.25, P(delivered\|legit)=0.92 |
| `synth_device_ip_match` | Bernoulli draw | P(match\|fraud)=0.15, P(match\|legit)=0.90 |
| `synth_prior_dispute_count` | Poisson draw | λ=1.2 for fraud, λ=0.3 for legit |
| `synth_dispute_reason` | Categorical draw | fraud skews toward `unauthorized_transaction`; legit spread across all 5 reason codes |
| `synth_days_to_dispute` | Gamma draw | fraud disputes filed faster (shape=2, scale=2) than legit (shape=3, scale=5) |

`synth_delivery_confirmed` is also randomly nulled on ~3% of rows to
simulate missing operational data — this is what exercises the
"escalate to human" guardrail described in `src/guardrails.py`.

## Why this matters for evaluation

Because the synthetic fields are generated *with knowledge of* `is_fraud`,
they cannot be used to claim the classifier "discovered" fraud from
scratch — they are deliberately informative, the same way real dispute
metadata would be. The **real** engineered features (distance, amount
z-score, night-time flag) are the fields that demonstrate the model
learning genuine fraud patterns from data it wasn't handed the answer key
for. Both are reported separately in `evaluation/metrics_report.json` via
a feature-importance breakdown so reviewers can see the split.
