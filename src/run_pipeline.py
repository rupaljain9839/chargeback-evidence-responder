"""
run_pipeline.py
End-to-end demo runner: takes a sample of real (feature-engineered)
disputes, scores them with the trained classifier, runs each through
the Evidence Agent (guardrails + reasoning), and writes every decision
to the audit trail.

This is the script the Streamlit demo app will call under the hood.
"""

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, build_feature_table
from evidence_agent import get_evidence_decision
from audit_log import log_decision


def run_on_sample(csv_path: str, n: int = 20, use_llm: bool = None):
    print("Loading model and scaler...")
    model = joblib.load("models/classifier.pkl")

    print("Building features...")
    df = build_feature_table(csv_path)

    sample = pd.concat([
        df[df["is_fraud"] == 1].sample(n=min(n // 2, (df["is_fraud"] == 1).sum()), random_state=1),
        df[df["is_fraud"] == 0].sample(n=n // 2, random_state=1),
    ]).sample(frac=1, random_state=2)  # shuffle

    X = sample[FEATURE_COLUMNS]
    probs = model.predict_proba(X)[:, 1]

    results = []
    for (_, row), prob in zip(sample.iterrows(), probs):
        record = row[FEATURE_COLUMNS].to_dict()
        record["fraud_probability"] = float(prob)
        record["category"] = row["category"]
        record["trans_num"] = row["trans_num"]
        # Fields not used by the classifier but needed by the evidence
        # agent / guardrails (e.g. delivery status, dispute reason)
        record["synth_delivery_confirmed"] = row["synth_delivery_confirmed"]
        record["synth_dispute_reason"] = row["synth_dispute_reason"]

        decision = get_evidence_decision(record, use_llm=use_llm)
        entry = log_decision(row["trans_num"], record, prob, decision)

        results.append({
            "trans_num": row["trans_num"],
            "actual_fraud": int(row["is_fraud"]),
            "fraud_probability": round(float(prob), 3),
            "action": decision["action"],
            "guardrail_triggered": decision["guardrail_triggered"],
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = run_on_sample("data/raw/fraudTrain_150k.csv", n=20)
    print("\n" + "=" * 90)
    print("PIPELINE RESULTS")
    print("=" * 90)
    print(results.to_string(index=False))

    print(f"\nAction distribution:\n{results['action'].value_counts()}")
    print(f"\nAudit trail written to logs/audit_trail.jsonl")
