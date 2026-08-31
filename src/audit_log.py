"""
audit_log.py
Writes one JSON line per dispute decision, capturing the full trail:
input features -> classifier score -> agent reasoning -> final action.

This is the literal, checkable artifact for "every money action
explainable, bounded and gated" — anyone can open logs/audit_trail.jsonl
and reconstruct exactly why each decision was made.
"""

import json
import os
from datetime import datetime, timezone


LOG_PATH = "logs/audit_trail.jsonl"


def log_decision(trans_num: str, features: dict, fraud_probability: float,
                  decision: dict, model_version: str = "rf_v1"):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    packet = decision.get("evidence_packet") or {}

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trans_num": trans_num,
        "model_version": model_version,
        "fraud_probability": round(float(fraud_probability), 4),
        # Surfaced at the top level (not just buried in evidence_packet)
        # so a reviewer scanning the log can see the full basis for the
        # decision without expanding nested structures.
        "evidence_completeness": packet.get("completeness_score"),
        "merchant_evidence_strength": packet.get("strength_score"),
        "critical_evidence_missing": packet.get("critical_missing", []),
        "input_features": features,
        "action": decision["action"],
        "reasoning": decision["reasoning"],
        "guardrail_triggered": decision.get("guardrail_triggered"),
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_audit_trail(limit: int = None) -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[-limit:] if limit else lines