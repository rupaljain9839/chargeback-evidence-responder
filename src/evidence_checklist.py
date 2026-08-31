"""
evidence_checklist.py
Maps each dispute reason to the evidence a reviewer would actually need to
decide it, mirroring how a real chargeback/representment workflow works
(different dispute reasons require different supporting evidence — proof
of delivery, device/IP match, transaction history, etc).

This turns the agent from "one LLM call over some fields" into an actual
multi-step workflow:
    dispute reason -> required evidence -> what's present -> completeness
    /strength scores -> critical-gap check -> (only then) a bounded
    recommendation.

Evidence items are split into CRITICAL and non-critical per dispute
reason. Missing a CRITICAL item forces ESCALATE regardless of overall
completeness — missing a non-critical item (e.g. delivery confirmation
on an unauthorized-transaction dispute) lowers the completeness score
but does not by itself force escalation. This mirrors how a real
reviewer would behave: some evidence is a hard requirement for a given
dispute reason, some is merely helpful context.
"""

# Every evidence item this system can check, per record field.
# `check` returns True if the field is present/usable (not missing).
# `signal` returns a float in [-1, 1] once present: positive means the
# evidence points toward the transaction being LEGITIMATE (favors
# CONTEST/merchant), negative means it points toward FRAUD (favors
# ACCEPT/refund/cardholder). This is what turns "we have the field"
# into "the field says something, and whose side it supports."
EVIDENCE_FIELD_CHECKS = {
    "device_ip_match": {
        "label": "Device/IP match on file",
        "check": lambda r: r.get("synth_device_ip_match") is not None,
        "describe": lambda r: f"Device/IP match: {r.get('synth_device_ip_match')}",
        "signal": lambda r: 1.0 if r.get("synth_device_ip_match") else -1.0,
    },
    "delivery_confirmation": {
        "label": "Delivery confirmation",
        "check": lambda r: (
            r.get("synth_delivery_confirmed") is not None
            and not _is_nan(r.get("synth_delivery_confirmed"))
        ),
        "describe": lambda r: f"Delivery confirmed: {r.get('synth_delivery_confirmed')}",
        "signal": lambda r: 1.0 if r.get("synth_delivery_confirmed") else -1.0,
    },
    "location_consistency": {
        "label": "Home-to-merchant distance",
        "check": lambda r: r.get("distance_km") is not None,
        "describe": lambda r: f"Distance from home: {r.get('distance_km'):.1f} km",
        # Beyond ~150km treated as increasingly inconsistent with normal
        # cardholder behavior; scaled and clamped to [-1, 1].
        "signal": lambda r: max(-1.0, min(1.0, 1.0 - (r.get("distance_km", 0) / 150.0))),
    },
    "spend_pattern_consistency": {
        "label": "Spend-pattern consistency (amount vs. category norm)",
        "check": lambda r: r.get("amt_zscore_in_category") is not None,
        "describe": lambda r: f"Amount z-score for category: {r.get('amt_zscore_in_category'):.2f}",
        # Large positive z-score (unusually high spend) reads as
        # fraud-favoring; near-zero or negative reads as legitimate.
        "signal": lambda r: max(-1.0, min(1.0, 1.0 - (abs(r.get("amt_zscore_in_category", 0)) / 3.0))),
    },
    "transaction_history": {
        "label": "Customer prior-dispute history",
        "check": lambda r: r.get("synth_prior_dispute_count") is not None,
        "describe": lambda r: f"Prior disputes by this customer: {r.get('synth_prior_dispute_count')}",
        # More prior disputes is mildly fraud-favoring (repeat pattern).
        "signal": lambda r: max(-1.0, min(1.0, 1.0 - (r.get("synth_prior_dispute_count", 0) * 0.6))),
    },
    "dispute_timing": {
        "label": "Time-to-dispute pattern",
        "check": lambda r: r.get("synth_days_to_dispute") is not None,
        "describe": lambda r: f"Days between transaction and dispute filing: {r.get('synth_days_to_dispute')}",
        # Very fast disputes (cardholder notices immediately) skew
        # fraud-favoring; slow disputes skew legitimate/friendly-fraud.
        "signal": lambda r: max(-1.0, min(1.0, (r.get("synth_days_to_dispute", 10) - 5.0) / 10.0)),
    },
}


def _is_nan(x):
    try:
        return x != x  # NaN != NaN is True; works without importing numpy
    except Exception:
        return False


# Which evidence a reviewer needs per dispute reason, split into
# critical (a hard requirement — missing it forces ESCALATE) and
# supporting (improves completeness/strength but a gap here alone does
# not force escalation). Mirrors real reviewer judgment: an
# unauthorized-transaction claim can't be safely auto-decided without
# device/IP and account-history signals, but a missing delivery record
# on that same dispute type is a lesser gap.
EVIDENCE_REQUIREMENTS = {
    "unauthorized_transaction": {
        "critical": ["device_ip_match", "location_consistency", "transaction_history"],
        "supporting": ["delivery_confirmation", "dispute_timing"],
    },
    "product_not_received": {
        "critical": ["delivery_confirmation"],
        "supporting": ["dispute_timing"],
    },
    "product_not_as_described": {
        "critical": ["delivery_confirmation"],
        "supporting": ["spend_pattern_consistency"],
    },
    "duplicate_charge": {
        "critical": ["transaction_history"],
        "supporting": ["dispute_timing"],
    },
    "credit_not_processed": {
        "critical": ["transaction_history"],
        "supporting": ["dispute_timing"],
    },
}


def _supports_label(present: bool, signal) -> str:
    """Direction label for one evidence item, reusing its numeric signal
    rather than introducing a second, separate judgment."""
    if not present:
        return "missing"
    if signal is None:
        return "neutral"
    if signal >= 0.15:
        return "merchant"
    if signal <= -0.15:
        return "cardholder"
    return "neutral"


def assess_evidence(record: dict) -> dict:
    """
    Runs the evidence-checklist workflow for one dispute record.

    Returns a structured packet: required items (critical + supporting),
    what's present, per-item direction (supports merchant / cardholder /
    missing / neutral), completeness score, merchant-strength score, and
    whether any CRITICAL item is missing (`has_critical_gap`) — this last
    flag, not the completeness percentage, is what guardrails.py uses to
    decide whether to escalate.

    Completeness and strength answer two different questions:
      - Completeness: do we have the evidence a reviewer would need
        (critical + supporting combined)?
      - Merchant evidence strength: given what we HAVE, how strongly
        does it support the MERCHANT'S side (contesting, keeping the
        money)? High = favors CONTEST. Low = favors ACCEPT (refund).
    A case can be 100% complete and still weakly-signaled either way,
    or thin but decisive — the two scores are reported separately so
    neither is mistaken for the other.
    """
    reason = record.get("synth_dispute_reason", "unauthorized_transaction")
    reqs = EVIDENCE_REQUIREMENTS.get(reason, EVIDENCE_REQUIREMENTS["unauthorized_transaction"])
    critical_keys = reqs["critical"]
    supporting_keys = reqs["supporting"]
    all_keys = critical_keys + supporting_keys

    items = []
    present_count = 0
    missing = []
    critical_missing = []
    signal_values = []

    for key in all_keys:
        spec = EVIDENCE_FIELD_CHECKS[key]
        is_critical = key in critical_keys
        present = spec["check"](record)
        signal = spec["signal"](record) if present else None
        supports = _supports_label(present, signal)

        items.append({
            "key": key,
            "label": spec["label"],
            "present": present,
            "critical": is_critical,
            "detail": spec["describe"](record) if present else "Not available",
            "signal": signal,  # -1 (fraud-favoring) .. +1 (legitimate-favoring), None if missing
            "supports": supports,  # "merchant" | "cardholder" | "neutral" | "missing"
        })

        if present:
            present_count += 1
            signal_values.append(signal)
        else:
            missing.append(spec["label"])
            if is_critical:
                critical_missing.append(spec["label"])

    completeness = round(100.0 * present_count / len(all_keys), 1) if all_keys else 100.0

    # Strength: average signal across present items, rescaled from
    # [-1, 1] to a 0-100 "how strongly does the available evidence
    # support CONTESTING (keeping the money)" score. Only computed over
    # evidence that's actually present — missing items don't get a vote.
    if signal_values:
        avg_signal = sum(signal_values) / len(signal_values)
        strength = round((avg_signal + 1) / 2 * 100, 1)
    else:
        strength = None

    return {
        "dispute_reason": reason,
        "critical_evidence": [EVIDENCE_FIELD_CHECKS[k]["label"] for k in critical_keys],
        "supporting_evidence": [EVIDENCE_FIELD_CHECKS[k]["label"] for k in supporting_keys],
        "evidence_items": items,
        "completeness_score": completeness,
        "strength_score": strength,
        "missing_evidence": missing,
        "critical_missing": critical_missing,
        "has_critical_gap": len(critical_missing) > 0,
    }