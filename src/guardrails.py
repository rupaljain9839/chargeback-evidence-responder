"""
guardrails.py
Enforces the "explainable, bounded and gated" requirement for every
money-affecting action the Evidence Agent takes.

Two responsibilities:
  1. BOUNDED ACTION SET — the agent may only ever return one of a fixed
     set of actions. Anything else (including free-form LLM output that
     doesn't parse cleanly) is rejected and forced to ESCALATE.
  2. FAILURE-CASE HANDLING — if required data is missing (e.g. delivery
     status wasn't recorded), the agent must not guess. It escalates to
     a human reviewer instead. This is the one failure case we deliberately
     inject (via features.py) and handle end-to-end.
"""

from enum import Enum


class DisputeAction(str, Enum):
    ACCEPT = "ACCEPT"        # agree with cardholder, refund
    CONTEST = "CONTEST"      # push back with evidence, do not refund
    ESCALATE = "ESCALATE"    # insufficient/ambiguous data -> human review


ALLOWED_ACTIONS = {a.value for a in DisputeAction}


class GuardrailViolation(Exception):
    """Raised when the agent attempts (or the LLM proposes) an action
    outside the bounded set, or when required inputs are missing and
    the caller tries to bypass escalation."""
    pass


def check_required_fields(record: dict) -> list:
    """
    Checks for the one input every decision absolutely needs regardless
    of dispute reason: a classifier score. Dispute-reason-specific
    evidence (delivery confirmation, device match, etc.) is checked by
    the evidence-completeness guardrail instead (see evidence_checklist.py
    and the `evidence_incomplete` path in decide_with_guardrails), since
    which evidence is "required" depends on the dispute reason.
    """
    missing = []
    if record.get("fraud_probability") is None:
        missing.append("classifier_score")
    return missing


def enforce_bounded_action(proposed_action: str) -> str:
    """
    Validates a proposed action string against the bounded set.
    Any unrecognized value is forced to ESCALATE rather than trusted —
    this is the hard gate: the agent can never take an action outside
    {ACCEPT, CONTEST, ESCALATE}.
    """
    normalized = (proposed_action or "").strip().upper()
    if normalized not in ALLOWED_ACTIONS:
        return DisputeAction.ESCALATE.value
    return normalized


def decide_with_guardrails(record: dict, propose_action_fn, evidence_packet: dict = None) -> dict:
    """
    Wraps any action-proposing function (rule-based or LLM-based) with
    three guardrails, checked in order:
      1. missing required inputs (classifier score absent)
      2. CRITICAL evidence missing for this dispute reason (see
         evidence_checklist.py) — not a blanket completeness threshold.
         Missing a critical item (e.g. device/IP match on an
         unauthorized-transaction dispute) always escalates, regardless
         of the overall completeness percentage. Missing only
         supporting/non-critical evidence does NOT by itself escalate.
      3. bounded-action enforcement on whatever the proposer returns

    `propose_action_fn` must accept `record` and return a dict with at
    least {"action": str, "reasoning": str}.

    This function is the single choke point every dispute decision must
    pass through — nothing downstream (audit log, UI) sees a decision
    that skipped this gate.
    """
    missing = check_required_fields(record)
    if missing:
        return {
            "action": DisputeAction.ESCALATE.value,
            "reasoning": (
                f"Escalated to human review: required field(s) missing or "
                f"unusable ({', '.join(missing)}). The agent does not guess "
                f"when evidence is incomplete."
            ),
            "guardrail_triggered": "missing_data",
        }

    if evidence_packet is not None and evidence_packet.get("has_critical_gap"):
        critical_missing = ", ".join(evidence_packet.get("critical_missing", [])) or "critical evidence"
        return {
            "action": DisputeAction.ESCALATE.value,
            "reasoning": (
                f"Escalated to human review: critical evidence is missing for "
                f"a '{evidence_packet['dispute_reason']}' dispute — "
                f"{critical_missing}. This evidence is required to safely "
                f"auto-decide this dispute type; supporting evidence alone "
                f"isn't enough to substitute for it, so the case is not "
                f"auto-decided even though the overall completeness score "
                f"is {evidence_packet['completeness_score']:.0f}%."
            ),
            "guardrail_triggered": "critical_evidence_missing",
        }

    proposed = propose_action_fn(record)
    final_action = enforce_bounded_action(proposed.get("action", ""))

    result = {
        "action": final_action,
        "reasoning": proposed.get("reasoning", ""),
        "guardrail_triggered": None if final_action == proposed.get("action", "").strip().upper() else "invalid_action_forced_escalate",
    }
    return result