"""
evidence_agent.py
The reasoning layer of the Chargeback Evidence Responder.

Takes a dispute record (raw transaction fields + engineered features +
the classifier's fraud probability) and produces a structured decision:
  - action: ACCEPT | CONTEST | ESCALATE  (bounded — see guardrails.py)
  - reasoning: a short, evidence-citing explanation a human reviewer
    (or a chargeback response form) can actually use.

Every decision passes through guardrails.decide_with_guardrails(), which
enforces the bounded action set and the missing-data escalation rule
BEFORE anything is logged or shown to the user. The LLM never has the
final word on its own — the guardrail layer does.

Two modes:
  - LLM mode: calls the Anthropic API to generate reasoning (requires
    ANTHROPIC_API_KEY in environment). Used for the real demo.
  - Rule-based fallback mode: used automatically if no API key is set,
    or for fast local testing without burning API calls. Keeps the
    pipeline runnable end-to-end at all times.
"""

import os
import json
from guardrails import decide_with_guardrails, DisputeAction
from evidence_checklist import assess_evidence

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT = """You are a chargeback evidence assistant for a payments platform.
You are given a disputed transaction's features, a fraud-risk model's
probability score, and an evidence completeness packet (which evidence
items were required for this dispute reason and which were found). Your
job is to recommend exactly ONE action:

- ACCEPT: accept the cardholder's dispute — side with the cardholder,
  issue a refund, do NOT contest. Use when evidence suggests the dispute
  is legitimate (i.e. fraud is likely — the merchant should not fight it)
- CONTEST: contest/represent the dispute — push back with evidence,
  refuse the refund. Use when evidence suggests the transaction was
  legitimate and the dispute itself is likely friendly fraud or a mistake

(You are only called when evidence completeness already cleared the
threshold, so always choose ACCEPT or CONTEST — never ESCALATE.)

Respond in strict JSON with exactly two keys: "action" and "reasoning".
"reasoning" must be 2-4 sentences, must cite specific evidence items you
were given (e.g. distance, device match, delivery confirmation, amount
pattern), and must NOT invent facts not present in the input. Do not
include any text outside the JSON object.
"""


def _build_user_prompt(record: dict, evidence_packet: dict) -> str:
    evidence_lines = "\n".join(
        f"  - {item['label']}: {item['detail']}" for item in evidence_packet["evidence_items"]
    )
    return f"""Disputed transaction:
- Amount: {record.get('amt')}
- Category: {record.get('category')}
- Fraud risk model probability: {record.get('fraud_probability'):.2%}
- Dispute reason given by cardholder: {evidence_packet['dispute_reason']}

Evidence packet (completeness {evidence_packet['completeness_score']:.0f}%,
required for this dispute reason):
{evidence_lines}

Recommend ACCEPT or CONTEST with reasoning citing the evidence above."""


def _propose_action_llm(record: dict, evidence_packet: dict) -> dict:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(record, evidence_packet)}],
    )
    text = response.content[0].text.strip()
    # Strip accidental markdown fences if the model adds them
    text = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)
    return {"action": parsed.get("action", ""), "reasoning": parsed.get("reasoning", "")}


def _propose_action_rule_based(record: dict, evidence_packet: dict) -> dict:
    """
    Deterministic fallback used when no LLM API key is available.
    Mirrors the kind of reasoning the LLM prompt asks for, so the
    pipeline (and demo) never breaks, and so guardrail behavior can be
    tested without API access.
    """
    prob = record.get("fraud_probability", 0)
    device_match = record.get("synth_device_ip_match")
    z = record.get("amt_zscore_in_category", 0)
    dist = record.get("distance_km", 0)
    score = evidence_packet["completeness_score"]

    if prob >= 0.5:
        reasoning = (
            f"Fraud risk model scored this transaction at {prob:.0%}, and the "
            f"evidence packet is {score:.0f}% complete for a "
            f"'{evidence_packet['dispute_reason']}' dispute. Amount z-score of "
            f"{z:.2f} indicates an unusual purchase for this category, and "
            f"device/IP match is {device_match}, consistent with an "
            f"unauthorized transaction. Recommend ACCEPTING the dispute — "
            f"i.e. siding with the cardholder and issuing a refund, since the "
            f"evidence points to fraud."
        )
        action = DisputeAction.ACCEPT.value
    else:
        reasoning = (
            f"Fraud risk model scored this transaction at {prob:.0%}, below the "
            f"contest threshold, with a {score:.0f}% complete evidence packet "
            f"for a '{evidence_packet['dispute_reason']}' dispute. Device/IP "
            f"match is {device_match} and the transaction occurred {dist:.1f} km "
            f"from the customer's registered address, consistent with normal "
            f"cardholder behavior. Recommend CONTESTING the dispute — i.e. "
            f"pushing back with this evidence and declining the refund."
        )
        action = DisputeAction.CONTEST.value

    return {"action": action, "reasoning": reasoning}


def get_evidence_decision(record: dict, use_llm: bool = None) -> dict:
    """
    Main entry point — runs the full multi-step workflow:
      1. Identify the dispute reason and look up required evidence
         (evidence_checklist.assess_evidence)
      2. Check what evidence is actually present, compute a completeness
         score
      3. Pass everything through guardrails: if the classifier score is
         missing OR evidence completeness is below threshold, escalate
         immediately — no LLM/rule call happens
      4. Otherwise, call the LLM (or rule-based fallback) to reason over
         the evidence packet and recommend ACCEPT/CONTEST
      5. Bounded-action enforcement on whatever comes back

    Returns the decision dict plus the evidence_packet for display/audit.
    """
    if use_llm is None:
        use_llm = _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))

    evidence_packet = assess_evidence(record)

    def propose_fn(rec):
        if use_llm:
            return _propose_action_llm(rec, evidence_packet)
        return _propose_action_rule_based(rec, evidence_packet)

    decision = decide_with_guardrails(record, propose_fn, evidence_packet=evidence_packet)
    decision["evidence_packet"] = evidence_packet
    return decision


if __name__ == "__main__":
    # Quick smoke test with a few hand-built records, including one that
    # triggers the missing-data guardrail.
    test_records = [
        {
            "trans_num": "demo_fraud_1",
            "amt": 548.0, "category": "shopping_net", "fraud_probability": 0.91,
            "distance_km": 340.2, "amt_zscore_in_category": 3.1, "hour": 2,
            "synth_device_ip_match": False, "synth_delivery_confirmed": False,
            "synth_dispute_reason": "unauthorized_transaction",
            "synth_days_to_dispute": 2.1, "synth_prior_dispute_count": 0,
        },
        {
            "trans_num": "demo_legit_1",
            "amt": 42.0, "category": "grocery_pos", "fraud_probability": 0.03,
            "distance_km": 5.4, "amt_zscore_in_category": -0.2, "hour": 14,
            "synth_device_ip_match": True, "synth_delivery_confirmed": True,
            "synth_dispute_reason": "product_not_as_described",
            "synth_days_to_dispute": 12.0, "synth_prior_dispute_count": 0,
        },
        {
            "trans_num": "demo_missing_data_1",
            "amt": 120.0, "category": "misc_net", "fraud_probability": 0.4,
            "distance_km": 88.0, "amt_zscore_in_category": 0.5, "hour": 20,
            "synth_device_ip_match": True, "synth_delivery_confirmed": None,
            "synth_dispute_reason": "product_not_received",
            "synth_days_to_dispute": 6.0, "synth_prior_dispute_count": 1,
        },
    ]

    print(f"Running in {'LLM' if (_ANTHROPIC_AVAILABLE and os.environ.get('ANTHROPIC_API_KEY')) else 'rule-based fallback'} mode\n")

    for rec in test_records:
        decision = get_evidence_decision(rec)
        print(f"--- {rec['trans_num']} ---")
        print(f"Action: {decision['action']}")
        print(f"Reasoning: {decision['reasoning']}")
        print(f"Guardrail triggered: {decision['guardrail_triggered']}")
        print()