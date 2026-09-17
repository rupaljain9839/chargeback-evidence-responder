"""
evidence_agent.py
The reasoning layer of the Chargeback Evidence Responder.

Takes a dispute record (raw transaction fields + engineered features +
the classifier's fraud probability) and produces a structured decision:
  - action: ACCEPT | CONTEST | ESCALATE  (bounded — see guardrails.py)
  - reasoning: a short, evidence-citing explanation a human reviewer
    (or a chargeback response form) can actually use.

Every decision passes through guardrails.decide_with_guardrails(), which
enforces the bounded action set and evidence-based escalation BEFORE
anything is logged or shown to the user. The LLM never has the final
word on its own — the guardrail layer does.

Three modes, in priority order:
  - Groq (fast, generous free tier — good default for a student project):
    requires GROQ_API_KEY in environment.
  - Anthropic (Claude): requires ANTHROPIC_API_KEY in environment.
    Used if Groq isn't configured.
  - Rule-based fallback: used automatically if neither key is present,
    or for fast local testing without burning API calls. Keeps the
    pipeline runnable end-to-end at all times, in every mode.
"""

import os
import json
from guardrails import decide_with_guardrails, DisputeAction
from evidence_checklist import assess_evidence

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads GROQ_API_KEY / ANTHROPIC_API_KEY from a .env
    # file in the project root, if one exists — no-ops safely if it
    # doesn't (falls back to whatever's already in the environment).
except ImportError:
    pass

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# Groq model to use for reasoning. Configurable via env var so this can
# be swapped without touching code (e.g. a smaller/faster model for
# quick demos vs. a larger one for final testing).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


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


def _extract_json(text: str) -> dict:
    """Strip accidental markdown fences some models add, then parse."""
    text = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def _propose_action_anthropic(record: dict, evidence_packet: dict) -> dict:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(record, evidence_packet)}],
    )
    text = response.content[0].text
    parsed = _extract_json(text)
    return {"action": parsed.get("action", ""), "reasoning": parsed.get("reasoning", "")}


def _propose_action_groq(record: dict, evidence_packet: dict) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=300,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(record, evidence_packet)},
        ],
    )
    text = response.choices[0].message.content
    parsed = _extract_json(text)
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


def _select_provider() -> str:
    """
    Priority order: Groq (free-tier friendly) > Anthropic > rule-based.
    Explicit env var LLM_PROVIDER=groq|anthropic|none overrides this if set.
    """
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if forced in ("groq", "anthropic", "none"):
        return forced
    if _GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
        return "groq"
    if _ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def get_evidence_decision(record: dict, use_llm: bool = None) -> dict:
    """
    Main entry point — runs the full multi-step workflow:
      1. Identify the dispute reason and look up required evidence
         (evidence_checklist.assess_evidence)
      2. Check what evidence is actually present, compute a completeness
         score
      3. Pass everything through guardrails: if the classifier score is
         missing OR critical evidence for this dispute reason is
         missing, escalate immediately — no LLM/rule call happens
      4. Otherwise, call Groq, Anthropic, or the rule-based fallback
         (in that priority order, or as forced by LLM_PROVIDER) to
         reason over the evidence packet and recommend ACCEPT/CONTEST
      5. Bounded-action enforcement on whatever comes back

    `use_llm=False` forces the rule-based path regardless of configured
    keys (useful for fast local testing without spending API credits).
    Returns the decision dict plus the evidence_packet for display/audit.
    """
    evidence_packet = assess_evidence(record)

    if use_llm is False:
        provider = "none"
    else:
        provider = _select_provider()

    def propose_fn(rec):
        if provider == "groq":
            return _propose_action_groq(rec, evidence_packet)
        if provider == "anthropic":
            return _propose_action_anthropic(rec, evidence_packet)
        return _propose_action_rule_based(rec, evidence_packet)

    decision = decide_with_guardrails(record, propose_fn, evidence_packet=evidence_packet)
    decision["evidence_packet"] = evidence_packet
    decision["llm_provider_used"] = provider
    return decision


if __name__ == "__main__":
    # Quick smoke test with a few hand-built records, including one that
    # triggers the missing-critical-evidence guardrail.
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
            "trans_num": "demo_critical_gap_1",
            "amt": 120.0, "category": "misc_net", "fraud_probability": 0.4,
            "distance_km": 88.0, "amt_zscore_in_category": 0.5, "hour": 20,
            "synth_device_ip_match": None,  # critical for unauthorized_transaction
            "synth_delivery_confirmed": True,
            "synth_dispute_reason": "unauthorized_transaction",
            "synth_days_to_dispute": 6.0, "synth_prior_dispute_count": 1,
        },
    ]

    provider = _select_provider()
    print(f"Running with provider: {provider}\n")

    for rec in test_records:
        decision = get_evidence_decision(rec)
        print(f"--- {rec['trans_num']} ---")
        print(f"Action: {decision['action']}")
        print(f"Reasoning: {decision['reasoning']}")
        print(f"Guardrail triggered: {decision['guardrail_triggered']}")
        print()