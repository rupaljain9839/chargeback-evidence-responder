"""
streamlit_app.py
Live demo interface for the Chargeback Evidence Responder.

Design: "case ledger / audit desk" — dark ink background, monospace
stamped-verdict treatment for ACCEPT/CONTEST/ESCALATE, restrained
risk-semantic accent palette. Custom CSS injected via st.markdown.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import uuid
import math
import streamlit as st
import pandas as pd
import joblib

from features import build_feature_table, FEATURE_COLUMNS
from evidence_agent import get_evidence_decision
from evidence_checklist import EVIDENCE_REQUIREMENTS
from audit_log import log_decision, read_audit_trail

st.set_page_config(
    page_title="Chargeback Evidence Responder",
    page_icon="📒",
    layout="wide",
)

# ---------------------------------------------------------------------
# DESIGN TOKENS + CUSTOM CSS
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #0B1220;
  --panel: #121B2E;
  --panel-border: #24304a;
  --muted: #A9B2C6;
  --text: #FFFFFF;
  --safe: #2FBF8F;
  --fraud: #D65B4A;
  --escalate: #E8A33D;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
  background-color: var(--ink);
  color: var(--text);
}

/* Hero header */
.ledger-hero {
  border-bottom: 1px solid var(--panel-border);
  padding: 1.4rem 0 1.2rem 0;
  margin-bottom: 1.6rem;
}
.ledger-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--escalate);
  font-size: 0.85rem;
  font-weight: 600;
  margin-bottom: 0.35rem;
}
.ledger-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--text);
  margin: 0;
  letter-spacing: -0.01em;
}
.ledger-sub {
  color: var(--muted);
  font-size: 1.08rem;
  margin-top: 0.4rem;
  max-width: 680px;
}

/* Legend chips */
.legend-row { display: flex; gap: 0.6rem; margin-top: 0.9rem; flex-wrap: wrap; }
.legend-chip {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  padding: 0.35rem 0.7rem;
  border-radius: 3px;
  border: 1.5px solid;
  text-transform: uppercase;
}
.legend-accept { color: var(--fraud); border-color: var(--fraud); background: rgba(214,91,74,0.08); }
.legend-contest { color: var(--safe); border-color: var(--safe); background: rgba(47,191,143,0.08); }
.legend-escalate { color: var(--escalate); border-color: var(--escalate); background: rgba(232,163,61,0.08); }

.data-disclosure {
  margin-top: 0.9rem;
  font-size: 0.8rem;
  color: var(--muted);
  border-top: 1px dashed var(--panel-border);
  padding-top: 0.7rem;
  max-width: 700px;
}
.data-disclosure code {
  color: var(--escalate);
  background: rgba(232,163,61,0.08);
  padding: 0.05rem 0.3rem;
  border-radius: 3px;
}

/* Case file panel */
.case-panel {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  padding: 1.1rem 1.3rem;
}
.case-row {
  display: flex;
  justify-content: space-between;
  padding: 0.55rem 0;
  border-bottom: 1px dashed var(--panel-border);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1rem;
}
.case-row:last-child { border-bottom: none; }
.case-label { color: var(--muted); font-weight: 500; }
.case-value { color: var(--text); font-weight: 600; }
.case-value.flag-missing { color: var(--escalate); font-weight: 700; }

/* Verdict stamp */
.stamp-wrap { display: flex; justify-content: center; padding: 1.6rem 0 0.6rem 0; }
.stamp {
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 700;
  font-size: 1.35rem;
  letter-spacing: 0.1em;
  border: 3px solid;
  border-radius: 8px;
  padding: 0.5rem 1.6rem;
  transform: rotate(-4deg);
  text-transform: uppercase;
  display: inline-block;
  text-align: center;
}
.stamp-sub {
  display: block;
  font-size: 0.62rem;
  letter-spacing: 0.08em;
  font-weight: 600;
  margin-top: 0.2rem;
  opacity: 0.85;
}
.stamp-accept { color: var(--fraud); border-color: var(--fraud); }
.stamp-contest { color: var(--safe); border-color: var(--safe); }
.stamp-escalate { color: var(--escalate); border-color: var(--escalate); }

/* KPI cards */
.kpi-card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  padding: 0.9rem 1rem;
  text-align: center;
}
.kpi-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.kpi-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.9rem;
  font-weight: 700;
  color: var(--text);
  margin-top: 0.2rem;
}

/* Reasoning box */
.reasoning-box {
  background: var(--panel);
  border-left: 3px solid var(--escalate);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  font-size: 1.02rem;
  color: var(--text);
  line-height: 1.6;
}

/* Evidence checklist */
.evidence-panel {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  padding: 1rem 1.2rem;
}
.evidence-score-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.7rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--panel-border);
}
.evidence-score-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.evidence-score-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.5rem;
  font-weight: 700;
}
.evidence-item {
  display: flex;
  gap: 0.6rem;
  padding: 0.4rem 0;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.95rem;
}
.evidence-check { width: 1.2rem; font-weight: 700; }
.evidence-check.yes { color: var(--safe); }
.evidence-check.no { color: var(--fraud); }
.evidence-detail { color: var(--muted); font-size: 0.88rem; }
.crit-tag {
  font-size: 0.62rem;
  font-weight: 700;
  color: var(--escalate);
  border: 1px solid var(--escalate);
  border-radius: 3px;
  padding: 0.05rem 0.3rem;
  margin-left: 0.3rem;
  letter-spacing: 0.04em;
}
.critical-gap-note {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  color: var(--escalate);
  background: rgba(232,163,61,0.1);
  border: 1px solid var(--escalate);
  border-radius: 4px;
  padding: 0.5rem 0.7rem;
  margin-bottom: 0.6rem;
}

/* Streamlit chrome overrides */
[data-testid="stTabs"] button { font-family: 'IBM Plex Mono', monospace; font-size: 0.98rem; font-weight: 600; }
[data-testid="stTabs"] button p { font-size: 0.98rem !important; }
.stMarkdown, .stMarkdown p, .stRadio label p { color: var(--text) !important; font-size: 1rem !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; font-size: 0.9rem !important; }
.stButton button {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1rem;
  font-weight: 600;
  border-radius: 4px;
  letter-spacing: 0.03em;
}
hr { border-color: var(--panel-border) !important; }

/* Form widget overrides — number inputs, selects, sliders, radios all
   ship with light-mode styling by default; force them to match the
   dark theme so labels and values are actually readable. */
.stNumberInput label p, .stSelectbox label p, .stSlider label p,
.stRadio label p, .stTextInput label p {
  color: var(--text) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
}

.stNumberInput input, .stTextInput input {
  background-color: var(--panel) !important;
  color: var(--text) !important;
  border: 1px solid var(--panel-border) !important;
  font-family: 'IBM Plex Mono', monospace !important;
}
.stNumberInput button {
  background-color: var(--panel) !important;
  border: 1px solid var(--panel-border) !important;
  color: var(--text) !important;
}
.stNumberInput button svg { fill: var(--text) !important; }

[data-baseweb="select"] > div {
  background-color: var(--panel) !important;
  border-color: var(--panel-border) !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }
[data-baseweb="popover"] { background-color: var(--panel) !important; }
[data-baseweb="menu"] { background-color: var(--panel) !important; }
[data-baseweb="menu"] li { color: var(--text) !important; }
[data-baseweb="menu"] li:hover { background-color: var(--panel-border) !important; }

[data-baseweb="radio"] label, [data-baseweb="radio"] div {
  color: var(--text) !important;
}
[data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 600 !important; }

.stSlider [data-baseweb="slider"] { color: var(--text); }
.stSlider [role="slider"] { background-color: var(--fraud) !important; }

/* Form/expander containers */
[data-testid="stForm"] {
  background-color: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  padding: 1.2rem 1.4rem;
}
[data-testid="stExpander"] {
  background-color: var(--panel) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 6px !important;
}
[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600 !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------
st.markdown("""
<div class="ledger-hero">
  <div class="ledger-eyebrow">AI Risk Manager · Razorpay AI Buildathon</div>
  <p class="ledger-title">Chargeback Evidence Responder</p>
  <p class="ledger-sub">
    Scores disputed transactions for fraud risk, then reasons over the
    evidence to recommend one of three bounded actions — never a guess,
    never an unbounded action.
  </p>
  <div class="legend-row">
    <span class="legend-chip legend-accept">Accept — refund cardholder</span>
    <span class="legend-chip legend-contest">Contest — evidence supports merchant</span>
    <span class="legend-chip legend-escalate">Escalate — insufficient data</span>
  </div>
  <div class="data-disclosure">
    Prototype evaluation uses real transaction patterns (Sparkov-simulated
    dataset) with controlled synthetic dispute-evidence fields layered on
    top — not proprietary production dispute data. Full breakdown in
    <code>data/synthetic_fields.md</code>.
  </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/classifier.pkl")


@st.cache_data
def load_data():
    return build_feature_table("data/raw/fraudTrain_150k.csv")


model = load_model()
df = load_data()


@st.cache_data
def get_category_stats(_df):
    """Real per-category amount mean/std, computed from the same training
    data the classifier saw — used so manually-entered disputes get their
    amt_zscore computed the same way as sample cases, not a guess."""
    stats = _df.groupby("category")["amt"].agg(["mean", "std"]).to_dict("index")
    return stats


category_stats = get_category_stats(df)


@st.cache_data
def load_feature_importances():
    """Real feature importances from the trained model's evaluation run
    (evaluation/metrics_report.json) — not fabricated per-instance
    attribution. Used to label which signals the model actually weighs
    most heavily, globally, alongside this case's own values."""
    try:
        with open("evaluation/metrics_report.json") as f:
            report = json.load(f)
        return report.get("feature_importances", {})
    except FileNotFoundError:
        return {}


FEATURE_LABELS = {
    "amt": "Transaction amount",
    "hour": "Transaction hour",
    "day_of_week": "Day of week",
    "is_night": "Late-night transaction",
    "age": "Customer age",
    "distance_km": "Distance from home",
    "amt_zscore_in_category": "Amount vs. category norm",
    "city_pop_log": "Customer city population",
    "synth_device_ip_match": "Device/IP match",
    "synth_prior_dispute_count": "Prior dispute count",
    "synth_days_to_dispute": "Time to dispute filing",
}


def _flag_feature(key: str, value) -> str:
    """
    Returns 'elevated' | 'normal' | 'context' for a feature value.
    This is a simple, documented threshold rule per feature — NOT a
    model-derived per-instance attribution (no SHAP here). Features
    without a clear directional heuristic are labeled 'context' rather
    than force-fit into elevated/normal.
    """
    if value is None:
        return "context"
    try:
        if key == "amt_zscore_in_category":
            return "elevated" if abs(value) > 1.5 else "normal"
        if key == "distance_km":
            return "elevated" if value > 100 else "normal"
        if key == "is_night":
            return "elevated" if value == 1 else "normal"
        if key == "synth_device_ip_match":
            return "elevated" if not value else "normal"
        if key == "synth_prior_dispute_count":
            return "elevated" if value >= 1 else "normal"
        if key == "synth_days_to_dispute":
            return "elevated" if value < 3 else "normal"
    except TypeError:
        return "context"
    return "context"


def _format_feature_value(key: str, value) -> str:
    if value is None:
        return "n/a"
    if key == "amt":
        return f"₹{value:.2f}"
    if key == "distance_km":
        return f"{value:.1f} km"
    if key == "amt_zscore_in_category":
        return f"{value:.2f}σ"
    if key == "is_night":
        return "Yes" if value == 1 else "No"
    if key == "synth_device_ip_match":
        return str(bool(value))
    if key == "hour":
        return f"{int(value):02d}:00"
    if key == "city_pop_log":
        return f"~{int(2.71828 ** value):,}"
    return str(value)


def render_risk_factors(record: dict, top_n: int = 6):
    """
    'Key Evidence Signals' panel — shows the model's real, globally
    computed feature importances (from training evaluation) alongside
    this specific case's values. This is deliberately NOT labeled as
    per-instance SHAP attribution, since none is computed here — it's
    global importance + this case's own values, honestly framed.
    """
    importances = load_feature_importances()
    if not importances:
        return
    ranked = sorted(importances.items(), key=lambda kv: -kv[1])[:top_n]

    rows_html = ""
    for key, weight in ranked:
        if key not in record:
            continue
        value = record.get(key)
        flag = _flag_feature(key, value)
        icon = {"elevated": "🔴", "normal": "🟢", "context": "⚪"}[flag]
        label = FEATURE_LABELS.get(key, key)
        display_val = _format_feature_value(key, value)
        rows_html += (
            f'<div class="evidence-item">'
            f'<span class="evidence-check">{icon}</span>'
            f'<span>{label}'
            f'<br><span class="evidence-detail">{display_val} · model weight {weight*100:.1f}%</span></span>'
            f'</div>'
        )

    if rows_html:
        st.markdown("**Key evidence signals**")
        st.markdown(f'<div class="evidence-panel">{rows_html}</div>', unsafe_allow_html=True)
        st.caption(
            "Weights are the trained model's real global feature importances "
            "(not per-prediction SHAP attribution). 🔴 = this value is in the "
            "range associated with higher risk; 🟢 = typical range."
        )


def render_decision_block(decision: dict, prob: float, record: dict = None, actual_label=None):
    """Shared result renderer — used by both the sample-case tab and the
    manual 'Analyze New Dispute' tab so both produce an identical,
    consistent decision display."""
    packet = decision.get("evidence_packet")

    stamp_class = {"ACCEPT": "stamp-accept", "CONTEST": "stamp-contest", "ESCALATE": "stamp-escalate"}[decision["action"]]
    stamp_sub = {
        "ACCEPT": "DISPUTE ACCEPTED — REFUND CARDHOLDER",
        "CONTEST": "DISPUTE CONTESTED — NO REFUND",
        "ESCALATE": "SENT TO HUMAN REVIEWER",
    }[decision["action"]]
    st.markdown(f"""
    <div class="stamp-wrap">
      <div class="stamp {stamp_class}">{decision['action']}<span class="stamp-sub">{stamp_sub}</span></div>
    </div>
    """, unsafe_allow_html=True)

    business_line = {
        "ACCEPT": "The available evidence supports the cardholder's dispute. The merchant should accept the dispute rather than defend the transaction.",
        "CONTEST": "The available evidence supports defending the transaction. The merchant can submit the evidence below to challenge the dispute.",
        "ESCALATE": "The system does not have sufficient confidence or evidence for an automated decision. Human review is required before acting.",
    }[decision["action"]]
    st.caption(business_line)

    n_cols = 5 if actual_label is not None else 4
    cols = st.columns(n_cols)
    with cols[0]:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Fraud risk score</div>
            <div class="kpi-value">{prob:.1%}</div></div>""", unsafe_allow_html=True)
    with cols[1]:
        score = packet["completeness_score"] if packet else 0
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Evidence completeness</div>
            <div class="kpi-value">{score:.0f}%</div></div>""", unsafe_allow_html=True)
    with cols[2]:
        strength = packet["strength_score"] if packet and packet["strength_score"] is not None else None
        strength_txt = f"{strength:.0f}%" if strength is not None else "—"
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Merchant evidence strength</div>
            <div class="kpi-value">{strength_txt}</div></div>""", unsafe_allow_html=True)
    col_idx = 3
    if actual_label is not None:
        with cols[3]:
            actual_txt = "FRAUD" if actual_label == 1 else "LEGITIMATE"
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Actual label</div>
                <div class="kpi-value">{actual_txt}</div></div>""", unsafe_allow_html=True)
        col_idx = 4
    with cols[col_idx]:
        gr = decision["guardrail_triggered"] or "none"
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Guardrail</div>
            <div class="kpi-value" style="font-size:1.1rem;">{gr}</div></div>""", unsafe_allow_html=True)

    st.write("")

    col_reason, col_evidence = st.columns([3, 2])
    with col_reason:
        st.markdown("**Agent reasoning**")
        st.markdown(f'<div class="reasoning-box">{decision["reasoning"]}</div>', unsafe_allow_html=True)

    with col_evidence:
        if packet:
            st.markdown(f"**Evidence checklist** — *{packet['dispute_reason']}*")
            supports_label_map = {
                "merchant": "Supports merchant", "cardholder": "Supports cardholder",
                "neutral": "Present, no strong signal", "missing": "Missing",
            }
            items_html = ""
            for item in packet["evidence_items"]:
                mark = "✓" if item["present"] else ("⚠" if item["critical"] else "✗")
                cls = "yes" if item["present"] else "no"
                crit_tag = ' <span class="crit-tag">CRITICAL</span>' if item["critical"] else ""
                supports_txt = supports_label_map[item["supports"]]
                items_html += (
                    f'<div class="evidence-item">'
                    f'<span class="evidence-check {cls}">{mark}</span>'
                    f'<span>{item["label"]}{crit_tag}'
                    f'<br><span class="evidence-detail">{item["detail"]} — <i>{supports_txt}</i></span></span>'
                    f'</div>'
                )
            gap_note = ""
            if packet.get("has_critical_gap"):
                gap_note = (
                    f'<div class="critical-gap-note">⚠ Critical evidence missing for this '
                    f'dispute reason — case will escalate regardless of completeness %.</div>'
                )
            st.markdown(f"""
            <div class="evidence-panel">
              <div class="evidence-score-row">
                <span class="evidence-score-label">Completeness</span>
                <span class="evidence-score-value">{packet['completeness_score']:.0f}%</span>
              </div>
              {gap_note}
              {items_html}
            </div>
            """, unsafe_allow_html=True)

    if record:
        st.write("")
        render_risk_factors(record)

    st.caption("Logged to the audit trail — see the Audit Trail tab.")


tab1, tab_manual, tab2, tab3 = st.tabs([
    "📁  Live Dispute Review", "➕  Analyze New Dispute", "📊  Model Metrics", "🗒️  Audit Trail",
])

# ---------------------------------------------------------------------
# TAB 1: Live dispute review
# ---------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**Case selection**")
        sample_type = st.radio(
            "Sample from:",
            ["Random transaction", "Known fraud case", "Known legitimate case",
             "Missing-data case (triggers escalation)"],
            label_visibility="collapsed",
        )
        if st.button("Pull new case", type="primary", use_container_width=True):
            if sample_type == "Known fraud case":
                row = df[df["is_fraud"] == 1].sample(1).iloc[0]
            elif sample_type == "Known legitimate case":
                row = df[df["is_fraud"] == 0].sample(1).iloc[0]
            elif sample_type == "Missing-data case (triggers escalation)":
                # Only sample rows where the missing delivery field is
                # actually required for that row's dispute reason —
                # otherwise the completeness score can still hit 100%
                # and the case won't escalate (e.g. credit_not_processed
                # doesn't need delivery confirmation at all).
                reasons_needing_delivery = [
                    r for r, reqs in EVIDENCE_REQUIREMENTS.items()
                    if "delivery_confirmation" in reqs
                ]
                missing = df[
                    df["synth_delivery_confirmed"].isna()
                    & df["synth_dispute_reason"].isin(reasons_needing_delivery)
                ]
                row = missing.sample(1).iloc[0]
            else:
                row = df.sample(1).iloc[0]
            st.session_state["current_row"] = row
            st.session_state.pop("decision", None)

    if "current_row" not in st.session_state:
        st.session_state["current_row"] = df.sample(1).iloc[0]

    row = st.session_state["current_row"]
    delivery_val = row["synth_delivery_confirmed"]
    delivery_display = "MISSING" if pd.isna(delivery_val) else str(bool(delivery_val))
    delivery_class = "flag-missing" if pd.isna(delivery_val) else ""

    with col_b:
        st.markdown(f"""
        <div class="case-panel">
          <div class="case-row"><span class="case-label">CASE ID</span><span class="case-value">{row['trans_num'][:24]}</span></div>
          <div class="case-row"><span class="case-label">AMOUNT</span><span class="case-value">₹{row['amt']:.2f}</span></div>
          <div class="case-row"><span class="case-label">CATEGORY</span><span class="case-value">{row['category']}</span></div>
          <div class="case-row"><span class="case-label">HOUR</span><span class="case-value">{int(row['hour']):02d}:00</span></div>
          <div class="case-row"><span class="case-label">DISTANCE FROM HOME</span><span class="case-value">{row['distance_km']:.1f} km</span></div>
          <div class="case-row"><span class="case-label">AMOUNT Z-SCORE</span><span class="case-value">{row['amt_zscore_in_category']:.2f}</span></div>
          <div class="case-row"><span class="case-label">DEVICE/IP MATCH</span><span class="case-value">{bool(row['synth_device_ip_match'])}</span></div>
          <div class="case-row"><span class="case-label">DELIVERY CONFIRMED</span><span class="case-value {delivery_class}">{delivery_display}</span></div>
          <div class="case-row"><span class="case-label">DISPUTE REASON</span><span class="case-value">{row['synth_dispute_reason']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("▶  Run pipeline on this case", type="primary", use_container_width=True)

    if run_clicked:
        X = row[FEATURE_COLUMNS].values.reshape(1, -1)
        prob = float(model.predict_proba(X)[:, 1][0])

        record = row[FEATURE_COLUMNS].to_dict()
        record["fraud_probability"] = prob
        record["category"] = row["category"]
        record["synth_delivery_confirmed"] = row["synth_delivery_confirmed"]
        record["synth_dispute_reason"] = row["synth_dispute_reason"]

        decision = get_evidence_decision(record)
        log_decision(row["trans_num"], record, prob, decision)
        st.session_state["decision"] = decision
        st.session_state["decision_prob"] = prob
        st.session_state["decision_label"] = row["is_fraud"]
        st.session_state["decision_record"] = record

    if "decision" in st.session_state:
        render_decision_block(
            st.session_state["decision"],
            st.session_state["decision_prob"],
            record=st.session_state.get("decision_record"),
            actual_label=st.session_state["decision_label"],
        )

# ---------------------------------------------------------------------
# TAB: Analyze New Dispute (manual entry)
# ---------------------------------------------------------------------
def analyze_manual_dispute(inputs: dict):
    """
    Runs the same feature-engineering + model + evidence-agent pipeline
    the manual form uses. Shared by both the initial form submission and
    the 'What changed the decision?' quick re-run, so a re-run is never
    a separate/lighter code path than the original analysis.
    """
    m_amt = inputs["amt"]
    m_category = inputs["category"]
    m_hour = inputs["hour"]
    m_day_of_week = inputs["day_of_week"]
    m_age = inputs["age"]
    m_distance = inputs["distance"]
    m_device_match = inputs["device_match"]
    m_prior_disputes = inputs["prior_disputes"]
    m_city_pop = inputs["city_pop"]
    m_delivery = inputs["delivery"]
    m_days_to_dispute = inputs["days_to_dispute"]
    m_reason = inputs["reason"]

    is_night = 1 if (m_hour >= 23 or m_hour <= 5) else 0
    cat_stat = category_stats.get(m_category, {"mean": m_amt, "std": 1.0})
    cat_std = cat_stat["std"] if cat_stat["std"] and cat_stat["std"] > 0 else 1.0
    amt_z = (m_amt - cat_stat["mean"]) / cat_std
    city_pop_log = math.log1p(m_city_pop)

    device_match_val = {"Yes": True, "No": False, "Unknown": None}[m_device_match]
    delivery_val = {"Yes": True, "No": False, "Unknown / not available": None}[m_delivery]

    feature_row = {
        "amt": m_amt,
        "hour": m_hour,
        "day_of_week": m_day_of_week,
        "is_night": is_night,
        "age": m_age,
        "distance_km": m_distance,
        "amt_zscore_in_category": amt_z,
        "city_pop_log": city_pop_log,
        # The ML model needs a numeric value even when device match is
        # "Unknown" — it gets 0 (conservative) for the RISK SCORE only.
        # The evidence agent sees the true None separately, so "unknown"
        # is never silently treated as "confirmed absent" in the
        # evidence-completeness reasoning.
        "synth_device_ip_match": 1 if device_match_val else 0,
        "synth_prior_dispute_count": m_prior_disputes,
        "synth_days_to_dispute": m_days_to_dispute,
    }
    X = pd.DataFrame([feature_row])[FEATURE_COLUMNS].values
    prob = float(model.predict_proba(X)[:, 1][0])

    record = dict(feature_row)
    record["fraud_probability"] = prob
    record["category"] = m_category
    record["synth_device_ip_match"] = device_match_val
    record["synth_delivery_confirmed"] = delivery_val
    record["synth_dispute_reason"] = m_reason

    decision = get_evidence_decision(record)
    return decision, prob, record


with tab_manual:
    st.markdown(
        "**Enter a disputed transaction's evidence.** The model infers "
        "risk from what you provide — it is not told the answer."
    )

    dispute_reasons = list(EVIDENCE_REQUIREMENTS.keys())
    categories = sorted(df["category"].unique().tolist())

    with st.form("manual_dispute_form"):
        st.markdown("**Transaction details**")
        c1, c2 = st.columns(2)
        with c1:
            m_amt = st.number_input("Amount (₹)", min_value=0.0, value=100.0, step=10.0)
            default_cat_idx = categories.index("shopping_net") if "shopping_net" in categories else 0
            m_category = st.selectbox("Category", categories, index=default_cat_idx)
        with c2:
            m_hour = st.slider("Transaction hour", 0, 23, 14)
            m_distance = st.number_input("Distance from customer's registered location (km)", min_value=0.0, value=10.0, step=5.0)

        st.markdown("**Customer / security evidence**")
        c4, c5 = st.columns(2)
        with c4:
            m_device_match = st.radio("Device/IP matches customer's known devices?", ["Yes", "No", "Unknown"], horizontal=True)
        with c5:
            m_prior_disputes = st.number_input("Customer's prior dispute count", min_value=0, value=0, step=1)

        st.markdown("**Fulfillment evidence**")
        c7, c8 = st.columns(2)
        with c7:
            m_delivery = st.radio("Delivery confirmed?", ["Yes", "No", "Unknown / not available"], horizontal=True)
        with c8:
            m_days_to_dispute = st.number_input("Days between transaction and dispute filing", min_value=0.0, value=5.0, step=1.0)

        st.markdown("**Dispute details**")
        m_reason = st.selectbox("Dispute reason (cardholder's claim)", dispute_reasons)

        with st.expander("Advanced (optional — used internally by the model, defaults are fine)"):
            st.caption(
                "These affect the model's risk score slightly but aren't things a "
                "merchant typically evaluates when reviewing a dispute — sensible "
                "defaults are pre-filled."
            )
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                m_age = st.number_input("Customer age", min_value=18, max_value=100, value=35)
            with ca2:
                m_city_pop = st.number_input("Customer's city population", min_value=100, value=50000, step=1000)
            with ca3:
                m_day_of_week = st.selectbox(
                    "Day of week", options=list(range(7)),
                    format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                    index=2,
                )

        submitted = st.form_submit_button("🔍  ANALYZE DISPUTE", type="primary", use_container_width=True)

    if submitted:
        inputs = {
            "amt": m_amt, "category": m_category, "hour": m_hour,
            "day_of_week": m_day_of_week, "age": m_age, "distance": m_distance,
            "device_match": m_device_match, "prior_disputes": m_prior_disputes,
            "city_pop": m_city_pop, "delivery": m_delivery,
            "days_to_dispute": m_days_to_dispute, "reason": m_reason,
        }
        decision, prob, record = analyze_manual_dispute(inputs)

        case_id = f"manual-{uuid.uuid4().hex[:16]}"
        log_decision(case_id, record, prob, decision)

        st.session_state["manual_decision"] = decision
        st.session_state["manual_prob"] = prob
        st.session_state["manual_inputs"] = inputs
        st.session_state["manual_record"] = record
        # Clear any stale "what changed" comparison from a previous case
        st.session_state.pop("manual_decision_v2", None)

    if "manual_decision" in st.session_state:
        st.divider()
        render_decision_block(
            st.session_state["manual_decision"],
            st.session_state["manual_prob"],
            record=st.session_state.get("manual_record"),
        )

        # -----------------------------------------------------------
        # "What changed the decision?" — tweak one piece of evidence
        # and re-run instantly through the SAME model, to show the
        # decision responds to evidence rather than being fixed.
        # -----------------------------------------------------------
        st.divider()
        with st.expander("🔁  What changed the decision? — tweak evidence and re-run instantly"):
            base = st.session_state["manual_inputs"]
            st.caption("Adjust one or more fields below, then recalculate. Everything else stays as originally entered.")

            wc1, wc2, wc3 = st.columns(3)
            with wc1:
                w_device_match = st.radio(
                    "Device/IP match", ["Yes", "No", "Unknown"],
                    index=["Yes", "No", "Unknown"].index(base["device_match"]),
                    horizontal=True, key="wc_device",
                )
            with wc2:
                w_delivery = st.radio(
                    "Delivery confirmed", ["Yes", "No", "Unknown / not available"],
                    index=["Yes", "No", "Unknown / not available"].index(base["delivery"]),
                    horizontal=True, key="wc_delivery",
                )
            with wc3:
                w_amt = st.number_input("Amount (₹)", min_value=0.0, value=float(base["amt"]), step=10.0, key="wc_amt")

            if st.button("↻  Recalculate with these changes", type="primary"):
                new_inputs = dict(base)
                new_inputs["device_match"] = w_device_match
                new_inputs["delivery"] = w_delivery
                new_inputs["amt"] = w_amt
                new_decision, new_prob, new_record = analyze_manual_dispute(new_inputs)
                st.session_state["manual_decision_v2"] = new_decision
                st.session_state["manual_prob_v2"] = new_prob
                st.session_state["manual_inputs_v2"] = new_inputs
                st.session_state["manual_record_v2"] = new_record

            if "manual_decision_v2" in st.session_state:
                old_prob = st.session_state["manual_prob"]
                new_prob = st.session_state["manual_prob_v2"]
                old_action = st.session_state["manual_decision"]["action"]
                new_action = st.session_state["manual_decision_v2"]["action"]

                st.markdown("**Before → After**")
                cb1, cb2 = st.columns(2)
                with cb1:
                    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Original</div>
                        <div class="kpi-value" style="font-size:1.3rem;">{old_action}</div>
                        <div style="color:var(--muted); font-family:'IBM Plex Mono',monospace; margin-top:0.3rem;">{old_prob:.1%} fraud risk</div>
                        </div>""", unsafe_allow_html=True)
                with cb2:
                    changed = " (changed)" if new_action != old_action else ""
                    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">After{changed}</div>
                        <div class="kpi-value" style="font-size:1.3rem;">{new_action}</div>
                        <div style="color:var(--muted); font-family:'IBM Plex Mono',monospace; margin-top:0.3rem;">{new_prob:.1%} fraud risk</div>
                        </div>""", unsafe_allow_html=True)

                st.write("")
                render_decision_block(
                    st.session_state["manual_decision_v2"],
                    new_prob,
                    record=st.session_state.get("manual_record_v2"),
                )

# ---------------------------------------------------------------------
# TAB 2: Model metrics
# ---------------------------------------------------------------------
with tab2:
    try:
        with open("evaluation/metrics_report.json") as f:
            report = json.load(f)

        best = next(r for r in report["models_compared"] if r["model"] == report["best_model"])

        st.markdown(f"**Best model: `{report['best_model']}`** — held-out test set")
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            [c1, c2, c3, c4],
            ["Precision", "Recall", "F1", "ROC-AUC"],
            [best["precision"], best["recall"], best["f1"], best["roc_auc"]],
        ):
            col.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div></div>""", unsafe_allow_html=True)

        st.write("")
        st.markdown(f"**Net value on test set:** ₹{best['net_value_inr']:,.2f}")

        st.write("")
        st.markdown("**Confusion matrix**")
        cm = best["confusion_matrix"]
        st.table(pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actual: Legit", "Actual: Fraud"],
            columns=["Predicted: Legit", "Predicted: Fraud"],
        ))

        if "real_features_only_comparison" in report:
            st.divider()
            st.markdown("**Honesty check — real data only, no synthetic fields**")
            real_only = report["real_features_only_comparison"]["result"]
            st.markdown(f"""
            <div class="reasoning-box">
            Trained on only real (non-synthetic) features, this model still achieves
            <b>F1 = {real_only['f1']}</b> (recall {real_only['recall']:.0%}) — proof the
            classifier learns genuine fraud patterns from real transaction data, not
            just synthetic dispute metadata.
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Feature importances**")
        fi = pd.DataFrame(
            list(report["feature_importances"].items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        st.bar_chart(fi.set_index("Feature"))

    except FileNotFoundError:
        st.error("Run `python src/model.py` first to generate metrics_report.json")

# ---------------------------------------------------------------------
# TAB 3: Audit trail
# ---------------------------------------------------------------------
with tab3:
    st.markdown("**Every decision, in order, fully traceable.**")
    entries = read_audit_trail(limit=50)
    if not entries:
        st.info("No decisions logged yet — run a case in the Live Dispute Review tab.")
    else:
        for e in reversed(entries):
            stamp_color = {"ACCEPT": "🔴", "CONTEST": "🟢", "ESCALATE": "🟡"}.get(e["action"], "⚪")
            with st.expander(f"{stamp_color} {e['timestamp'][:19]} — {e['trans_num'][:14]}… — {e['action']}"):
                st.json(e)