"""
Customer Memory Agent — LangGraph v2

Autonomous customer memory enrichment. Runs in two modes:

1. On-demand fact capture: pass a name_hint + raw_text, agent extracts facts
   and upserts to customer_profiles via the customer-memory bridge.

2. Scheduled background enrichment: nightly cron sweeps every VIP/Premium contact,
   ensures their rich_notes is populated from their latest GHL data
   (enrichment_data from ghl-webhook-receiver metadata).

Architecture follows Doctrine §79 (LangGraph 5-step ritual) + §81 (subject
disambiguation via GHL) + §82 (Paige mirror on every write) + §86 (GHL = comms,
Paige = deals/coaches).

v2 (Jun 28, 2026):
  - Added enrich_from_paige node: pulls deals + coach assignments from Paige Bridge
    using the new Wave 2 verbs (get_opportunities_for_contact, get_coach_for_client)
    and merges them into the customer profile as structured facts. Skips silently
    if PAIGE_BRIDGE_API_KEY isn't set or the profile has no confirmed identity.
"""

from __future__ import annotations

import os
import json
from typing import Annotated, TypedDict, Optional, List

import httpx
from langgraph.graph import StateGraph, START, END

# --- Configuration --------------------------------------------------------

MMA_OS_BRIDGE_URL = os.environ.get(
    "MMA_OS_BRIDGE_URL",
    "https://slcqeiqcrhepicqxqjng.supabase.co/functions/v1/mma-os-bridge",
)
CUSTOMER_MEMORY_URL = os.environ.get(
    "CUSTOMER_MEMORY_URL",
    "https://slcqeiqcrhepicqxqjng.supabase.co/functions/v1/customer-memory",
)
PAIGE_BRIDGE_URL = os.environ.get(
    "PAIGE_BRIDGE_URL",
    "https://bfmyebsjyuoecmjskqhs.supabase.co/functions/v1/paige-bridge",
)
MMA_OS_BRIDGE_API_KEY = os.environ.get("MMA_OS_BRIDGE_API_KEY", "")
PAIGE_BRIDGE_API_KEY = os.environ.get("PAIGE_BRIDGE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20251022")


# --- State ----------------------------------------------------------------


class MemoryState(TypedDict, total=False):
    mode: str
    name_hint: Optional[str]
    email: Optional[str]
    raw_text: Optional[str]
    source: str
    source_meta: dict
    captured_by: str
    identity: dict
    extracted_facts: List[dict]
    structured: dict
    capture_result: dict
    paige_context: dict
    enrichment_summary: dict
    error: Optional[str]


# --- HTTP helpers ---------------------------------------------------------


def _post(url: str, body: dict, bearer: str, timeout: float = 30.0) -> dict:
    if not bearer:
        return {"ok": False, "error": "missing bearer token"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {bearer}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            try:
                return resp.json()
            except Exception:
                return {"ok": False, "error": f"non-json response: {resp.text[:200]}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _bridge_post(url: str, body: dict, timeout: float = 30.0) -> dict:
    return _post(url, body, MMA_OS_BRIDGE_API_KEY, timeout)


def _paige_post(body: dict, timeout: float = 30.0) -> dict:
    return _post(PAIGE_BRIDGE_URL, body, PAIGE_BRIDGE_API_KEY, timeout)


def _claude_extract_facts(raw_text: str, name_hint: Optional[str] = None) -> List[dict]:
    if not ANTHROPIC_API_KEY or not raw_text:
        return [{"fact_text": raw_text[:500], "category": "raw"}]

    system_prompt = (
        "You extract structured personal facts about people from free-form text. "
        "Return ONLY a JSON object: { \"facts\": [{ \"fact_text\": str, \"category\": str }] } "
        "where category is one of: family, business, employment, location, personal, "
        "health, communication, goal, milestone, preference, general. "
        "Each fact_text should be a complete sentence written in third person about the subject. "
        "Skip anything that's not a personal/contextual fact (e.g., transactional 'I want a refund' is NOT a fact). "
        "Be liberal with extraction — capture every meaningful detail."
    )
    user_msg = f"Subject: {name_hint or 'unknown'}\n\nText to extract from:\n{raw_text}"

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 1500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_msg}],
                },
            )
            data = resp.json()
            content_blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            parsed = json.loads(text)
            return parsed.get("facts", [])
    except Exception as exc:
        return [{"fact_text": raw_text[:500], "category": "raw", "extraction_error": str(exc)}]


# --- Graph nodes ----------------------------------------------------------


def resolve_identity(state: MemoryState) -> MemoryState:
    if state.get("email"):
        return {**state, "identity": {"match_status": "confirmed", "suggested_email": state["email"]}}

    name_hint = (state.get("name_hint") or "").strip()
    if not name_hint:
        return {**state, "error": "no email or name_hint provided", "identity": {"match_status": "no_input"}}

    result = _bridge_post(CUSTOMER_MEMORY_URL, {"verb": "resolve_identity", "name_hint": name_hint})
    if not result.get("ok"):
        return {**state, "error": f"resolve_identity failed: {result.get('error')}", "identity": result}
    return {**state, "identity": result}


def extract_facts(state: MemoryState) -> MemoryState:
    if state.get("error"):
        return state
    raw_text = state.get("raw_text", "")
    if not raw_text:
        return {**state, "extracted_facts": []}
    facts = _claude_extract_facts(raw_text, state.get("name_hint"))
    return {**state, "extracted_facts": facts}


def capture_to_bridge(state: MemoryState) -> MemoryState:
    if state.get("error"):
        return state

    identity = state.get("identity", {})
    if identity.get("match_status") not in ("confirmed", "multiple_matches"):
        return {**state, "capture_result": {"ok": False, "error": "identity not confirmed; skipping write"}}

    body = {
        "verb": "capture",
        "facts": state.get("extracted_facts", []),
        "source": state.get("source", "langgraph_customer_memory"),
        "source_meta": state.get("source_meta", {}),
        "raw_excerpt": (state.get("raw_text") or "")[:500],
        "captured_by": state.get("captured_by", "customer_memory_agent"),
        "structured": state.get("structured", {}),
    }
    if state.get("email"):
        body["email"] = state["email"]
    elif identity.get("suggested_email"):
        body["email"] = identity["suggested_email"]
    else:
        body["name_hint"] = state.get("name_hint")

    result = _bridge_post(CUSTOMER_MEMORY_URL, body)
    return {**state, "capture_result": result}


def enrich_from_paige(state: MemoryState) -> MemoryState:
    """Pull coach assignments + deals from Paige (Wave 2 verbs) and merge as
    additional facts. Skips silently if not configured. Doctrine §86."""
    capture = state.get("capture_result", {})
    if not capture.get("ok") or not PAIGE_BRIDGE_API_KEY:
        return {**state, "paige_context": {"ok": False, "skipped": "no capture or no Paige bridge key"}}

    profile = capture.get("profile") or {}
    email = profile.get("email")
    if not email:
        return {**state, "paige_context": {"ok": False, "skipped": "no email on profile"}}

    coach_res = _paige_post({"verb": "get_coach_for_client", "payload": {"email": email}}, timeout=15.0)
    deals_res = _paige_post({"verb": "get_opportunities_for_contact", "payload": {"email": email, "limit": 25}}, timeout=15.0)

    paige_facts: List[dict] = []

    assignments = (coach_res.get("data") or coach_res).get("assignments", []) if isinstance(coach_res, dict) else []
    for a in assignments or []:
        role = a.get("role")
        email_a = a.get("email")
        if role and email_a:
            paige_facts.append({
                "fact_text": f"Assigned to {email_a} as {role} (Paige).",
                "category": "paige_assignment",
            })

    deals = (deals_res.get("data") or deals_res).get("deals", []) if isinstance(deals_res, dict) else []
    for d in deals or []:
        title = d.get("title") or "Untitled deal"
        stage = d.get("stage") or "unknown stage"
        pipeline = d.get("pipeline") or "unknown pipeline"
        status = d.get("status") or "open"
        value_cents = d.get("value_cents")
        amount_str = f" — ${value_cents/100:,.0f}" if isinstance(value_cents, (int, float)) else ""
        paige_facts.append({
            "fact_text": f"Deal '{title}' in {pipeline} → {stage} ({status}){amount_str}.",
            "category": "paige_deal",
        })

    if not paige_facts:
        return {**state, "paige_context": {
            "ok": True,
            "coach_assignments": 0,
            "deals": 0,
            "facts_added": 0,
            "note": "no coach/deal data in Paige for this contact"
        }}

    follow_up = _bridge_post(CUSTOMER_MEMORY_URL, {
        "verb": "capture",
        "email": email,
        "facts": paige_facts,
        "source": "paige_enrichment",
        "source_meta": {
            "via_agent": "customer_memory",
            "coach_count": len([f for f in paige_facts if f["category"] == "paige_assignment"]),
            "deal_count": len([f for f in paige_facts if f["category"] == "paige_deal"]),
        },
        "raw_excerpt": "Pulled from Paige Bridge get_coach_for_client + get_opportunities_for_contact",
        "captured_by": "customer_memory_agent_enrich_from_paige",
    })

    return {**state, "paige_context": {
        "ok": follow_up.get("ok", False),
        "coach_assignments": len([f for f in paige_facts if f["category"] == "paige_assignment"]),
        "deals": len([f for f in paige_facts if f["category"] == "paige_deal"]),
        "facts_added": follow_up.get("facts_added", 0),
        "follow_up_result": follow_up,
    }}


def notify_admin(state: MemoryState) -> MemoryState:
    capture = state.get("capture_result", {})
    paige = state.get("paige_context", {})
    if not capture.get("ok"):
        msg = f"🧠 Customer Memory: capture FAILED — {capture.get('error', 'unknown')}"
    else:
        profile = capture.get("profile", {})
        facts_added = capture.get("facts_added", 0)
        facts_total = capture.get("facts_total", 0)
        paige_bits = ""
        if paige.get("ok") and paige.get("facts_added"):
            paige_bits = f"\n• +{paige['facts_added']} Paige facts (deals: {paige.get('deals',0)}, coaches: {paige.get('coach_assignments',0)})"
        msg = (
            f"🧠 Customer Memory updated\n"
            f"• {profile.get('full_name', profile.get('email', 'unknown'))}\n"
            f"• +{facts_added} facts (total: {facts_total})\n"
            f"• action: {capture.get('action', 'updated')}{paige_bits}\n"
            f"• Paige mirror: {'✅' if capture.get('paige_mirror', {}).get('ok') else '⚠️'}"
        )

    _bridge_post(MMA_OS_BRIDGE_URL, {
        "verb": "push_admin_notification",
        "category": "customer_memory",
        "severity": "info",
        "message": msg,
        "metadata": {"capture_result": capture, "paige_context": paige},
    })
    return {**state, "enrichment_summary": {"message_sent": msg, "ok": capture.get("ok", False)}}


# --- Graph wiring ---------------------------------------------------------


def build_graph() -> StateGraph:
    g = StateGraph(MemoryState)
    g.add_node("resolve_identity", resolve_identity)
    g.add_node("extract_facts", extract_facts)
    g.add_node("capture_to_bridge", capture_to_bridge)
    g.add_node("enrich_from_paige", enrich_from_paige)
    g.add_node("notify_admin", notify_admin)

    g.add_edge(START, "resolve_identity")
    g.add_edge("resolve_identity", "extract_facts")
    g.add_edge("extract_facts", "capture_to_bridge")
    g.add_edge("capture_to_bridge", "enrich_from_paige")
    g.add_edge("enrich_from_paige", "notify_admin")
    g.add_edge("notify_admin", END)
    return g.compile()


graph = build_graph()
