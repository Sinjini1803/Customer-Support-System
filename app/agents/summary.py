import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")

_client = None
_hf_quota_depleted = False

if HF_TOKEN:
    try:
        from huggingface_hub import InferenceClient
        _client = InferenceClient(api_key=HF_TOKEN, provider="auto")
    except Exception as exc:
        print(f"[Summary Agent] InferenceClient init note: {exc}")


def _extract_content(response) -> str:
    if not response:
        return ""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if not message:
        return ""
    content = getattr(message, "content", None)
    return str(content).strip() if content else ""


def _safe_json(text: str) -> Optional[dict]:
    text = text.strip()
    for pattern in [r"```json(.*?)```", r"```(.*?)```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break
    try:
        return json.loads(text)
    except Exception:
        return None


def _build_sentiment_journey(conversation: List[Dict]) -> List[Dict]:
    journey = []
    turn_num = 0
    for msg in conversation:
        turn_num += 1
        role = msg.get("role", "user")
        content = msg.get("content", "")
        analysis = msg.get("analysis") or {}
        journey.append({
            "turn": turn_num,
            "role": role,
            "content_snippet": content[:80] + ("..." if len(content) > 80 else ""),
            "emotion": analysis.get("emotion", "neutral") if role == "user" else "-",
            "frustration": analysis.get("frustration_level", 0) if role == "user" else 0,
            "sentiment": analysis.get("sentiment", "neutral") if role == "user" else "-",
            "escalation_risk": analysis.get("escalation_risk", "low") if role == "user" else "-",
        })
    return journey


def _heuristic_quality_score(conversation: List[Dict]) -> Dict:
    customer_msgs = [m for m in conversation if m.get("role") == "user" and m.get("analysis")]
    if not customer_msgs:
        return {"total": 50.0, "issue_resolution": 50.0, "communication_quality": 50.0,
                "empathy_shown": 50.0, "guideline_adherence": 50.0}

    final_analysis = customer_msgs[-1].get("analysis", {})
    final_frustration = final_analysis.get("frustration_level", 5)
    final_trend = final_analysis.get("satisfaction_trend", "stable")
    final_risk = final_analysis.get("escalation_risk", "low")

    rs = 70.0
    if final_trend == "improving":
        rs += 20.0
    elif final_trend == "declining":
        rs -= 20.0
    if final_frustration <= 3:
        rs += 10.0
    elif final_frustration >= 7:
        rs -= 15.0
    if final_risk in ("high", "critical"):
        rs -= 10.0
    rs = max(0.0, min(100.0, rs))

    evals = [m["analysis"].get("response_evaluation", {}) for m in customer_msgs if m.get("analysis")]
    tone_scores = [e.get("tone", 3) for e in evals if isinstance(e, dict)]
    clarity_scores = [e.get("clarity", 3) for e in evals if isinstance(e, dict)]
    cq = ((sum(tone_scores) / max(1, len(tone_scores))) + (sum(clarity_scores) / max(1, len(clarity_scores)))) / 2 * 20
    cq = max(0.0, min(100.0, cq))

    empathy_scores = [e.get("empathy", 3) for e in evals if isinstance(e, dict)]
    em = (sum(empathy_scores) / max(1, len(empathy_scores))) * 20
    em = max(0.0, min(100.0, em))

    prof_scores = [e.get("professionalism", 3) for e in evals if isinstance(e, dict)]
    ad = (sum(prof_scores) / max(1, len(prof_scores))) * 20
    ad = max(0.0, min(100.0, ad))

    total = rs * 0.40 + cq * 0.20 + em * 0.20 + ad * 0.20
    return {
        "total": round(total, 1),
        "issue_resolution": round(rs, 1),
        "communication_quality": round(cq, 1),
        "empathy_shown": round(em, 1),
        "guideline_adherence": round(ad, 1),
    }


def _heuristic_report(conversation: List[Dict], session_id: str) -> Dict:
    customer_msgs = [m for m in conversation if m.get("role") == "user"]
    agent_msgs = [m for m in conversation if m.get("role") == "assistant"]
    analyzed_msgs = [m for m in customer_msgs if m.get("analysis")]

    intents = [m["analysis"].get("intent", "general_inquiry") for m in analyzed_msgs]
    dominant_intent = Counter(intents).most_common(1)[0][0] if intents else "general_inquiry"
    frustrations = [m["analysis"].get("frustration_level", 0) for m in analyzed_msgs]
    max_frustration = max(frustrations) if frustrations else 0

    final_analysis = analyzed_msgs[-1]["analysis"] if analyzed_msgs else {}
    final_sentiment = final_analysis.get("sentiment", "neutral")
    final_risk = final_analysis.get("escalation_risk", "low")
    final_trend = final_analysis.get("satisfaction_trend", "stable")

    if final_trend == "improving" and final_analysis.get("frustration_level", 5) < 4:
        resolution_status = "resolved"
    elif final_risk in ("high", "critical"):
        resolution_status = "escalated"
    elif final_trend == "stable":
        resolution_status = "partially_resolved"
    else:
        resolution_status = "unresolved"

    intent_label = dominant_intent.replace("_", " ")
    summary = (
        f"The customer contacted support regarding a {intent_label} issue. "
        f"The conversation spanned {len(customer_msgs)} customer turns and {len(agent_msgs)} agent responses. "
        f"The customer's frustration peaked at {max_frustration}/10 during the interaction. "
        f"The final sentiment was {final_sentiment} with a {final_trend} satisfaction trend. "
        f"The overall escalation risk concluded as {final_risk}."
    )

    quality = _heuristic_quality_score(conversation)
    strengths, weaknesses = [], []
    if quality["empathy_shown"] >= 60:
        strengths.append("Demonstrated consistent empathy throughout the conversation")
    else:
        weaknesses.append("Empathy levels were below expectations - use more acknowledging language")
    if quality["communication_quality"] >= 60:
        strengths.append("Maintained clear and coherent communication")
    else:
        weaknesses.append("Communication clarity could be improved - structure responses with numbered steps")
    if quality["issue_resolution"] >= 65:
        strengths.append("Successfully guided the conversation toward resolution")
    else:
        weaknesses.append("Issue resolution was incomplete - follow up with actionable next steps")
    if quality["guideline_adherence"] >= 60:
        strengths.append("Maintained professional tone and followed support guidelines")
    else:
        weaknesses.append("Adherence to professional guidelines needs improvement")
    if final_trend == "improving":
        strengths.append("Customer satisfaction was progressively improving by end of conversation")

    coaching = []
    if max_frustration >= 7:
        coaching.append("Practice de-escalation techniques - acknowledge frustration explicitly before offering solutions")
    if quality["empathy_shown"] < 60:
        coaching.append("Use empathy statements like 'I completely understand how frustrating this must be' at the start of each response")
    if quality["communication_quality"] < 60:
        coaching.append("Structure responses with numbered steps to improve clarity, especially for complex issues")
    if quality["issue_resolution"] < 60:
        coaching.append("Always confirm resolution explicitly by asking 'Does this fully resolve your concern?' before closing")
    if final_risk in ("high", "critical"):
        coaching.append("For high-risk interactions, offer supervisor escalation proactively before the customer demands it")
    if not coaching:
        coaching.append("Continue your strong performance - maintain the empathy and clarity shown in this session")

    return {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_turns": len(conversation),
        "customer_turns": len(customer_msgs),
        "agent_turns": len(agent_msgs),
        "conversation_summary": summary,
        "primary_issue": f"Customer reported a {intent_label} issue requiring agent assistance",
        "final_resolution": f"Conversation ended with {resolution_status.replace('_', ' ')} status. Final sentiment: {final_sentiment}.",
        "resolution_status": resolution_status,
        "sentiment_journey": _build_sentiment_journey(conversation),
        "resolution_quality": quality,
        "agent_strengths": strengths,
        "agent_weaknesses": weaknesses,
        "coaching_recommendations": coaching,
        "max_frustration": max_frustration,
        "final_sentiment": final_sentiment,
        "final_escalation_risk": final_risk,
        "dominant_intent": dominant_intent,
    }


def _llm_report(conversation: List[Dict], session_id: str) -> Optional[Dict]:
    global _hf_quota_depleted
    if not _client or _hf_quota_depleted:
        return None

    transcript_lines = []
    for i, msg in enumerate(conversation, 1):
        role_label = "Customer" if msg.get("role") == "user" else "Agent"
        transcript_lines.append(f"[{i}] {role_label}: {msg.get('content', '')[:200]}")
    transcript = "\n".join(transcript_lines)

    analyzed = [m for m in conversation if m.get("role") == "user" and m.get("analysis")]
    analysis_context = ""
    if analyzed:
        last_a = analyzed[-1]["analysis"]
        analysis_context = (
            f"Intent:{last_a.get('intent')}, Emotion:{last_a.get('emotion')}, "
            f"Frustration:{last_a.get('frustration_level')}/10, "
            f"Trend:{last_a.get('satisfaction_trend')}, Risk:{last_a.get('escalation_risk')}"
        )

    system_prompt = (
        'You are an expert Post-Interaction Summary Agent for a customer support system. '
        'Analyze the support conversation and return a JSON object with EXACTLY these keys: '
        '"conversation_summary" (3-5 sentences), "primary_issue", "final_resolution", '
        '"resolution_status" (one of: resolved/partially_resolved/unresolved/escalated), '
        '"agent_strengths" (list of strings, max 4), "agent_weaknesses" (list, max 4), '
        '"coaching_recommendations" (list, max 4). Return ONLY raw JSON, no markdown.'
    )
    try:
        response = _client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nCONTEXT:{analysis_context}\n\nGenerate JSON:"},
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        content = _extract_content(response)
        data = _safe_json(content)
        if not data:
            return None
        heuristic = _heuristic_report(conversation, session_id)
        for key in ("conversation_summary", "primary_issue", "final_resolution",
                    "resolution_status", "agent_strengths", "agent_weaknesses", "coaching_recommendations"):
            if data.get(key):
                heuristic[key] = data[key]
        return heuristic
    except Exception as exc:
        err_str = str(exc)
        if "402" in err_str or "quota" in err_str.lower() or "credits" in err_str.lower():
            _hf_quota_depleted = True
        print(f"[Summary Agent] LLM error: {exc}. Falling back to heuristics.")
        return None


def generate_post_interaction_report(
    conversation: List[Dict],
    session_id: Optional[str] = None,
) -> Dict:
    """Main entry point. Returns a PostInteractionReport-compatible dict."""
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    if not conversation:
        return {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_turns": 0, "customer_turns": 0, "agent_turns": 0,
            "conversation_summary": "No conversation data provided.",
            "primary_issue": "N/A", "final_resolution": "N/A",
            "resolution_status": "unresolved", "sentiment_journey": [],
            "resolution_quality": {"total": 0.0, "issue_resolution": 0.0,
                                   "communication_quality": 0.0, "empathy_shown": 0.0,
                                   "guideline_adherence": 0.0},
            "agent_strengths": [], "agent_weaknesses": [],
            "coaching_recommendations": [],
            "max_frustration": 0, "final_sentiment": "neutral",
            "final_escalation_risk": "low", "dominant_intent": "general_inquiry",
        }
    return _llm_report(conversation, session_id) or _heuristic_report(conversation, session_id)