"""
Performance Analytics Module (Task 8)
Aggregates session data from session_store.json and returns dashboard metrics.
"""
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

SESSION_STORE_PATH = Path(__file__).parent.parent.parent / "data" / "session_store.json"


def _load_sessions() -> List[Dict]:
    """Load all saved session records from disk."""
    if not SESSION_STORE_PATH.exists():
        return []
    try:
        with open(SESSION_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[Analytics] Failed to load sessions: {exc}")
        return []


def _save_sessions(sessions: List[Dict]) -> None:
    """Persist session records to disk."""
    SESSION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(SESSION_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[Analytics] Failed to save sessions: {exc}")


def save_session(report: Dict) -> Dict:
    """
    Extract key metrics from a PostInteractionReport and persist as a SessionRecord.
    Returns the saved SessionRecord dict.
    """
    quality = report.get("resolution_quality") or {}
    if isinstance(quality, dict):
        quality_total = float(quality.get("total", 0.0))
    else:
        quality_total = 0.0

    record = {
        "session_id": report.get("session_id", ""),
        "timestamp": report.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "scenario": report.get("dominant_intent", "general_inquiry").replace("_", " ").title(),
        "total_turns": report.get("total_turns", 0),
        "resolution_status": report.get("resolution_status", "unresolved"),
        "resolution_quality_total": quality_total,
        "max_frustration": report.get("max_frustration", 0),
        "final_sentiment": report.get("final_sentiment", "neutral"),
        "final_escalation_risk": report.get("final_escalation_risk", "low"),
        "dominant_intent": report.get("dominant_intent", "general_inquiry"),
        "coaching_recommendations": report.get("coaching_recommendations", []),
        "agent_strengths": report.get("agent_strengths", []),
        "agent_weaknesses": report.get("agent_weaknesses", []),
    }

    sessions = _load_sessions()
    # Avoid duplicate session_ids
    sessions = [s for s in sessions if s.get("session_id") != record["session_id"]]
    sessions.append(record)
    _save_sessions(sessions)
    return record


def get_dashboard() -> Dict:
    """
    Aggregate all saved sessions and compute analytics dashboard metrics.
    Returns an AnalyticsDashboard-compatible dict.
    """
    sessions = _load_sessions()
    if not sessions:
        return {
            "total_sessions": 0,
            "resolved_count": 0,
            "partially_resolved_count": 0,
            "unresolved_count": 0,
            "escalated_count": 0,
            "avg_resolution_quality": 0.0,
            "avg_frustration": 0.0,
            "resolution_trend": [],
            "escalation_frequency": {},
            "common_intents": {},
            "knowledge_gaps": [],
            "improvement_indicators": {},
            "top_coaching_recommendations": [],
            "sessions": [],
        }

    total = len(sessions)
    resolved = sum(1 for s in sessions if s.get("resolution_status") == "resolved")
    partial = sum(1 for s in sessions if s.get("resolution_status") == "partially_resolved")
    unresolved = sum(1 for s in sessions if s.get("resolution_status") == "unresolved")
    escalated = sum(1 for s in sessions if s.get("resolution_status") == "escalated")

    quality_scores = [s.get("resolution_quality_total", 0.0) for s in sessions]
    avg_quality = round(sum(quality_scores) / max(1, len(quality_scores)), 1)

    frustrations = [s.get("max_frustration", 0) for s in sessions]
    avg_frustration = round(sum(frustrations) / max(1, len(frustrations)), 1)

    # Resolution quality trend (per session, ordered by timestamp)
    sorted_sessions = sorted(sessions, key=lambda x: x.get("timestamp", ""))
    resolution_trend = [
        {"session_id": s.get("session_id"), "quality": s.get("resolution_quality_total", 0),
         "timestamp": s.get("timestamp", "")[:10]}
        for s in sorted_sessions
    ]

    # Escalation frequency
    escalation_counts = Counter(s.get("final_escalation_risk", "low") for s in sessions)

    # Common intents
    intent_counts = Counter(s.get("dominant_intent", "general_inquiry") for s in sessions)
    common_intents = dict(intent_counts.most_common(8))

    # Knowledge gaps: intents with avg quality < 60
    intent_quality_map: Dict[str, List[float]] = {}
    for s in sessions:
        intent = s.get("dominant_intent", "general_inquiry")
        q = s.get("resolution_quality_total", 50.0)
        intent_quality_map.setdefault(intent, []).append(q)
    knowledge_gaps = [
        intent.replace("_", " ").title()
        for intent, scores in intent_quality_map.items()
        if (sum(scores) / len(scores)) < 60 and len(scores) >= 1
    ]

    # Improvement indicators: compare first half vs second half of sessions
    half = max(1, total // 2)
    first_half = sorted_sessions[:half]
    second_half = sorted_sessions[half:]
    first_avg = sum(s.get("resolution_quality_total", 0) for s in first_half) / max(1, len(first_half))
    second_avg = sum(s.get("resolution_quality_total", 0) for s in second_half) / max(1, len(second_half))
    first_frust = sum(s.get("max_frustration", 0) for s in first_half) / max(1, len(first_half))
    second_frust = sum(s.get("max_frustration", 0) for s in second_half) / max(1, len(second_half))

    improvement_indicators = {
        "quality_trend": "improving" if second_avg > first_avg + 2 else ("declining" if second_avg < first_avg - 2 else "stable"),
        "frustration_trend": "improving" if second_frust < first_frust - 0.5 else ("declining" if second_frust > first_frust + 0.5 else "stable"),
        "resolution_rate": round(resolved / max(1, total) * 100, 1),
        "escalation_rate": round(escalated / max(1, total) * 100, 1),
        "first_half_avg_quality": round(first_avg, 1),
        "second_half_avg_quality": round(second_avg, 1),
    }

    # Top coaching recommendations (most frequently appearing)
    all_coaching = []
    for s in sessions:
        all_coaching.extend(s.get("coaching_recommendations", []))
    top_coaching = [tip for tip, _ in Counter(all_coaching).most_common(5)]

    return {
        "total_sessions": total,
        "resolved_count": resolved,
        "partially_resolved_count": partial,
        "unresolved_count": unresolved,
        "escalated_count": escalated,
        "avg_resolution_quality": avg_quality,
        "avg_frustration": avg_frustration,
        "resolution_trend": resolution_trend,
        "escalation_frequency": dict(escalation_counts),
        "common_intents": common_intents,
        "knowledge_gaps": knowledge_gaps,
        "improvement_indicators": improvement_indicators,
        "top_coaching_recommendations": top_coaching,
        "sessions": sorted_sessions,
    }


def clear_all_sessions() -> int:
    """Clear all saved sessions. Returns the count that was deleted."""
    sessions = _load_sessions()
    count = len(sessions)
    _save_sessions([])
    return count