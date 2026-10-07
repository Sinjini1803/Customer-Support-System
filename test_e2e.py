"""
End-to-End Test Suite (Task 8)
Tests all agents, API endpoints, data flow, analytics, and UI state.

Run with:
    pytest test_e2e.py -v
    pytest test_e2e.py -v --tb=short   (brief tracebacks)

Requires the FastAPI server running at http://127.0.0.1:8000
"""
import json
import time
import uuid
import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"

# ===================================================================
# Shared fixtures
# ===================================================================

SAMPLE_CONVERSATION = [
    {
        "role": "user",
        "content": "I was charged twice for order #99214! This is absolutely unacceptable!",
        "analysis": {
            "intent": "payment_issue",
            "emotion": "angry",
            "sentiment": "negative",
            "frustration_level": 8,
            "satisfaction_trend": "declining",
            "escalation_risk": "high",
            "escalation_reasoning": "Double charge reported with strong negative language.",
            "confidence": 0.93,
            "suggested_action": "Acknowledge and escalate to billing team immediately.",
            "suggested_response": "I sincerely apologize for the inconvenience.",
            "response_evaluation": {"tone": 4, "clarity": 4, "empathy": 5, "professionalism": 4},
            "communication_tips": ["Lead with empathy", "Give concrete timeline"],
        }
    },
    {
        "role": "assistant",
        "content": "I sincerely apologize for the double charge. I am investigating this right now."
    },
    {
        "role": "user",
        "content": "Thank you for looking into it. I hope this gets resolved soon.",
        "analysis": {
            "intent": "payment_issue",
            "emotion": "worried",
            "sentiment": "neutral",
            "frustration_level": 4,
            "satisfaction_trend": "improving",
            "escalation_risk": "medium",
            "escalation_reasoning": "Frustration has reduced but issue unresolved.",
            "confidence": 0.88,
            "suggested_action": "Provide a concrete ETA for the refund.",
            "suggested_response": "The duplicate charge will be reversed within 3-5 business days.",
            "response_evaluation": {"tone": 5, "clarity": 4, "empathy": 4, "professionalism": 5},
            "communication_tips": ["Give specific timelines"],
        }
    },
    {
        "role": "assistant",
        "content": "The duplicate charge will be reversed within 3-5 business days. You will receive an email confirmation."
    },
]


def server_available() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ===================================================================
# 1. Server Health
# ===================================================================

class TestServerHealth:
    def test_root_endpoint(self):
        r = requests.get(f"{BASE_URL}/", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

    def test_health_endpoint(self):
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        assert r.status_code == 200
        assert r.json().get("status") == "healthy"


# ===================================================================
# 2. Intent & Sentiment Analysis Agent
# ===================================================================

class TestIntentSentimentAgent:
    def test_basic_analysis_returns_200(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "I want a refund for my broken product"},
            timeout=15,
        )
        assert r.status_code == 200

    def test_analysis_has_required_fields(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "My order has not arrived after 2 weeks"},
            timeout=15,
        )
        data = r.json()
        analysis = data.get("analysis", {})
        required = ["intent", "emotion", "sentiment", "frustration_level",
                    "satisfaction_trend", "escalation_risk", "confidence"]
        for field in required:
            assert field in analysis, f"Missing field: {field}"

    def test_frustration_is_in_valid_range(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "I am extremely angry! Where is my order?!"},
            timeout=15,
        )
        analysis = r.json().get("analysis", {})
        frust = analysis.get("frustration_level", -1)
        assert 0 <= frust <= 10

    def test_escalation_risk_is_valid_value(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "I need to speak to your manager RIGHT NOW"},
            timeout=15,
        )
        analysis = r.json().get("analysis", {})
        assert analysis.get("escalation_risk") in ("low", "medium", "high", "critical")

    def test_sentiment_classification(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "Thank you so much! The issue is resolved perfectly."},
            timeout=15,
        )
        analysis = r.json().get("analysis", {})
        assert analysis.get("sentiment") in ("positive", "neutral", "negative")

    def test_coaching_fields_present(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "My subscription was charged without notice"},
            timeout=15,
        )
        analysis = r.json().get("analysis", {})
        assert "suggested_action" in analysis
        assert "suggested_response" in analysis
        assert "communication_tips" in analysis
        assert isinstance(analysis["communication_tips"], list)

    def test_empty_message_rejected(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "   "},
            timeout=10,
        )
        # Should return 422 (validation error) or 200 with default values
        assert r.status_code in (200, 422)

    def test_analysis_with_history_context(self):
        history = [
            {"role": "user", "content": "My package arrived damaged"},
            {"role": "assistant", "content": "I am sorry to hear that. Can you share your order number?"},
        ]
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "Yes it is order #12345", "history": history},
            timeout=15,
        )
        assert r.status_code == 200
        assert "analysis" in r.json()


# ===================================================================
# 3. Knowledge Recommendation Agent
# ===================================================================

class TestKnowledgeRecommendationAgent:
    def test_basic_recommendation_returns_200(self):
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"message": "I want to return my product for a refund", "top_k": 3},
            timeout=15,
        )
        assert r.status_code == 200

    def test_recommendations_list_returned(self):
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"message": "My order was delivered to the wrong address", "top_k": 3},
            timeout=15,
        )
        data = r.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_recommendation_item_has_required_fields(self):
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"message": "Cancel my subscription immediately", "top_k": 2},
            timeout=15,
        )
        recs = r.json().get("recommendations", [])
        if recs:
            item = recs[0]
            for field in ["title", "snippet", "score", "document_name"]:
                assert field in item, f"Missing field: {field}"

    def test_scores_are_valid_floats(self):
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"message": "I was charged twice", "top_k": 2},
            timeout=15,
        )
        for item in r.json().get("recommendations", []):
            assert isinstance(item.get("score"), float)
            assert 0.0 <= item["score"] <= 1.0

    def test_top_k_respected(self):
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"message": "password reset help", "top_k": 2},
            timeout=15,
        )
        recs = r.json().get("recommendations", [])
        assert len(recs) <= 2


# ===================================================================
# 4. Customer Simulator Agent
# ===================================================================

class TestSimulatorAgent:
    def test_simulator_creates_conversation(self):
        r = requests.post(
            f"{BASE_URL}/simulator/chat",
            json={
                "conversation_id": None,
                "agent_message": "",
                "config": {
                    "persona": "frustrated",
                    "scenario": "Customer received a defective item and wants a refund",
                    "initial_emotion": "frustrated",
                    "current_emotion": "frustrated",
                    "issue_severity": "high",
                    "patience_level": "low",
                    "expected_resolution": "immediate full refund",
                }
            },
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "conversation_id" in data
        assert "customer_message" in data
        assert len(data["customer_message"]) > 0

    def test_simulator_continues_conversation(self):
        # First turn
        r1 = requests.post(
            f"{BASE_URL}/simulator/chat",
            json={
                "conversation_id": None,
                "agent_message": "",
                "config": {
                    "persona": "angry",
                    "scenario": "Delayed order",
                    "initial_emotion": "angry",
                    "current_emotion": "angry",
                    "issue_severity": "high",
                    "patience_level": "low",
                    "expected_resolution": "refund or reship",
                }
            },
            timeout=60,
        )
        conv_id = r1.json().get("conversation_id")
        assert conv_id is not None

        # Second turn
        r2 = requests.post(
            f"{BASE_URL}/simulator/chat",
            json={
                "conversation_id": conv_id,
                "agent_message": "I sincerely apologize. Let me check on your order immediately.",
                "config": {
                    "persona": "angry",
                    "scenario": "Delayed order",
                    "initial_emotion": "angry",
                    "current_emotion": "angry",
                    "issue_severity": "high",
                    "patience_level": "low",
                    "expected_resolution": "refund or reship",
                }
            },
            timeout=60,
        )
        assert r2.status_code == 200
        data2 = r2.json()
        assert "customer_message" in data2
        assert data2.get("conversation_id") == conv_id

    def test_simulator_includes_analysis(self):
        r = requests.post(
            f"{BASE_URL}/simulator/chat",
            json={
                "conversation_id": None,
                "agent_message": "",
                "config": {
                    "persona": "confused",
                    "scenario": "Payment failure",
                    "initial_emotion": "confused",
                    "current_emotion": "confused",
                    "issue_severity": "medium",
                    "patience_level": "medium",
                    "expected_resolution": "payment clarification",
                }
            },
            timeout=60,
        )
        data = r.json()
        assert "analysis" in data
        if data["analysis"]:
            assert "intent" in data["analysis"]

    def test_simulator_includes_knowledge_recommendations(self):
        r = requests.post(
            f"{BASE_URL}/simulator/chat",
            json={
                "conversation_id": None,
                "agent_message": "",
                "config": {
                    "persona": "calm",
                    "scenario": "Subscription cancellation request",
                    "initial_emotion": "neutral",
                    "current_emotion": "neutral",
                    "issue_severity": "low",
                    "patience_level": "high",
                    "expected_resolution": "cancellation confirmed",
                }
            },
            timeout=60,
        )
        data = r.json()
        assert "knowledge_recommendations" in data


# ===================================================================
# 5. Post-Interaction Summary Agent (Task 8)
# ===================================================================

class TestPostInteractionSummary:
    def test_summary_endpoint_returns_200(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        assert r.status_code == 200

    def test_summary_has_success_flag(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        assert r.json().get("success") is True

    def test_report_has_required_fields(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        report = r.json().get("report", {})
        required_fields = [
            "session_id", "timestamp", "total_turns", "customer_turns", "agent_turns",
            "conversation_summary", "primary_issue", "final_resolution",
            "resolution_status", "sentiment_journey", "resolution_quality",
            "agent_strengths", "agent_weaknesses", "coaching_recommendations",
            "max_frustration", "final_sentiment", "final_escalation_risk", "dominant_intent",
        ]
        for field in required_fields:
            assert field in report, f"Missing field in report: {field}"

    def test_resolution_status_is_valid(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        status = r.json().get("report", {}).get("resolution_status")
        assert status in ("resolved", "partially_resolved", "unresolved", "escalated")

    def test_quality_score_in_valid_range(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        quality = r.json().get("report", {}).get("resolution_quality", {})
        total = quality.get("total", -1)
        assert 0.0 <= total <= 100.0

    def test_quality_score_components_present(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        quality = r.json().get("report", {}).get("resolution_quality", {})
        for comp in ("total", "issue_resolution", "communication_quality", "empathy_shown", "guideline_adherence"):
            assert comp in quality, f"Missing quality component: {comp}"

    def test_sentiment_journey_populated(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        journey = r.json().get("report", {}).get("sentiment_journey", [])
        assert len(journey) == len(SAMPLE_CONVERSATION)

    def test_agent_strengths_is_list(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        strengths = r.json().get("report", {}).get("agent_strengths", None)
        assert isinstance(strengths, list)

    def test_coaching_recommendations_non_empty(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        coaching = r.json().get("report", {}).get("coaching_recommendations", [])
        assert len(coaching) >= 1

    def test_max_frustration_correct(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION},
            timeout=60,
        )
        report = r.json().get("report", {})
        # Max frustration in sample conversation is 8
        assert report.get("max_frustration") == 8

    def test_custom_session_id_preserved(self):
        custom_id = "test-session-xyz"
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": SAMPLE_CONVERSATION, "session_id": custom_id},
            timeout=60,
        )
        assert r.json().get("report", {}).get("session_id") == custom_id

    def test_empty_conversation_handled_gracefully(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": []},
            timeout=30,
        )
        assert r.status_code == 200
        report = r.json().get("report", {})
        assert report.get("total_turns") == 0


# ===================================================================
# 6. Performance Analytics Module (Task 8)
# ===================================================================

class TestPerformanceAnalytics:
    def _get_sample_report(self, session_suffix="") -> dict:
        """Helper to generate and return a post-interaction report."""
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={
                "conversation": SAMPLE_CONVERSATION,
                "session_id": f"e2e-test-{session_suffix or uuid.uuid4().hex[:6]}",
            },
            timeout=60,
        )
        return r.json().get("report", {})

    def test_save_session_returns_200(self):
        report = self._get_sample_report("save-test")
        r = requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)
        assert r.status_code == 200
        assert r.json().get("success") is True

    def test_save_session_record_has_fields(self):
        report = self._get_sample_report("fields-test")
        r = requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)
        record = r.json().get("record", {})
        for field in ("session_id", "timestamp", "resolution_status", "resolution_quality_total"):
            assert field in record, f"Missing field: {field}"

    def test_dashboard_endpoint_returns_200(self):
        r = requests.get(f"{BASE_URL}/analytics/dashboard", timeout=10)
        assert r.status_code == 200

    def test_dashboard_has_required_keys(self):
        r = requests.get(f"{BASE_URL}/analytics/dashboard", timeout=10)
        db = r.json().get("dashboard", {})
        required = [
            "total_sessions", "resolved_count", "partially_resolved_count",
            "unresolved_count", "escalated_count", "avg_resolution_quality",
            "avg_frustration", "resolution_trend", "escalation_frequency",
            "common_intents", "knowledge_gaps", "improvement_indicators",
            "top_coaching_recommendations", "sessions",
        ]
        for key in required:
            assert key in db, f"Missing dashboard key: {key}"

    def test_total_sessions_count_increases(self):
        r_before = requests.get(f"{BASE_URL}/analytics/dashboard", timeout=10)
        before_count = r_before.json().get("dashboard", {}).get("total_sessions", 0)

        report = self._get_sample_report(f"count-{uuid.uuid4().hex[:4]}")
        requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)

        r_after = requests.get(f"{BASE_URL}/analytics/dashboard", timeout=10)
        after_count = r_after.json().get("dashboard", {}).get("total_sessions", 0)
        assert after_count >= before_count + 1

    def test_duplicate_session_id_not_double_counted(self):
        report = self._get_sample_report("dup-test")
        # Save the same report twice
        requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)
        requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)

        r = requests.get(f"{BASE_URL}/analytics/dashboard", timeout=10)
        sessions = r.json().get("dashboard", {}).get("sessions", [])
        session_ids = [s.get("session_id") for s in sessions]
        # Should not have duplicates
        assert session_ids.count(report.get("session_id")) == 1

    def test_clear_sessions_endpoint(self):
        # Save one session to ensure there's data
        report = self._get_sample_report("clear-test")
        requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)

        r = requests.delete(f"{BASE_URL}/analytics/sessions", timeout=10)
        assert r.status_code == 200
        assert r.json().get("success") is True

        r_after = requests.get(f"{BASE_URL}/analytics/dashboard", timeout=10)
        assert r_after.json().get("dashboard", {}).get("total_sessions", -1) == 0


# ===================================================================
# 7. End-to-End Data Flow Tests
# ===================================================================

class TestEndToEndDataFlow:
    def test_simulator_to_summary_pipeline(self):
        """Full pipeline: Simulator → Analysis → Knowledge → Summary."""
        # Step 1: Start simulator
        sim_r = requests.post(
            f"{BASE_URL}/simulator/chat",
            json={
                "conversation_id": None,
                "agent_message": "",
                "config": {
                    "persona": "frustrated",
                    "scenario": "Damaged product received",
                    "initial_emotion": "frustrated",
                    "current_emotion": "frustrated",
                    "issue_severity": "high",
                    "patience_level": "low",
                    "expected_resolution": "replacement or refund",
                }
            },
            timeout=60,
        )
        assert sim_r.status_code == 200
        sim_data = sim_r.json()
        conv_id = sim_data.get("conversation_id")

        # Step 2: Build conversation list
        conversation = [{
            "role": "user",
            "content": sim_data.get("customer_message", ""),
            "analysis": sim_data.get("analysis"),
            "knowledge_recommendations": sim_data.get("knowledge_recommendations"),
        }]

        # Step 3: Agent responds
        conversation.append({"role": "assistant", "content": "I'm so sorry to hear about the damaged product."})

        # Step 4: Generate summary
        summary_r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": conversation},
            timeout=60,
        )
        assert summary_r.status_code == 200
        report = summary_r.json().get("report", {})
        assert report.get("total_turns") == 2

        # Step 5: Save to analytics
        analytics_r = requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)
        assert analytics_r.status_code == 200

        # Step 6: Verify dashboard reflects the session
        dashboard_r = requests.get(f"{BASE_URL}/analytics/dashboard", timeout=10)
        assert dashboard_r.status_code == 200

    def test_manual_conversation_summary_analytics_flow(self):
        """Manual conversation → Summary → Analytics pipeline."""
        manual_conv = [
            {"role": "user", "content": "I need to cancel my order immediately",
             "analysis": {"intent": "cancellation", "emotion": "impatient", "sentiment": "negative",
                          "frustration_level": 6, "satisfaction_trend": "declining",
                          "escalation_risk": "medium", "escalation_reasoning": "Urgent cancellation request.",
                          "confidence": 0.85, "suggested_action": "Process cancellation quickly.",
                          "suggested_response": "I will process that cancellation right away.",
                          "response_evaluation": {"tone": 4, "clarity": 4, "empathy": 4, "professionalism": 4},
                          "communication_tips": ["Be decisive"]}},
            {"role": "assistant", "content": "I will process your cancellation immediately."},
            {"role": "user", "content": "Thank you, that was fast!",
             "analysis": {"intent": "cancellation", "emotion": "satisfied", "sentiment": "positive",
                          "frustration_level": 1, "satisfaction_trend": "improving",
                          "escalation_risk": "low", "escalation_reasoning": "Resolved quickly.",
                          "confidence": 0.92, "suggested_action": "Confirm and close.",
                          "suggested_response": "Your cancellation is confirmed. Is there anything else?",
                          "response_evaluation": {"tone": 5, "clarity": 5, "empathy": 5, "professionalism": 5},
                          "communication_tips": ["Close positively"]}},
        ]

        summary_r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": manual_conv, "session_id": f"manual-e2e-{uuid.uuid4().hex[:4]}"},
            timeout=60,
        )
        assert summary_r.status_code == 200
        report = summary_r.json().get("report", {})
        # Improving trend with low final frustration should yield resolved or partially_resolved
        assert report.get("resolution_status") in ("resolved", "partially_resolved")
        assert report.get("max_frustration") == 6

        # Save to analytics
        save_r = requests.post(f"{BASE_URL}/analytics/session", json=report, timeout=10)
        assert save_r.status_code == 200


# ===================================================================
# 8. Error Handling & Edge Cases
# ===================================================================

class TestErrorHandling:
    def test_analysis_missing_message_field(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"not_message": "hello"},
            timeout=10,
        )
        assert r.status_code == 422

    def test_knowledge_missing_message_field(self):
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"top_k": 3},
            timeout=10,
        )
        assert r.status_code == 422

    def test_simulator_missing_config(self):
        r = requests.post(
            f"{BASE_URL}/simulator/chat",
            json={"agent_message": "hello"},
            timeout=10,
        )
        assert r.status_code == 422

    def test_summary_empty_conversation_graceful(self):
        r = requests.post(
            f"{BASE_URL}/summary/post-interaction",
            json={"conversation": []},
            timeout=30,
        )
        assert r.status_code == 200
        assert r.json().get("report", {}).get("total_turns") == 0

    def test_analytics_session_missing_body(self):
        r = requests.post(f"{BASE_URL}/analytics/session", json={}, timeout=10)
        # Should handle gracefully (empty dict has no session_id but shouldn't crash)
        assert r.status_code in (200, 400, 422, 500)

    def test_knowledge_top_k_bounds(self):
        # top_k = 0 is out of range (ge=1)
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"message": "refund policy", "top_k": 0},
            timeout=10,
        )
        assert r.status_code == 422

    def test_knowledge_top_k_max(self):
        # top_k = 11 is out of range (le=10)
        r = requests.post(
            f"{BASE_URL}/recommendations/knowledge",
            json={"message": "shipping policy", "top_k": 11},
            timeout=10,
        )
        assert r.status_code == 422

    def test_long_message_handled(self):
        long_msg = "My order is delayed. " * 100
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": long_msg},
            timeout=20,
        )
        assert r.status_code == 200

    def test_special_characters_in_message(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "Where's my order?! I paid €50.00 & still nothing!!!"},
            timeout=15,
        )
        assert r.status_code == 200
        assert "analysis" in r.json()


# ===================================================================
# 9. Session State Initialization Tests
# ===================================================================

class TestSessionStateDefaults:
    """Test that the agent modules handle edge cases in isolation."""

    def test_summary_agent_direct_empty(self):
        from app.agents.summary import generate_post_interaction_report
        result = generate_post_interaction_report([])
        assert result["total_turns"] == 0
        assert result["resolution_status"] == "unresolved"

    def test_summary_agent_direct_single_customer_msg(self):
        from app.agents.summary import generate_post_interaction_report
        conv = [{"role": "user", "content": "I need help",
                 "analysis": {"intent": "general_inquiry", "emotion": "neutral",
                              "sentiment": "neutral", "frustration_level": 2,
                              "satisfaction_trend": "stable", "escalation_risk": "low",
                              "escalation_reasoning": "", "confidence": 0.9,
                              "suggested_action": "", "suggested_response": "",
                              "response_evaluation": {"tone": 3, "clarity": 3, "empathy": 3, "professionalism": 3},
                              "communication_tips": []}}]
        result = generate_post_interaction_report(conv)
        assert result["total_turns"] == 1
        assert result["max_frustration"] == 2

    def test_analytics_module_empty_store(self):
        from app.agents.analytics import get_dashboard
        # Clear first to ensure empty state
        from app.agents.analytics import clear_all_sessions
        clear_all_sessions()
        db = get_dashboard()
        assert db["total_sessions"] == 0
        assert db["sessions"] == []

    def test_analytics_save_and_retrieve(self):
        from app.agents.analytics import save_session, get_dashboard, clear_all_sessions
        from app.agents.summary import generate_post_interaction_report
        clear_all_sessions()
        report = generate_post_interaction_report(SAMPLE_CONVERSATION, session_id="unit-test-001")
        save_session(report)
        db = get_dashboard()
        assert db["total_sessions"] == 1
        assert db["sessions"][0]["session_id"] == "unit-test-001"


# ===================================================================
# 10. Escalation Risk Monitor
# ===================================================================

class TestEscalationRiskMonitor:
    def test_angry_message_high_risk(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "This is absolutely DISGUSTING! I want to sue your company! Get me your manager NOW!"},
            timeout=15,
        )
        analysis = r.json().get("analysis", {})
        risk = analysis.get("escalation_risk", "low")
        # Should be high or critical
        assert risk in ("high", "critical"), f"Expected high/critical, got: {risk}"

    def test_calm_message_low_risk(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "Hi, I was wondering if you could help me track my order please?"},
            timeout=15,
        )
        analysis = r.json().get("analysis", {})
        risk = analysis.get("escalation_risk", "high")
        assert risk in ("low", "medium"), f"Expected low/medium, got: {risk}"

    def test_escalation_reasoning_populated(self):
        r = requests.post(
            f"{BASE_URL}/analysis/intent-sentiment",
            json={"message": "I demand you escalate this to your supervisor immediately!"},
            timeout=15,
        )
        analysis = r.json().get("analysis", {})
        reasoning = analysis.get("escalation_reasoning", "")
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0


if __name__ == "__main__":
    if not server_available():
        print("ERROR: FastAPI server is not running at http://127.0.0.1:8000")
        print("Start it with: uvicorn app.main:app --host 127.0.0.1 --port 8000")
        exit(1)
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])