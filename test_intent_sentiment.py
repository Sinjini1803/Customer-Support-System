"""
Unit Test Suite for Task 4: Intent and Sentiment Analysis Agent
Covers 22 comprehensive customer scenarios testing:
- Intent detection
- Emotion classification
- Sentiment analysis
- Frustration scoring (0-10)
- Satisfaction trend (improving, declining, stable)
- Escalation risk (low, medium, high)
- API endpoint integration
"""

import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.agents.intent_sentiment import analyze_customer_message

client = TestClient(app)

class TestIntentSentimentAgent(unittest.TestCase):

    def test_scenario_01_angry_refund_high_escalation(self):
        msg = "I am LIVID! This product broke within two hours of arrival. Give me an immediate refund now or I'm taking this to a lawyer!"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "refund")
        self.assertIn(res.emotion, ["angry", "frustrated"])
        self.assertEqual(res.sentiment, "negative")
        self.assertGreaterEqual(res.frustration_level, 8)
        self.assertEqual(res.escalation_risk, "high")

    def test_scenario_02_polite_refund_request(self):
        msg = "Hello, could you please guide me on how to request a refund for order #1234? It was not quite what I expected, thank you."
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "refund")
        self.assertEqual(res.sentiment, "neutral")
        self.assertLessEqual(res.frustration_level, 3)
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_03_calm_cancellation(self):
        msg = "Hi, I accidentally made a duplicate purchase and would like to cancel order #9876."
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "cancellation")
        self.assertEqual(res.sentiment, "neutral")
        self.assertLessEqual(res.frustration_level, 3)
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_04_angry_cancellation_complaint(self):
        msg = "Cancel my subscription immediately! Your service is completely terrible and unacceptable."
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "cancellation")
        self.assertEqual(res.sentiment, "negative")
        self.assertGreaterEqual(res.frustration_level, 5)
        self.assertIn(res.escalation_risk, ["medium", "high"])

    def test_scenario_05_urgent_delayed_delivery(self):
        msg = "My package has been delayed for 5 days and I needed it for my daughter's birthday today! Where is my delivery?"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "delivery_issue")
        self.assertIn(res.emotion, ["frustrated", "impatient", "worried", "angry"])
        self.assertEqual(res.sentiment, "negative")
        self.assertGreaterEqual(res.frustration_level, 4)

    def test_scenario_06_worried_tracking_inquiry(self):
        msg = "Hello, I am a bit worried because the tracking number has not shown any movement for several days. Is my order lost?"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "delivery_issue")
        self.assertIn(res.emotion, ["worried", "confused", "neutral"])
        self.assertIn(res.escalation_risk, ["low", "medium"])

    def test_scenario_07_confused_double_charge(self):
        msg = "I checked my bank statement and noticed I was charged twice for the same transaction. Why was I charged double?"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "payment_issue")
        self.assertIn(res.emotion, ["confused", "worried", "frustrated"])
        self.assertIn(res.sentiment, ["neutral", "negative"])

    def test_scenario_08_card_declined(self):
        msg = "My credit card was declined during checkout even though I have sufficient funds. How can I complete my payment?"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "payment_issue")
        self.assertIn(res.sentiment, ["neutral", "negative"])

    def test_scenario_09_account_password_reset(self):
        msg = "I forgot my password and cannot sign in to my account. Could you send me a password reset link?"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "account_issue")
        self.assertEqual(res.emotion, "neutral")
        self.assertEqual(res.sentiment, "neutral")
        self.assertLessEqual(res.frustration_level, 2)
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_10_account_unauthorized_access(self):
        msg = "Someone accessed my account without authorization and changed my email! This is a security emergency!"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "account_issue")
        self.assertIn(res.emotion, ["worried", "frustrated", "angry"])
        self.assertGreaterEqual(res.frustration_level, 3)

    def test_scenario_11_service_complaint(self):
        msg = "The previous representative was extremely rude and unhelpful. I want to file an official complaint regarding this awful experience."
        res = analyze_customer_message(msg)
        self.assertIn(res.intent, ["complaint", "general_inquiry"])
        self.assertIn(res.emotion, ["angry", "frustrated"])
        self.assertEqual(res.sentiment, "negative")
        self.assertGreaterEqual(res.frustration_level, 5)

    def test_scenario_12_damaged_return_exchange(self):
        msg = "The package arrived with the glass screen completely broken and shattered. I need a replacement or exchange right away."
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "return_exchange")
        self.assertIn(res.sentiment, ["neutral", "negative"])

    def test_scenario_13_neutral_size_exchange(self):
        msg = "Hi, the shoes are a bit too small. Can I return them to exchange for size 10?"
        res = analyze_customer_message(msg)
        self.assertEqual(res.intent, "return_exchange")
        self.assertEqual(res.sentiment, "neutral")
        self.assertLessEqual(res.frustration_level, 2)

    def test_scenario_14_general_policy_inquiry(self):
        msg = "Hello, could you let me know what your standard international shipping rates and estimated delivery times are?"
        res = analyze_customer_message(msg)
        self.assertIn(res.intent, ["general_inquiry", "delivery_issue"])
        self.assertEqual(res.sentiment, "neutral")
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_15_happy_positive_feedback(self):
        msg = "Thank you so much! Your product is wonderful and your help solved everything perfectly. Have an amazing day!"
        res = analyze_customer_message(msg)
        self.assertEqual(res.sentiment, "positive")
        self.assertIn(res.emotion, ["happy", "satisfied"])
        self.assertLessEqual(res.frustration_level, 1)
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_16_manager_escalation_trigger(self):
        msg = "I have had enough of this runaround. Transfer me to a supervisor right now!"
        res = analyze_customer_message(msg)
        self.assertEqual(res.escalation_risk, "high")
        self.assertGreaterEqual(res.frustration_level, 8)
        self.assertIn(res.emotion, ["angry", "frustrated"])

    def test_scenario_17_legal_threat_escalation(self):
        msg = "If this isn't resolved today, I will report you to the consumer court and BBB."
        res = analyze_customer_message(msg)
        self.assertEqual(res.escalation_risk, "high")
        self.assertGreaterEqual(res.frustration_level, 8)

    def test_scenario_18_declining_satisfaction_trend(self):
        history = [
            {"role": "user", "content": "Where is my package? It is late."},
            {"role": "assistant", "content": "Please wait another 24 hours."},
            {"role": "user", "content": "I already waited 24 hours! You are not helping me at all!"}
        ]
        msg = "This is the third time you gave me the exact same excuse. I am sick and tired of this!"
        res = analyze_customer_message(msg, history=history)
        self.assertEqual(res.satisfaction_trend, "declining")
        self.assertEqual(res.sentiment, "negative")
        self.assertGreaterEqual(res.frustration_level, 6)

    def test_scenario_19_improving_satisfaction_trend(self):
        history = [
            {"role": "user", "content": "My card was charged twice and I am very angry!"},
            {"role": "assistant", "content": "I apologize. I just processed an immediate reversal for the extra charge."},
        ]
        msg = "Oh, that was fast! The duplicate charge is gone now. Thank you so much for fixing it!"
        res = analyze_customer_message(msg, history=history)
        self.assertEqual(res.satisfaction_trend, "improving")
        self.assertEqual(res.sentiment, "positive")
        self.assertIn(res.emotion, ["satisfied", "happy"])
        self.assertLessEqual(res.frustration_level, 2)
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_20_stable_polite_inquiry(self):
        history = [
            {"role": "user", "content": "Hello, do you support Apple Pay?"},
            {"role": "assistant", "content": "Yes, we accept Apple Pay at checkout."}
        ]
        msg = "Understood, thank you. Does it also support gift cards during the same transaction?"
        res = analyze_customer_message(msg, history=history)
        self.assertEqual(res.satisfaction_trend, "stable")
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_21_terse_neutral_reply(self):
        msg = "ok, got it"
        res = analyze_customer_message(msg)
        self.assertEqual(res.sentiment, "neutral")
        self.assertLessEqual(res.frustration_level, 2)
        self.assertEqual(res.escalation_risk, "low")

    def test_scenario_22_api_endpoint_integration(self):
        # 1. Test POST /analysis/intent-sentiment
        payload = {
            "message": "I demand an immediate refund for my cancelled order. Transfer me to your manager!",
        }
        resp = client.post("/analysis/intent-sentiment", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        analysis = data["analysis"]
        self.assertEqual(analysis["intent"], "refund")
        self.assertEqual(analysis["escalation_risk"], "high")
        self.assertGreaterEqual(analysis["frustration_level"], 8)
        self.assertIn(analysis["sentiment"], ["negative"])
        self.assertIn("confidence", analysis)

        # 2. Test Simulator integration returns analysis
        sim_payload = {
            "agent_message": "Hello, how can I assist you with your order?",
            "config": {
                "persona": "angry",
                "scenario": "refund request for defective laptop",
                "initial_emotion": "angry",
                "current_emotion": "angry",
                "issue_severity": "high",
                "patience_level": "low",
                "expected_resolution": "full refund"
            }
        }
        sim_resp = client.post("/simulator/chat", json=sim_payload)
        self.assertEqual(sim_resp.status_code, 200)
        sim_data = sim_resp.json()
        self.assertIn("analysis", sim_data)
        self.assertIsNotNone(sim_data["analysis"])
        self.assertIn("intent", sim_data["analysis"])
        self.assertIn("frustration_level", sim_data["analysis"])
        self.assertIn("escalation_risk", sim_data["analysis"])
        self.assertIn("suggested_action", sim_data["analysis"])
        self.assertTrue(len(sim_data["analysis"]["suggested_action"]) > 0)

    def test_scenario_23_dynamic_suggested_actions_change_per_message(self):
        msg_refund = "I want a full refund for my defective shoes."
        msg_order_id = "My order number is #ORD-99214. Please check it."
        msg_billing = "I was charged twice on my credit card for one purchase."
        msg_thanks = "Thank you so much! The issue is completely solved now."
        msg_escalate = "Transfer me to your manager immediately!"

        r_refund = analyze_customer_message(msg_refund)
        r_order = analyze_customer_message(msg_order_id)
        r_billing = analyze_customer_message(msg_billing)
        r_thanks = analyze_customer_message(msg_thanks)
        r_escalate = analyze_customer_message(msg_escalate)

        # Assert all suggested actions are populated
        for r in [r_refund, r_order, r_billing, r_thanks, r_escalate]:
            self.assertTrue(len(r.suggested_action) > 0)
            self.assertTrue(len(r.suggested_response) > 0)

        # Assert all suggested actions are distinct and tailored to each message
        actions = [r_refund.suggested_action, r_order.suggested_action, r_billing.suggested_action, r_thanks.suggested_action, r_escalate.suggested_action]
        self.assertEqual(len(set(actions)), 5, "Each message must produce a unique suggested action!")

        responses = [r_refund.suggested_response, r_order.suggested_response, r_billing.suggested_response, r_thanks.suggested_response, r_escalate.suggested_response]
        self.assertEqual(len(set(responses)), 5, "Each message must produce a unique recommended response template!")


if __name__ == "__main__":
    unittest.main()
