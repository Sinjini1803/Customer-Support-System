import json
import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from app.models.schemas import IntentSentimentAnalysis

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")

client = None
_hf_quota_depleted = False

if HF_TOKEN:
    try:
        client = InferenceClient(api_key=HF_TOKEN, provider="auto")
    except Exception as exc:
        print(f"InferenceClient init note: {exc}")


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
    if content:
        return str(content).strip()
    return ""


# ============================================================
# Main Analysis Entry Point
# ============================================================

def analyze_customer_message(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> IntentSentimentAnalysis:
    """
    Analyzes a customer message for:
    - Intent
    - Emotion
    - Sentiment
    - Frustration Level (0-10)
    - Satisfaction Trend (improving, declining, stable)
    - Escalation Risk (low, medium, high)
    - Confidence score (0.0 - 1.0)
    
    Uses LLM when available; falls back to an advanced deterministic NLP analyzer.
    """
    global _hf_quota_depleted
    cleaned_msg = message.strip()
    if not cleaned_msg:
        return IntentSentimentAnalysis(
            intent="general_inquiry",
            emotion="neutral",
            sentiment="neutral",
            frustration_level=0,
            satisfaction_trend="stable",
            escalation_risk="low",
            confidence=0.90,
        )

    # Attempt LLM analysis if client is available and quota not flagged as depleted
    if client and not _hf_quota_depleted:
        try:
            llm_result = _analyze_with_llm(cleaned_msg, history)
            if llm_result:
                return llm_result
        except Exception as exc:
            err_str = str(exc)
            if "402" in err_str or "quota" in err_str.lower() or "credits" in err_str.lower():
                _hf_quota_depleted = True
            print(f"INTENT/SENTIMENT LLM NOTICE: {exc}. Using deterministic engine.")

    # Deterministic NLP Engine
    return _analyze_with_heuristics(cleaned_msg, history)


# ============================================================
# LLM Analysis Method
# ============================================================

def _analyze_with_llm(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[IntentSentimentAnalysis]:
    recent_history = history[-6:] if history else []
    history_text = "\n".join(
        f"{'Customer' if m.get('role') == 'user' else 'Agent'}: {m.get('content', '')}"
        for m in recent_history
    ) if recent_history else "No previous conversation."

    system_prompt = """
You are an expert customer support Intent & Sentiment Analysis Agent.
Analyze the customer's latest message in context of the conversation history.

You must return a JSON object with EXACTLY the following keys:
- "intent": detected customer intent (one of: "refund", "cancellation", "delivery_issue", "payment_issue", "account_issue", "complaint", "return_exchange", "general_inquiry")
- "emotion": customer emotional state (one of: "happy", "neutral", "confused", "worried", "frustrated", "angry", "satisfied")
- "sentiment": sentiment classification (one of: "positive", "neutral", "negative")
- "frustration_level": integer score from 0 to 10
- "satisfaction_trend": one of: "improving", "declining", "stable"
- "escalation_risk": one of: "low", "medium", "high"
- "confidence": float between 0.0 and 1.0
- "suggested_action": specific, actionable coaching recommendation tailored specifically to this exact message and issue
- "suggested_response": recommended response template for the agent to reply with

Return ONLY the raw JSON object without markdown formatting.
"""

    user_prompt = f"""
CONVERSATION HISTORY:
{history_text}

LATEST CUSTOMER MESSAGE:
"{message}"

Provide JSON analysis:
"""

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        max_tokens=500,
        temperature=0.1,
    )

    content = _extract_content(response)
    if not content:
        return None

    json_str = content
    if "```json" in json_str:
        m = re.search(r"```json(.*?)```", json_str, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
    elif "```" in json_str:
        m = re.search(r"```(.*?)```", json_str, re.DOTALL)
        if m:
            json_str = m.group(1).strip()

    data = json.loads(json_str)

    frustration = max(0, min(10, int(data.get("frustration_level", 5))))
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.9))))

    # Fallback to dynamic rule generator if LLM omitted action or response
    suggested_action = str(data.get("suggested_action", "")).strip()
    suggested_response = str(data.get("suggested_response", "")).strip()
    if not suggested_action or not suggested_response:
        def_action, def_resp = _generate_coaching_guidance(
            message,
            str(data.get("intent", "general_inquiry")).lower(),
            str(data.get("emotion", "neutral")).lower(),
            frustration,
            str(data.get("escalation_risk", "low")).lower(),
            history,
        )
        suggested_action = suggested_action or def_action
        suggested_response = suggested_response or def_resp

    return IntentSentimentAnalysis(
        intent=str(data.get("intent", "general_inquiry")).lower(),
        emotion=str(data.get("emotion", "neutral")).lower(),
        sentiment=str(data.get("sentiment", "neutral")).lower(),
        frustration_level=frustration,
        satisfaction_trend=str(data.get("satisfaction_trend", "stable")).lower(),
        escalation_risk=str(data.get("escalation_risk", "low")).lower(),
        confidence=confidence,
        suggested_action=suggested_action,
        suggested_response=suggested_response,
    )


# ============================================================
# Deterministic Comprehensive NLP Engine
# ============================================================

def _analyze_with_heuristics(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> IntentSentimentAnalysis:
    msg_lower = message.lower()
    raw_words = re.findall(r"\b\w+\b", message)
    upper_words = [w for w in raw_words if w.isupper() and len(w) > 1]
    caps_ratio = len(upper_words) / max(len(raw_words), 1)
    exclamations = message.count("!")

    # ---------------------------------------------------------
    # 1. ESCALATION TRIGGERS
    # ---------------------------------------------------------
    escalation_patterns = [
        r"\b(speak|talk|transfer|escalate|demand|get)\b.*?\b(manager|supervisor)\b",
        r"\b(someone|anyone)\b.*?\b(authorize|approve|override)\b",
        r"\b(lawyer|attorney|lawsuit|legal action|consumer court|better business bureau|\bbbb\b|police)\b",
        r"\b(report you|sue you|take you to court)\b",
        r"\b(escalate this|escalate to|transfer me)\b",
        r"\b(file an official complaint|official complaint)\b"
    ]
    has_escalation_phrase = any(re.search(pat, msg_lower) for pat in escalation_patterns)

    # ---------------------------------------------------------
    # 2. INTENT DETECTION
    # ---------------------------------------------------------
    intent = "general_inquiry"
    intent_confidence = 0.88

    refund_kw = ["refund", "money back", "reimburse", "reimbursement", "charge back", "chargeback"]
    cancel_kw = ["cancel", "cancellation", "stop subscription", "terminate account", "unsubscribe"]
    delivery_kw = ["delivery", "delivered", "shipping", "shipment", "package", "order", "arrived", "transit", "carrier", "courier", "tracking", "late", "delay", "delayed", "where is"]
    payment_kw = ["payment", "charged", "billing", "credit card", "debit card", "transaction", "invoice", "double charge", "charged twice", "declined", "pay"]
    account_kw = ["account", "password", "login", "logged out", "sign in", "reset", "hacked", "unauthorized", "profile", "locked out", "access"]
    return_kw = ["return", "exchange", "replace", "replacement", "defective", "broken", "wrong size", "damaged item", "faulty", "shattered"]
    complaint_kw = ["complaint", "rude", "unacceptable", "terrible", "awful", "horrible", "worst service", "incompetent", "bad experience", "poor service"]

    if any(k in msg_lower for k in refund_kw):
        intent = "refund"
        intent_confidence = 0.95
    elif any(k in msg_lower for k in cancel_kw):
        intent = "cancellation"
        intent_confidence = 0.94
    elif any(k in msg_lower for k in payment_kw) and ("double" in msg_lower or "charged" in msg_lower or "twice" in msg_lower or "declined" in msg_lower):
        intent = "payment_issue"
        intent_confidence = 0.93
    elif any(k in msg_lower for k in account_kw):
        intent = "account_issue"
        intent_confidence = 0.92
    elif any(k in msg_lower for k in return_kw):
        intent = "return_exchange"
        intent_confidence = 0.91
    elif any(k in msg_lower for k in delivery_kw):
        intent = "delivery_issue"
        intent_confidence = 0.92
    elif any(k in msg_lower for k in complaint_kw):
        intent = "complaint"
        intent_confidence = 0.90

    # ---------------------------------------------------------
    # 3. FRUSTRATION LEVEL CALCULATION (0 - 10)
    # ---------------------------------------------------------
    frustration_score = 1

    furious_kw = ["furious", "outraged", "disgusted", "livid", "scam", "fraud", "thieves", "stealing", "rip off", "bullshit", "had enough"]
    angry_kw = ["angry", "unacceptable", "terrible", "horrible", "ridiculous", "fed up", "sick and tired", "waste of time", "pissed", "garbage", "trash", "awful", "rude", "unhelpful"]
    annoyed_kw = ["frustrated", "annoyed", "irritated", "disappointed", "waiting forever", "still not", "how hard can it be", "too long", "runaround"]
    confused_kw = ["confused", "don't understand", "not sure", "why was i", "what happened", "unclear", "makes no sense"]
    worried_kw = ["worried", "concerned", "anxious", "scared", "nervous", "security emergency", "unauthorized", "hacked"]

    # Check for desire to resolve (active, ongoing request - NOT already solved)
    desires_resolution = any(p in msg_lower for p in [
        "need this resolved", "need it resolved", "get this resolved", "want this resolved",
        "hope this is resolved", "to be resolved", "get it resolved", "resolve this today",
        "resolved today", "need to resolve", "trying to resolve"
    ])

    # Check for genuine confirmed positive resolution (avoid substring bugs like 'solved' in 'resolved')
    positive_resolution_kw = [
        "thank you so much", "thank you very much", "problem is solved", "issue is solved",
        "fixed it", "that worked", "that fixed it", "such a relief", "huge relief",
        "perfect, thank you", "wonderful, thank you", "amazing day", "great help",
        "solved now", "fixed now", "resolved now", "all sorted out"
    ]

    # Check for negated resolution
    negated_resolution = any(neg in msg_lower for neg in ["not resolved", "isn't resolved", "not solved", "unresolved", "not working", "still not"])

    impatient_kw = ["delayed", "delay", "late", "urgent", "urgently", "asap", "where is my", "needed it", "waiting for", "how much longer"]

    if any(k in msg_lower for k in furious_kw):
        frustration_score += 6
    elif any(k in msg_lower for k in angry_kw):
        frustration_score += 4
    elif any(k in msg_lower for k in annoyed_kw):
        frustration_score += 3
    elif any(k in msg_lower for k in impatient_kw):
        frustration_score += 3
    elif any(k in msg_lower for k in worried_kw):
        frustration_score += 2
    elif any(k in msg_lower for k in confused_kw):
        frustration_score += 2

    if desires_resolution:
        frustration_score = max(frustration_score, 4)

    if caps_ratio > 0.25 and len(raw_words) >= 3:
        frustration_score += 2
    if exclamations >= 2:
        frustration_score += 1
    if exclamations >= 4:
        frustration_score += 1

    if has_escalation_phrase:
        frustration_score = max(frustration_score, 8)
        frustration_score += 2

    if negated_resolution:
        frustration_score = max(frustration_score, 6)

    # History-based repeated complaints
    past_customer_msgs = [m.get("content", "") for m in (history or []) if m.get("role") == "user"]
    if len(past_customer_msgs) >= 2 and any(k in msg_lower for k in angry_kw + annoyed_kw):
        frustration_score += 1
    if len(past_customer_msgs) >= 3:
        frustration_score += 1

    # Dampen frustration if genuine positive resolution without complaints
    if any(k in msg_lower for k in positive_resolution_kw) and not any(k in msg_lower for k in angry_kw + furious_kw) and not has_escalation_phrase and not desires_resolution and not negated_resolution:
        frustration_score = min(frustration_score, 1)

    frustration_level = max(0, min(10, frustration_score))

    # ---------------------------------------------------------
    # 4. EMOTION & SENTIMENT CLASSIFICATION
    # ---------------------------------------------------------
    sentiment = "neutral"
    emotion = "neutral"

    if any(k in msg_lower for k in positive_resolution_kw) and not has_escalation_phrase and not any(k in msg_lower for k in angry_kw + furious_kw) and not desires_resolution and not negated_resolution:
        sentiment = "positive"
        emotion = "happy" if any(k in msg_lower for k in ["great", "awesome", "wonderful", "amazing", "love"]) else "satisfied"
        frustration_level = min(frustration_level, 2)
    elif frustration_level >= 7 or has_escalation_phrase or any(k in msg_lower for k in furious_kw):
        sentiment = "negative"
        emotion = "angry"
    elif frustration_level >= 4 or any(k in msg_lower for k in angry_kw + annoyed_kw) or desires_resolution:
        sentiment = "negative"
        emotion = "impatient" if any(k in msg_lower for k in impatient_kw) else "frustrated"
    elif any(k in msg_lower for k in worried_kw):
        sentiment = "negative" if frustration_level >= 3 else "neutral"
        emotion = "worried"
    elif any(k in msg_lower for k in confused_kw):
        sentiment = "neutral" if frustration_level <= 2 else "negative"
        emotion = "confused"
    elif any(k in msg_lower for k in ["please", "could you", "hello", "hi", "inquiry", "how to", "ok", "got it"]):
        sentiment = "neutral"
        emotion = "neutral"

    # ---------------------------------------------------------
    # 5. SATISFACTION TREND (Improving, Declining, Stable)
    # ---------------------------------------------------------
    satisfaction_trend = "stable"
    if past_customer_msgs:
        prev_msg = past_customer_msgs[-1].lower()
        prev_was_angry = any(k in prev_msg for k in angry_kw + annoyed_kw + furious_kw) or "!" in prev_msg

        if prev_was_angry and (sentiment == "positive" or emotion in ["satisfied", "happy"] or any(k in msg_lower for k in positive_resolution_kw)):
            satisfaction_trend = "improving"
        elif frustration_level >= 6 and (has_escalation_phrase or len(past_customer_msgs) >= 2 or prev_was_angry):
            satisfaction_trend = "declining"
        elif sentiment == "positive" and prev_was_angry:
            satisfaction_trend = "improving"
        elif sentiment == "negative" and len(past_customer_msgs) >= 2:
            satisfaction_trend = "declining"
        else:
            satisfaction_trend = "stable"
    else:
        if sentiment == "positive":
            satisfaction_trend = "improving"
        elif frustration_level >= 7 or has_escalation_phrase:
            satisfaction_trend = "declining"
        else:
            satisfaction_trend = "stable"

    # ---------------------------------------------------------
    # 6. ESCALATION RISK ASSESSMENT
    # ---------------------------------------------------------
    if has_escalation_phrase or frustration_level >= 8:
        escalation_risk = "high"
    elif frustration_level >= 4 or satisfaction_trend == "declining":
        escalation_risk = "medium"
    else:
        escalation_risk = "low"

    confidence = round(min(0.98, max(0.85, intent_confidence + (0.04 if has_escalation_phrase else 0.0))), 2)

    # ---------------------------------------------------------
    # 7. DYNAMIC PER-MESSAGE COACHING GUIDANCE & RESPONSE
    # ---------------------------------------------------------
    suggested_action, suggested_response = _generate_coaching_guidance(
        message=message,
        intent=intent,
        emotion=emotion,
        frustration_level=frustration_level,
        escalation_risk=escalation_risk,
        history=history,
    )

    return IntentSentimentAnalysis(
        intent=intent,
        emotion=emotion,
        sentiment=sentiment,
        frustration_level=frustration_level,
        satisfaction_trend=satisfaction_trend,
        escalation_risk=escalation_risk,
        confidence=confidence,
        suggested_action=suggested_action,
        suggested_response=suggested_response,
    )


# ============================================================
# Dynamic Per-Message Coaching Guidance Engine
# ============================================================

def _generate_coaching_guidance(
    message: str,
    intent: str,
    emotion: str,
    frustration_level: int,
    escalation_risk: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, str]:
    """
    Generates tailored, turn-by-turn actionable coaching advice and
    recommended agent response templates that dynamically change with
    each message, tracking conversation stage to avoid repetition and
    progressively drive toward resolution.
    """
    msg_lower = message.lower()
    history = history or []

    # ----------------------------------------------------------
    # Detect Conversation Stage
    # ----------------------------------------------------------
    customer_turns = [m for m in history if m.get("role") == "user"]
    agent_turns = [m for m in history if m.get("role") == "assistant"]
    turn_number = len(customer_turns) + 1   # current turn (1-indexed)

    # What did the agent last say?
    last_agent_msg = agent_turns[-1].get("content", "").lower() if agent_turns else ""
    # What did the customer previously say?
    last_customer_msgs = [m.get("content", "").lower() for m in customer_turns[-3:]]

    # Stage flags based on agent history
    agent_has_asked_for_info = any(
        kw in last_agent_msg for kw in [
            "order number", "order id", "email", "could you share", "could you provide",
            "verify", "may i have", "please provide", "your account", "reference"
        ]
    )
    agent_has_confirmed_action = any(
        kw in last_agent_msg for kw in [
            "initiated", "processed", "cancelled", "refund", "issued", "dispatched",
            "shipped", "sent to your email", "released", "reversed", "resolved",
            "waived", "expedited", "upgraded", "locked", "reset link", "credited"
        ]
    )
    agent_is_investigating = any(
        kw in last_agent_msg for kw in [
            "checking", "looking into", "investigating", "will check", "let me look",
            "pulling up", "reviewing", "I am looking", "coordinating", "one moment"
        ]
    )

    # Customer-side state flags
    customer_has_given_info = bool(
        re.search(r"[\\w\\.-]+@[\\w\\.-]+\\.\\w+", msg_lower) or
        re.search(r"(order|ref|ticket|#)\\s*[:#]?\\s*[a-zA-Z0-9\\-]{4,}", msg_lower) or
        "my email" in msg_lower or "registered email" in msg_lower or
        "order number is" in msg_lower or "reference number" in msg_lower
    )
    customer_is_still_waiting = any(
        kw in msg_lower for kw in [
            "still waiting", "nothing yet", "no update", "not received",
            "still not", "any update", "what's happening", "how much longer",
            "any progress", "haven't heard", "checked yet"
        ]
    )
    customer_is_escalating = escalation_risk == "high" or any(
        kw in msg_lower for kw in [
            "manager", "supervisor", "legal", "lawsuit", "complaint", "bbb",
            "authorize", "someone who can", "report you", "consumer court"
        ]
    )
    customer_is_satisfied = any(
        kw in msg_lower for kw in [
            "thank you so much", "thank you very much", "problem is solved",
            "issue is solved", "such a relief", "appreciate your help",
            "solved now", "fixed now", "resolved now", "all sorted out",
            "that worked", "great help", "wonderful"
        ]
    ) and frustration_level <= 2

    # ----------------------------------------------------------
    # PRIORITY 1: Customer Confirms Resolution / Thanks
    # ----------------------------------------------------------
    if customer_is_satisfied and not any(
        kw in msg_lower for kw in ["not resolved", "still", "unresolved"]
    ):
        action = (
            "Resolution confirmed! Warmly acknowledge the customer's satisfaction, "
            "offer a brief service survey or feedback, and ensure they know you're "
            "available for any future needs."
        )
        response = (
            "You're very welcome! I'm so glad we were able to get this sorted out for you. "
            "If you have a moment, we'd love your feedback on today's support experience. "
            "Please don't hesitate to reach out if there's ever anything else we can help with!"
        )
        return action, response

    # ----------------------------------------------------------
    # PRIORITY 2: Escalation Demand
    # ----------------------------------------------------------
    if customer_is_escalating:
        if turn_number >= 3 and agent_is_investigating:
            action = (
                "Customer has escalated after repeated stalling. STOP investigating — "
                "take immediate ownership: authorize the resolution directly or transfer "
                "to a supervisor NOW. Do not delay further."
            )
            response = (
                "I completely understand your frustration, and I sincerely apologize "
                "that this has taken longer than it should. I am personally authorizing "
                "the resolution for your case right now, and I will stay on this until "
                "it's fully confirmed. Would you also like me to connect you with a "
                "senior supervisor to oversee this?"
            )
        else:
            action = (
                "De-escalate immediately: acknowledge urgency without defensiveness, "
                "validate the customer's right to escalate, and offer a direct supervisor "
                "transfer while you personally take ownership of the case."
            )
            response = (
                "I sincerely apologize for the frustration this has caused. I am "
                "personally taking ownership of your case right now. I can escalate "
                "this directly to a senior supervisor if you'd prefer — I'm here to "
                "ensure this is fully resolved for you today."
            )
        return action, response

    # ----------------------------------------------------------
    # PRIORITY 3: Customer Still Waiting After Agent Action
    # ----------------------------------------------------------
    if customer_is_still_waiting and agent_has_confirmed_action:
        action = (
            "Customer is following up on a previously confirmed action. "
            "Provide a concrete status update with timestamp. If delay exists, "
            "offer an ETA or escalate to senior support."
        )
        response = (
            "I can see the action was recorded on our end — let me pull up the "
            "exact processing status right now to confirm it's reflecting on your "
            "account. I'll have a specific update for you in just a moment."
        )
        return action, response

    if customer_is_still_waiting and agent_is_investigating:
        action = (
            "Customer is growing impatient with an open investigation. "
            "Stop investigating and deliver a definitive answer NOW — "
            "even if partial. Give a specific timeline and next step."
        )
        response = (
            "I apologize for the wait — I have your case open right in front of me. "
            "Here is the current status: [insert update]. The next step will be "
            "[specific action] and you can expect [timeframe]. I will not leave you "
            "without a clear answer today."
        )
        return action, response

    # ----------------------------------------------------------
    # PRIORITY 4: Customer Provides Information (Email/Order ID)
    # ----------------------------------------------------------
    if customer_has_given_info:
        action = (
            f"Customer has just provided their identifier. Immediately verify the "
            f"account and pull up the {intent.replace('_', ' ')} details. "
            f"Deliver a concrete status update — do NOT ask for more information."
        )
        response = (
            f"Thank you for providing that. I've pulled up your account and I'm "
            f"reviewing the {intent.replace('_', ' ')} details right now. "
            f"Here is what I can see: [insert status]. I'll proceed with the "
            f"resolution immediately."
        )
        return action, response

    # ----------------------------------------------------------
    # PRIORITY 5: Agent Already Asked for Info — Customer Didn't Provide
    # ----------------------------------------------------------
    if agent_has_asked_for_info and not customer_has_given_info and turn_number >= 2:
        action = (
            "Customer has not yet provided the requested information. "
            "Gently re-ask with clear context on WHY it's needed and how "
            "it will immediately help resolve their issue."
        )
        response = (
            "To resolve this as quickly as possible, I just need your order number "
            "or the email address on your account — this will let me look up the "
            "exact transaction and take action right away. What would you like to share?"
        )
        return action, response

    # ----------------------------------------------------------
    # PRIORITY 6: Intent + Turn-Specific Guidance
    # ----------------------------------------------------------

    # REFUND
    if intent == "refund":
        if turn_number == 1:
            if frustration_level >= 6 or emotion in ["angry", "frustrated"]:
                action = (
                    "First contact — angry refund request. Immediately validate the "
                    "customer's dissatisfaction, ask for the order number, and commit "
                    "to processing a full refund with no restocking fee."
                )
                response = (
                    "I'm so sorry to hear about this experience — that's absolutely "
                    "not acceptable. I want to resolve this for you right away. "
                    "Could you please share your order number so I can initiate your "
                    "full refund immediately?"
                )
            else:
                action = (
                    "Polite first contact for refund. Acknowledge the request warmly "
                    "and ask for the order reference to begin processing."
                )
                response = (
                    "I'd be happy to assist with your refund request! Could you please "
                    "share your order number so I can pull up the details and get this "
                    "started for you right away?"
                )
        elif turn_number == 2:
            if agent_has_asked_for_info:
                action = (
                    "You've asked for the order number — if customer provided it, "
                    "pull it up and confirm eligibility. If not yet provided, "
                    "clarify the refund policy timeline while waiting."
                )
                response = (
                    "Once I have your order number, I can immediately check the "
                    "transaction and initiate the refund. Our standard processing takes "
                    "3–5 business days back to your original payment method."
                )
            else:
                action = (
                    "Acknowledge follow-up refund inquiry. Confirm eligibility and "
                    "provide a specific refund timeline commitment."
                )
                response = (
                    "I'm actively reviewing your refund case now. Based on the policy, "
                    "you're eligible for a full refund. I'm initiating the reimbursement "
                    "to your original payment method — you'll receive a confirmation "
                    "email within 15 minutes."
                )
        else:
            action = (
                "Turn 3+: Refund case should be closing. If not yet resolved, "
                "process the refund immediately and provide a confirmation number. "
                "Do not ask more questions."
            )
            response = (
                "I'm processing your full refund right now — no further information "
                "needed. Your refund confirmation number is [REF-XXXXX] and the funds "
                "will appear on your statement within 3–5 business days. Is there "
                "anything else I can help you with today?"
            )
        return action, response

    # CANCELLATION
    if intent == "cancellation":
        if turn_number == 1:
            action = (
                "First contact for cancellation. Confirm whether the order/subscription "
                "is still within the cancellation window and take immediate action."
            )
            response = (
                "I can definitely help with that cancellation. Let me check the "
                "current order status — if it hasn't shipped yet, I can cancel it "
                "immediately and release any pending authorization hold."
            )
        elif agent_has_confirmed_action:
            action = (
                "Cancellation already confirmed. Provide the cancellation reference "
                "and confirm any associated holds are released."
            )
            response = (
                "Your cancellation has been fully processed. Reference number: "
                "[CAN-XXXXX]. Any authorization hold on your account will be "
                "released within 1–2 business days. A confirmation email is on its way!"
            )
        else:
            action = (
                "Follow-up cancellation turn. Confirm the cancellation is complete, "
                "send a confirmation email, and offer any applicable refund."
            )
            response = (
                "I've just confirmed the cancellation is complete. Your confirmation "
                "email is being sent right now, and any pending charges will be "
                "reversed within 1–2 business days."
            )
        return action, response

    # DELIVERY / SHIPPING
    if intent == "delivery_issue":
        if turn_number == 1:
            action = (
                "First contact — delivery issue. Pull up carrier tracking immediately "
                "and provide the last known scan location and estimated arrival window."
            )
            response = (
                "I'm sorry to hear your delivery is delayed! Let me pull up the "
                "live carrier tracking for your order right now. Could you share "
                "your order number so I can check the latest transit status?"
            )
        elif turn_number == 2:
            if "tracking" in msg_lower or "where" in msg_lower or "status" in msg_lower:
                action = (
                    "Customer is asking for tracking status again. Deliver the "
                    "EXACT last scan location, timestamp, and revised delivery window. "
                    "Do not say 'still checking' — commit to a specific answer."
                )
                response = (
                    "I've pulled up the tracking — here is the current status: "
                    "Last scan: [Location] at [Time]. Expected delivery: [Date/Window]. "
                    "If this doesn't arrive by [date], I will immediately initiate a "
                    "priority dispatch trace and shipping fee waiver."
                )
            else:
                action = (
                    "Acknowledge ongoing delay frustration. Contact priority dispatch "
                    "team and offer shipping fee reimbursement as goodwill gesture."
                )
                response = (
                    "I completely understand how frustrating a delayed delivery is, "
                    "especially when it's time-sensitive. I've just contacted our "
                    "priority dispatch team to locate your package and expedite it. "
                    "I'm also waiving your shipping fee as an apology for the delay."
                )
        else:
            action = (
                "Turn 3+: Delivery still unresolved. Escalate to priority carrier "
                "trace, offer a confirmed re-dispatch or full refund, and "
                "give a specific delivery guarantee."
            )
            response = (
                "I have now escalated this to our priority carrier liaison. Your "
                "options are: (1) Priority re-dispatch arriving within [X] days, or "
                "(2) Full refund processed today. Which would you prefer? "
                "I will make sure this is resolved for you right now."
            )
        return action, response

    # PAYMENT / BILLING
    if intent == "payment_issue":
        if turn_number == 1:
            if "twice" in msg_lower or "double" in msg_lower:
                action = (
                    "First contact — double charge reported. Immediately explain "
                    "pre-authorization holds vs settled charges and check payment logs."
                )
                response = (
                    "I can see why that would be alarming — let me pull up your payment "
                    "gateway records right now. In many cases, the second charge is a "
                    "temporary pre-authorization that drops within 24–72 hours. "
                    "Could you share your order number so I can confirm the exact status?"
                )
            elif "declined" in msg_lower:
                action = (
                    "Payment declined — check for common causes: billing address mismatch, "
                    "daily limit, or card block. Suggest alternative payment methods."
                )
                response = (
                    "I'm sorry your payment didn't go through. This is often caused by "
                    "a billing address mismatch or a bank block. Could you verify your "
                    "billing zip code matches your card records? Alternatively, I can "
                    "help you use PayPal or Apple Pay to complete your order."
                )
            else:
                action = (
                    "Payment inquiry — pull itemized invoice and clarify all charges "
                    "with timestamps and amounts."
                )
                response = (
                    "Let me pull up your complete billing history and itemized statement "
                    "so we can review each charge together. Could you share your order "
                    "number or account email to get started?"
                )
        elif turn_number == 2:
            action = (
                "Follow-up payment turn. Provide specific outcome from payment gateway "
                "check — confirm hold release date or issue refund for erroneous charge."
            )
            response = (
                "I've checked the payment gateway records. Here's the confirmed status: "
                "[insert: hold release date / refund initiated / single charge confirmed]. "
                "You will receive an updated statement within 1–2 business days. "
                "Is there anything else you'd like me to verify?"
            )
        else:
            action = (
                "Turn 3+: Payment issue must be resolved NOW. Confirm the specific "
                "outcome (refund issued / hold released) with a reference number and "
                "expected timeline. Close the loop."
            )
            response = (
                "I've confirmed the resolution: [specific outcome — e.g., the pre-auth "
                "hold has been released / refund of $XX initiated with reference #XXXXX]. "
                "The correction will reflect on your statement within 1–2 business days. "
                "Thank you for your patience — is there anything else I can help with?"
            )
        return action, response

    # ACCOUNT / SECURITY
    if intent == "account_issue":
        if any(k in msg_lower for k in ["unauthorized", "hacked", "security", "stolen", "breach"]):
            action = (
                "SECURITY PRIORITY: Lock all active sessions immediately, verify "
                "customer identity via secondary channel, and initiate emergency "
                "password reset with breach report."
            )
            response = (
                "I'm treating this as an urgent security matter. I've locked all "
                "active sessions on your account right now to prevent further "
                "unauthorized access. I'm sending a secure one-time verification "
                "link to your registered email/phone. Please do not share it with anyone."
            )
        elif turn_number == 1:
            action = (
                "First contact — account access issue. Send password reset email "
                "immediately and guide customer through the steps."
            )
            response = (
                "I can help you regain access right away! I'm sending a secure "
                "password reset link to your registered email address now. "
                "It should arrive within 2 minutes — please check your spam folder "
                "if you don't see it. Let me know when you have it and I'll walk "
                "you through the next steps."
            )
        else:
            action = (
                "Follow-up account turn. Confirm whether the reset link was received "
                "and guide through completion. If not received, resend or use alternate."
            )
            response = (
                "Have you received the password reset link yet? If not, I can resend it "
                "to a different email address or use an alternative verification method. "
                "Once you're back in, I recommend updating your security questions as well."
            )
        return action, response

    # RETURN / EXCHANGE
    if intent == "return_exchange":
        if turn_number == 1:
            if any(k in msg_lower for k in ["broken", "defective", "damaged", "shattered"]):
                action = (
                    "Damaged item on first contact. Apologize immediately, waive "
                    "the return of the broken item, and dispatch a free replacement."
                )
                response = (
                    "I am so sorry your item arrived damaged — that is completely "
                    "unacceptable! I have arranged for a free replacement to be sent "
                    "to you right away. You do NOT need to return the broken item. "
                    "Can I confirm your shipping address is still the same?"
                )
            else:
                action = (
                    "Standard exchange — confirm return window eligibility and "
                    "email a prepaid return label immediately."
                )
                response = (
                    "I'd be happy to help with your exchange! I'm generating a prepaid "
                    "return shipping label for you right now and I'll have the replacement "
                    "reserved. You'll receive the label by email in the next 5 minutes."
                )
        else:
            action = (
                "Follow-up return/exchange turn. Confirm label was received, "
                "provide replacement shipping timeline, and offer tracking number."
            )
            response = (
                "Your prepaid return label has been sent. Once we receive the item back "
                "([X] business days), your replacement will ship within 24 hours. "
                "Would you like me to provide the replacement tracking number now?"
            )
        return action, response

    # COMPLAINT
    if intent == "complaint":
        if turn_number == 1:
            action = (
                "Service complaint on first turn. Offer an unreserved apology, "
                "document the feedback, and immediately offer a concrete remedy."
            )
            response = (
                "I sincerely apologize for the experience you've had — that is not "
                "the standard we hold ourselves to. I've documented your feedback "
                "and I'm personally handling your case to make sure it's resolved "
                "correctly. What would you like us to do to make this right?"
            )
        else:
            action = (
                "Follow-up complaint turn. Confirm the remedy that was offered and "
                "deliver the resolution — do not apologize again without action."
            )
            response = (
                "I've confirmed the remedy for your situation: [specific action taken]. "
                "I want to assure you that your feedback has also been escalated to our "
                "quality team to prevent this from happening again. Is there anything "
                "else I can do for you today?"
            )
        return action, response

    # IMPATIENT / FRUSTRATED — no specific intent match
    if emotion in ["impatient", "frustrated"] or customer_is_still_waiting:
        if turn_number >= 3:
            action = (
                "Customer is frustrated after multiple turns without resolution. "
                "Stop stalling — deliver the resolution NOW with a specific outcome "
                "and confirmation number. Avoid any more vague assurances."
            )
            response = (
                "I hear you — and I'm delivering the resolution right now: "
                "[specific action completed]. Confirmation reference: [XXXXX]. "
                "I apologize this took longer than it should have. Is there "
                "anything else I can resolve for you today?"
            )
        else:
            action = (
                "Customer is growing impatient. Acknowledge the wait explicitly, "
                "provide a specific next step with a time commitment — "
                "avoid scripted phrases like 'please hold'."
            )
            response = (
                "I completely understand your frustration and I apologize for the "
                "delay. I am working on this right now and will have a concrete "
                "answer for you within the next 60 seconds. You have my full attention."
            )
        return action, response

    # CONFUSED
    if emotion == "confused":
        action = (
            "Customer is confused. Break down the situation into clear numbered "
            "steps, use plain language, and confirm understanding before proceeding."
        )
        response = (
            "Let me explain this clearly in simple steps:\n"
            "1. [First thing that happened]\n"
            "2. [Current status]\n"
            "3. [What happens next]\n"
            "Does that make sense? I'm here to walk you through each step."
        )
        return action, response

    # DEFAULT — General Inquiry
    if turn_number == 1:
        action = (
            "First contact for general inquiry. Provide a direct, accurate answer "
            "and proactively offer any related resources or next steps."
        )
        response = (
            "I'd be happy to help with that! Here's the information you need: "
            "[specific answer]. Is there anything else you'd like to know?"
        )
    else:
        action = (
            "Follow-up general inquiry. Confirm the previous answer was satisfactory "
            "and proactively address any remaining questions."
        )
        response = (
            "I hope that answered your question! To summarize what we covered: "
            "[brief recap]. Please feel free to ask if you need any further clarification."
        )
    return action, response


