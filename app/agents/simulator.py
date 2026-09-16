import json
import os
from typing import Dict, Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from app.models.schemas import SimulatorConfig

# ============================================================
# Configuration
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
)

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set in the .env file.")

client = InferenceClient(api_key=HF_TOKEN, provider="auto")


# ============================================================
# Helper: safely get message content
# ============================================================

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
# Generate Customer Turn
# ============================================================

def generate_customer_turn(
    agent_message: str,
    history: list,
    config: SimulatorConfig,
) -> Dict[str, Any]:
    """
    Simulates the customer's next message given the scenario,
    persona, and the agent's response. Also updates the emotional state.
    """
    
    # --------------------------------------------------------
    # Conversation context
    # --------------------------------------------------------
    
    # Only keep the latest 10 messages for context
    recent_history = history[-10:] if history else []
    
    history_text = ""
    if recent_history:
        history_text = "\n".join(
            f'{"Customer" if msg["role"] == "user" else "Agent"}: {msg["content"]}'
            for msg in recent_history
        )
    else:
        history_text = "No previous conversation."

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    system_prompt = f"""
You are a Customer Simulator Agent for a customer support training platform.
Your task is to generate realistic, turn-by-turn customer responses based on the provided scenario and persona.

SCENARIO CONFIGURATION:
- Persona: {config.persona}
- Scenario: {config.scenario}
- Current Emotion: {config.current_emotion}
- Issue Severity: {config.issue_severity}
- Patience Level: {config.patience_level}
- Expected Resolution: {config.expected_resolution}

RULES:
1. You must respond IN CHARACTER as the customer. 
2. Consider the 'Agent's Response' carefully. If the agent is helpful, your emotion may improve. If unhelpful or slow, your emotion may worsen.
3. Be realistic. Do not be overly verbose unless the persona demands it.
4. Output YOUR RESPONSE as a JSON object with EXACTLY these three keys:
   - "customer_message": Your actual dialogue/message as the customer.
   - "current_emotion": Your new emotional state (e.g., frustrated, calm, happy, angry).
   - "patience_level": Your new patience level (high, medium, low, exhausted).

Output ONLY the raw JSON object, without any markdown formatting like ```json ... ```.
"""

    user_prompt = f"""
CONVERSATION HISTORY:
{history_text}

AGENT'S LATEST RESPONSE:
{agent_message}

Generate your next response and updated emotional state as a JSON object.
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    try:
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            extra_body={
                "reasoning_effort": "low",
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "low",
                },
            },
        )

        content = _extract_content(response)

        if not content:
            raise ValueError("Empty response from LLM")
            
        # Try to parse JSON. Sometimes LLMs still wrap in markdown despite instructions.
        import re
        
        # Strip potential markdown code blocks
        json_str = content
        if "```json" in json_str:
            json_str = re.search(r'```json(.*?)```', json_str, re.DOTALL)
            if json_str:
                json_str = json_str.group(1).strip()
            else:
                json_str = content # fallback
        elif "```" in json_str:
            json_str = re.search(r'```(.*?)```', json_str, re.DOTALL)
            if json_str:
                json_str = json_str.group(1).strip()
            else:
                json_str = content
                
        try:
            parsed_result = json.loads(json_str)
            return {
                "customer_message": parsed_result.get("customer_message", "I don't know what to say."),
                "current_emotion": parsed_result.get("current_emotion", config.current_emotion),
                "patience_level": parsed_result.get("patience_level", config.patience_level)
            }
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}. Raw content: {content}")
            # Fallback if the LLM didn't return valid JSON
            return {
                "customer_message": content, # Just dump the content
                "current_emotion": config.current_emotion,
                "patience_level": config.patience_level
            }

    except Exception as exc:
        print(f"SIMULATOR API WARNING/FALLBACK: {repr(exc)}")
        # Graceful dynamic fallback generator to ensure simulator always generates realistic,
        # scenario-consistent messages and emotional progression even when API quota is exhausted.
        return _generate_fallback_turn(agent_message, history, config)


# ============================================================
# Scenario-driven Realistic Fallback Engine
# ============================================================

def _generate_fallback_turn(
    agent_message: str,
    history: list,
    config: SimulatorConfig,
) -> Dict[str, Any]:
    """
    Heuristic-based contextual response generator that adheres to the configured
    persona, scenario, issue severity, and emotional progression rules.
    Used when external LLM endpoints hit quota or network errors.
    """
    msg_lower = agent_message.lower()
    curr_emotion = (config.current_emotion or config.initial_emotion or "neutral").lower()
    patience = (config.patience_level or "medium").lower()

    # Determine turn count based on history
    turn_count = len([m for m in (history or []) if m.get("role") in ("user", "assistant")]) // 2 + 1

    # 1. Agent Refusal / Hostility / Rude Extortion Detection
    agent_refusal_kw = [
        "will not help", "won't help", "cannot help", "can't help", "not helping", "not going to help",
        "give me money", "more money", "pay me", "not my problem", "not my job", "shut up", "get lost",
        "don't care", "so what", "refuse", "do it yourself", "no help"
    ]
    is_agent_refusal = any(kw in msg_lower for kw in agent_refusal_kw)

    # 2. Agent Unhelpful / Clueless Stalling Detection
    agent_unhelpful_kw = [
        "so what can i do", "what do you want me to do", "i don't know", "what can i do",
        "what should i do", "idk", "why are you asking me"
    ]
    is_agent_unhelpful = any(kw in msg_lower for kw in agent_unhelpful_kw)

    # 3. Agent Explicit Denial Detection
    is_denial = any(neg in msg_lower for neg in [
        "no refund", "cannot refund", "can't refund", "won't refund", "not refund",
        "cannot cancel", "can't cancel", "won't cancel", "no cancellation",
        "cannot replace", "can't replace", "policy does not allow", "we cannot do that"
    ])

    # 4. Positive Resolution Detection (Ensure not negated or refusal)
    is_positive_resolution = not is_denial and not is_agent_refusal and any(kw in msg_lower for kw in [
        "refund", "initiated", "resolved", "overnight", "upgraded", "cancelled",
        "released", "credited", "sent to your inbox", "zero cost", "processed",
        "reversed", "waived"
    ])

    # 5. Agent Information Request Detection
    is_info_request = not is_agent_refusal and not is_agent_unhelpful and any(kw in msg_lower for kw in [
        "order number", "order id", "email", "tracking", "account details",
        "verify", "details", "could you please share", "confirm your"
    ])

    new_emotion = curr_emotion
    new_patience = patience

    if is_agent_refusal:
        new_emotion = "angry"
        new_patience = "low"
    elif is_denial:
        new_emotion = "angry"
        new_patience = "low"
    elif is_agent_unhelpful:
        new_emotion = "frustrated"
        new_patience = "low"
    elif is_positive_resolution:
        if curr_emotion in ["angry", "frustrated", "impatient"]:
            new_emotion = "relieved" if turn_count >= 2 else "calm"
            new_patience = "high"
        elif curr_emotion in ["confused"]:
            new_emotion = "relieved"
            new_patience = "high"
        else:
            new_emotion = "satisfied"
            new_patience = "high"
    elif is_info_request:
        if curr_emotion == "angry" and patience == "low":
            new_emotion = "frustrated" # slight shift as progress is being made
            new_patience = "medium"
        elif curr_emotion == "impatient":
            new_patience = "medium"
    else:
        # Generic or stalling response
        if curr_emotion in ["angry", "frustrated"]:
            new_patience = "low"
        elif curr_emotion == "calm":
            new_emotion = "neutral"

    # --------------------------------------------------------
    # Dialogue Generation based on Turn & Context
    # --------------------------------------------------------
    scenario_lower = config.scenario.lower()
    persona_lower = config.persona.lower()

    if is_agent_refusal:
        message = (
            f"Excuse me?! Did you just say '{agent_message.strip()}'? "
            f"That is completely unacceptable, rude, and unprofessional conduct for a customer support representative! "
            f"I demand to speak to your manager or supervisor right now, and I will be filing an official complaint regarding this interaction!"
        )
    elif is_denial:
        message = (
            f"What do you mean you cannot assist with that? This is completely unacceptable. "
            f"Under company policy, I am entitled to a proper resolution for my {config.scenario}. "
            f"Please transfer me to someone who has the authority to approve my {config.expected_resolution}!"
        )
    elif is_agent_unhelpful:
        message = (
            f"You are the customer support representative—you should be telling me how to resolve this, not asking me! "
            f"I need you to check my account and resolve my {config.expected_resolution} instead of giving me excuses."
        )
    elif is_positive_resolution:
        if persona_lower == "angry":
            message = (
                f"Alright, I see you processed the {config.expected_resolution}. "
                f"It shouldn't have taken this much hassle, but I appreciate that it's finally handled. "
                f"Please ensure a confirmation receipt is sent to my email immediately."
            )
        elif persona_lower == "confused":
            message = (
                f"Oh, that's such a relief! Thank you for clarifying that for me. "
                f"I'm glad the {config.scenario} is sorted out now and I don't have to worry about extra charges."
            )
        elif persona_lower == "impatient":
            message = (
                f"Finally, thank you. I'm glad this was expedited without further delay. "
                f"I'll be watching for the tracking and confirmation email within the hour."
            )
        else:
            message = (
                f"Thank you so much for your quick help in resolving this! "
                f"I really appreciate your assistance with my {config.scenario}."
            )

    elif is_info_request:
        if "order" in msg_lower:
            ref_info = "My order number is #ORD-98421."
        elif "email" in msg_lower:
            ref_info = "My registered email address is customer.support.test@example.com."
        else:
            ref_info = "My reference number is #REF-44219."

        if persona_lower == "angry":
            message = (
                f"{ref_info} Here it is. Now please look into this right away, "
                f"because I've already waited too long for my {config.scenario}."
            )
        elif persona_lower == "confused":
            message = (
                f"Sure, {ref_info} Is that what you need, or do you need me to provide any billing statements as well?"
            )
        elif persona_lower == "impatient":
            message = (
                f"{ref_info} Please check it quickly—I need this expedited immediately."
            )
        else:
            message = (
                f"Yes, certainly. {ref_info} Please let me know if you need any additional details."
            )

    else:
        # Context-aware multi-turn progression
        # Check what the agent actually said to generate appropriate customer follow-up
        agent_lower = agent_message.lower()

        agent_stalling = any(kw in agent_lower for kw in [
            "checking", "looking into", "one moment", "let me check",
            "will check", "investigating", "pulling up", "reviewing", "coordinating"
        ])
        agent_took_action = any(kw in agent_lower for kw in [
            "refund", "cancelled", "initiated", "processed", "shipped", "dispatched",
            "released", "reversed", "credited", "waived", "upgraded", "expedited",
            "sent to your email", "reset link", "locked", "resolved", "confirmed"
        ])
        agent_asked_info = any(kw in agent_lower for kw in [
            "order number", "order id", "email", "could you share", "could you provide",
            "may i have", "please provide", "verify", "reference"
        ])

        if agent_stalling and turn_count >= 3:
            # Customer loses patience after repeated stalling
            new_emotion = "angry"
            new_patience = "low"
            if persona_lower in ["angry", "impatient"]:
                message = (
                    f"You have been 'checking' and 'looking into' this for multiple messages now "
                    f"and I still have no resolution for my {config.scenario}. "
                    f"Either resolve my {config.expected_resolution} right now or transfer me to a supervisor immediately!"
                )
            else:
                message = (
                    f"I understand you're still investigating, but I've been waiting through several "
                    f"messages now. Can you please give me a definitive answer on my {config.scenario}? "
                    f"I really need my {config.expected_resolution} resolved today."
                )

        elif agent_stalling and turn_count == 2:
            # Agent is still stalling on turn 2 — customer is impatient but gives benefit of doubt
            if persona_lower == "angry":
                message = (
                    f"I've already been waiting and this is taking too long. "
                    f"Can you tell me exactly when my {config.expected_resolution} will be completed for my {config.scenario}?"
                )
            elif persona_lower == "impatient":
                message = f"I need a concrete update. How much longer will this take? I need my {config.expected_resolution} sorted urgently."
            elif persona_lower == "confused":
                message = f"I'm still not sure what the status is. What are the next steps to resolve my {config.scenario}?"
            else:
                message = f"Thank you for looking into this. Could you let me know the expected timeframe for my {config.expected_resolution}?"

        elif agent_took_action:
            # Agent confirmed they did something — customer reacts positively or verifies
            if persona_lower == "angry":
                message = (
                    f"Alright. I see you've taken action. Please send me the written confirmation "
                    f"and reference number for my {config.expected_resolution} immediately."
                )
            elif persona_lower == "confused":
                message = (
                    f"Oh, thank you. Just to confirm — will this also handle the {config.scenario} "
                    f"completely, and when will I see the changes on my account?"
                )
            elif persona_lower == "impatient":
                message = (
                    f"Good. I'll be watching for the confirmation. Please make sure the "
                    f"{config.expected_resolution} is fully processed and I receive written proof."
                )
            else:
                message = (
                    f"Thank you so much for handling that. Could you confirm the details and "
                    f"provide a reference number for my {config.expected_resolution}?"
                )

        elif agent_asked_info:
            # Agent asked for info — customer provides it
            # To prevent looping, check if customer already provided info in recent history
            past_customer_msgs = " ".join([m.get("content", "").lower() for m in history if m.get("role") == "user"][-3:])
            already_provided = "my order number is" in past_customer_msgs or "reference number is" in past_customer_msgs or "registered email" in past_customer_msgs

            if already_provided:
                # Customer is annoyed at repeating themselves
                new_emotion = "angry"
                new_patience = "low"
                message = (
                    f"I've already given you my information! I don't understand why you keep asking. "
                    f"Just resolve my {config.expected_resolution} right now!"
                )
            else:
                if "order" in agent_lower:
                    ref_info = "My order number is #ORD-98421."
                elif "email" in agent_lower:
                    ref_info = "My registered email address is customer.support.test@example.com."
                else:
                    ref_info = "My reference number is #REF-44219."

                if persona_lower == "angry":
                    message = (
                        f"{ref_info} Here it is — now please look into this right away "
                        f"because I've already waited too long for my {config.scenario}."
                    )
                elif persona_lower == "confused":
                    message = f"Sure, {ref_info} Is that what you need, or do you need billing statements as well?"
                elif persona_lower == "impatient":
                    message = f"{ref_info} Please check this quickly — I need the {config.expected_resolution} expedited immediately."
                else:
                    message = f"Yes, certainly. {ref_info} Please let me know if you need any additional details."

        elif turn_count == 1:
            # First turn — opening complaint message
            if "refund" in scenario_lower:
                message = f"Hello. I am contacting you because I need a {config.expected_resolution} for my {config.scenario}. Can you assist with this?"
            elif "delay" in scenario_lower or "order" in scenario_lower or "delivery" in scenario_lower:
                message = f"Hi, my package was supposed to arrive already and the status has not updated. This is urgent — I need it as soon as possible."
            elif "payment" in scenario_lower or "card" in scenario_lower or "charge" in scenario_lower:
                message = f"Hello, I noticed an issue with my payment for my recent order. I appear to have been charged, but the status is unclear. Could you check what happened?"
            elif "cancel" in scenario_lower:
                message = f"Hi, I would like to request a cancellation for my order/subscription. Could you please process this for me?"
            elif "replacement" in scenario_lower or "damaged" in scenario_lower or "broken" in scenario_lower:
                message = f"Hi, my item arrived completely broken and damaged. Can you arrange an immediate replacement for me?"
            elif "subscription" in scenario_lower:
                message = f"I received an unexpected charge for a subscription renewal. I'd like to cancel this and get a refund."
            elif "account" in scenario_lower or "password" in scenario_lower or "login" in scenario_lower:
                message = f"Hello, I'm locked out of my account and need help resetting my password urgently."
            else:
                message = f"Hello, I'm experiencing an issue regarding {config.scenario} and I would appreciate your help getting it resolved."

            if persona_lower == "angry":
                message = message.replace("Hello.", "Look, I am very upset.").replace("Hi,", "This is completely unacceptable —")
            elif persona_lower == "impatient":
                message = f"I need urgent assistance right now. {message}"
            elif persona_lower == "confused":
                message = f"Hi, I'm really confused about what's going on. {message}"

        else:
            # Generic follow-up for unmatched cases — escalate toward resolution
            new_emotion = "frustrated" if curr_emotion in ["confused", "neutral"] else "angry"
            new_patience = "low"
            if persona_lower in ["angry", "impatient"]:
                message = (
                    f"This is taking way too long and I'm not getting anywhere with my {config.scenario}. "
                    f"Please transfer me to your supervisor right now or resolve my {config.expected_resolution} immediately!"
                )
            elif persona_lower == "confused":
                message = (
                    f"I'm still really concerned about this {config.scenario}. "
                    f"If you can't fix it, can you connect me with a manager who can help?"
                )
            else:
                message = (
                    f"I understand you are checking, but I really need my {config.scenario} resolved today. "
                    f"Is there someone who can authorize my {config.expected_resolution}?"
                )

    return {
        "customer_message": message,
        "current_emotion": new_emotion,
        "patience_level": new_patience
    }


# ============================================================
# Opening Customer Message Generator
# ============================================================

def generate_initial_customer_message(config: SimulatorConfig) -> str:
    """
    Generates a realistic opening message from the customer based on the selected
    scenario, persona, and expected resolution to initiate live support simulations.
    """
    s_lower = config.scenario.lower()
    p_lower = config.persona.lower()

    if "refund" in s_lower:
        msg = f"Hello, I am contacting you because I need a {config.expected_resolution} for my {config.scenario}. It arrived defective and I cannot use it."
    elif "delay" in s_lower or "delivery" in s_lower or "shipping" in s_lower:
        msg = f"Hi, my package was supposed to arrive four days ago and the carrier tracking has not updated. Where is my delivery?"
    elif "payment" in s_lower or "charge" in s_lower or "bill" in s_lower:
        msg = f"Hello, I noticed an issue on my bank statement: I was charged twice for order #99214 during checkout, but status still says pending. Why was I charged double?"
    elif "cancel" in s_lower:
        msg = f"Hi, I accidentally made a duplicate purchase 20 minutes ago and would like to cancel the second order right away."
    elif "replacement" in s_lower or "damaged" in s_lower or "broken" in s_lower:
        msg = f"The package arrived today but the glass screen is completely shattered and broken into pieces! Can you send a replacement unit?"
    elif "subscription" in s_lower:
        msg = f"I saw an unexpected charge for an annual subscription renewal today. How can I cancel this and get a refund?"
    elif "account" in s_lower or "password" in s_lower or "login" in s_lower:
        msg = f"Hello, I am locked out of my account because I forgot my password and my attempts failed. Can you help me reset it?"
    else:
        msg = f"Hello, I'm reaching out regarding {config.scenario}. I need assistance with {config.expected_resolution}."

    if p_lower == "angry":
        msg = msg.replace("Hello,", "Look, I am very upset!").replace("Hi,", "This is completely unacceptable!")
    elif p_lower == "impatient":
        msg = f"I need urgent assistance right now. {msg}"
    elif p_lower == "confused":
        msg = f"Hi, I'm really confused about what's going on. {msg}"

    return msg

