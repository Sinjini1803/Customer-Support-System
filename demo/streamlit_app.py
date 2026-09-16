import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import streamlit as st
from app.models.schemas import SimulatorConfig
from app.agents.simulator import generate_initial_customer_message

# ============================================================
# Configuration
# ============================================================

API_URL = "http://127.0.0.1:8000"

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Customer Support Coaching & Simulation Platform",
    page_icon="🎧",
    layout="wide",
)

# ============================================================
# Mode Selection in Sidebar (Only 2 Modes as Requested)
# ============================================================

st.sidebar.title("🎛️ Navigation & Modes")
app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    [
        "🏛️ Three-Panel Live Support Console",
        "📚 Support Knowledge Base & Ingestion",
    ],
    index=0,
)

st.sidebar.divider()

# ============================================================
# MODE 1: THREE-PANEL LIVE SUPPORT CONSOLE
# ============================================================

if app_mode == "🏛️ Three-Panel Live Support Console":
    st.title("🏛️ Live Support Console — Three-Panel Coaching Interface")
    st.caption(
        "Proactive real-time coaching platform: Multi-agent pipeline analyzes customer intent, sentiment, "
        "and frustration, surfaces context-aware RAG knowledge recommendations, and guides support agents turn by turn."
    )

    # Initialize Session State
    if "sim_conversation_id" not in st.session_state:
        st.session_state.sim_conversation_id = None
    if "sim_messages" not in st.session_state:
        st.session_state.sim_messages = []
    if "sim_current_emotion" not in st.session_state:
        st.session_state.sim_current_emotion = "frustrated"
    if "sim_patience" not in st.session_state:
        st.session_state.sim_patience = "low"
    if "latest_analysis" not in st.session_state:
        st.session_state.latest_analysis = None
    if "latest_knowledge_recs" not in st.session_state:
        st.session_state.latest_knowledge_recs = []
    if "last_scenario_preset" not in st.session_state:
        st.session_state.last_scenario_preset = None

    # Sidebar: Scenario & Persona Configuration
    with st.sidebar:
        st.header("⚙️ Scenario & Persona Setup")

        persona = st.selectbox(
            "Customer Persona:",
            ["angry", "frustrated", "impatient", "confused", "calm", "polite"],
            index=0,
            help="Configures customer tone, communication style, and emotional reactivity."
        )

        scenario_presets = {
            "Refund Request": {
                "desc": "Customer received a defective item 2 days ago and demands an immediate full refund without restocking fees.",
                "emotion": "angry",
                "severity": "high",
                "patience": "low",
                "resolution": "immediate full refund with all fees waived"
            },
            "Delayed Order Delivery": {
                "desc": "Birthday gift order delayed 4 days past guaranteed date with no tracking updates.",
                "emotion": "impatient",
                "severity": "high",
                "patience": "low",
                "resolution": "priority dispatch trace and shipping fee refund"
            },
            "Payment Failure / Double Charge": {
                "desc": "Customer noticed duplicate charges during checkout on credit card statement, status pending.",
                "emotion": "confused",
                "severity": "medium",
                "patience": "medium",
                "resolution": "release of pre-authorization hold and single charge confirmation"
            },
            "Order Cancellation": {
                "desc": "Customer accidentally placed duplicate order 15 minutes ago and wants immediate cancellation.",
                "emotion": "calm",
                "severity": "low",
                "patience": "high",
                "resolution": "cancellation confirmation email and hold release"
            },
            "Product Replacement (Damaged)": {
                "desc": "Item arrived with glass screen completely shattered; customer asks for replacement.",
                "emotion": "frustrated",
                "severity": "high",
                "patience": "medium",
                "resolution": "complimentary replacement shipped overnight without returning broken glass"
            },
            "Subscription Renewal Issue": {
                "desc": "Customer billed for annual subscription renewal unexpectedly, inquires about cancellation.",
                "emotion": "worried",
                "severity": "medium",
                "patience": "medium",
                "resolution": "full refund within 14 days and cancellation of auto-renewal"
            },
            "Account Lockout & Security": {
                "desc": "Customer locked out of account after password reset attempts, reports security concern.",
                "emotion": "worried",
                "severity": "high",
                "patience": "low",
                "resolution": "emergency session lock and secure verification link"
            },
            "Supervisor Escalation Demand": {
                "desc": "Customer experienced multiple delays and explicitly demands to speak with a supervisor immediately.",
                "emotion": "angry",
                "severity": "high",
                "patience": "low",
                "resolution": "empathetic de-escalation and warm supervisor queue transfer with ticket ID"
            }
        }

        selected_preset = st.selectbox(
            "Scenario Category:",
            list(scenario_presets.keys()),
            index=0,
        )

        preset_data = scenario_presets[selected_preset]
        scenario_desc = st.text_area("Scenario Context:", value=preset_data["desc"], height=80)

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            initial_emotion = st.selectbox(
                "Starting Emotion:",
                ["angry", "frustrated", "impatient", "confused", "worried", "calm", "neutral"],
                index=["angry", "frustrated", "impatient", "confused", "worried", "calm", "neutral"].index(preset_data["emotion"]),
            )
        with c_s2:
            issue_severity = st.selectbox(
                "Severity:",
                ["critical", "high", "medium", "low"],
                index=["critical", "high", "medium", "low"].index(preset_data["severity"]),
            )

        initial_patience = st.select_slider(
            "Starting Patience Level:",
            options=["low", "medium", "high"],
            value=preset_data["patience"],
        )

        expected_resolution = st.text_input(
            "Expected Customer Resolution:",
            value=preset_data["resolution"]
        )

        st.divider()

        # Check if preset changed, or if user triggers Reset
        reset_triggered = st.button("🔄 Start / Reset Simulation", use_container_width=True, type="primary")

    # Helper function to initialize opening customer message and live coaching
    def initialize_session():
        cfg = SimulatorConfig(
            persona=persona,
            scenario=scenario_desc,
            initial_emotion=initial_emotion,
            current_emotion=initial_emotion,
            issue_severity=issue_severity,
            patience_level=initial_patience,
            expected_resolution=expected_resolution
        )
        opening_text = generate_initial_customer_message(cfg)

        # 1. Real-time Intent & Sentiment for opening message
        analysis = None
        try:
            a_res = requests.post(f"{API_URL}/analysis/intent-sentiment", json={"message": opening_text}, timeout=3)
            if a_res.status_code == 200:
                analysis = a_res.json().get("analysis")
        except Exception:
            pass

        # 2. Real-time Knowledge Recommendations for opening message
        recs = []
        try:
            k_res = requests.post(f"{API_URL}/recommendations/knowledge", json={"message": opening_text, "top_k": 4}, timeout=3)
            if k_res.status_code == 200:
                recs = k_res.json().get("recommendations", [])
        except Exception:
            pass

        st.session_state.sim_conversation_id = None
        st.session_state.sim_messages = [{
            "role": "user",
            "content": opening_text,
            "analysis": analysis,
            "knowledge_recommendations": recs
        }]
        st.session_state.sim_current_emotion = initial_emotion
        st.session_state.sim_patience = initial_patience
        st.session_state.latest_analysis = analysis
        st.session_state.latest_knowledge_recs = recs
        st.session_state.last_scenario_preset = selected_preset

    # Automatically initialize if first load or if scenario preset was changed or reset button clicked
    if reset_triggered or not st.session_state.sim_messages or st.session_state.last_scenario_preset != selected_preset:
        initialize_session()

    # Three-Panel Layout: Panel 1 = Conversation, Panel 2 = Live Coaching, Panel 3 = Knowledge Recommendations
    col_chat, col_coaching, col_knowledge = st.columns([5, 4, 4])

    # -------------------------------------------------------------
    # PANEL 1: LIVE CONVERSATION WINDOW
    # -------------------------------------------------------------
    with col_chat:
        st.subheader("💬 Panel 1: Conversation Window")
        st.caption("Live interaction between Support Agent and Customer Simulator")

        # Display full conversation history with turn numbers
        customer_turn_idx = 0
        for msg in st.session_state.sim_messages:
            role = msg["role"]
            is_agent = (role == "assistant")
            if not is_agent:
                customer_turn_idx += 1
            with st.chat_message(role, avatar="🧑‍💼" if is_agent else "👤"):
                if is_agent:
                    st.markdown("**Support Agent:**")
                else:
                    st.markdown(f"**Customer** *(Turn {customer_turn_idx})*:")
                st.write(msg["content"])
                if not is_agent and msg.get("analysis"):
                    a = msg["analysis"]
                    sentiment_color = "🔴" if a.get('sentiment') == 'negative' else ("🟢" if a.get('sentiment') == 'positive' else "⚪")
                    st.caption(
                        f"🏷️ `{a.get('intent','').replace('_',' ').upper()}` | "
                        f"{sentiment_color} {a.get('sentiment','').title()} | "
                        f"Emotion: *{a.get('emotion','').title()}* | "
                        f"Frustration: **{a.get('frustration_level', 0)}/10** | "
                        f"Risk: **{a.get('escalation_risk', 'low').upper()}**"
                    )

        # Agent Reply Form
        agent_input = st.chat_input("Type your support response to the customer...")

        if agent_input:
            # 1. Add agent message
            st.session_state.sim_messages.append({
                "role": "assistant",
                "content": agent_input
            })

            # 2. Call Simulator Pipeline
            with st.spinner("Simulated customer is typing & coaching pipeline is updating..."):
                try:
                    cfg_payload = {
                        "persona": persona,
                        "scenario": scenario_desc,
                        "initial_emotion": initial_emotion,
                        "current_emotion": st.session_state.sim_current_emotion,
                        "issue_severity": issue_severity,
                        "patience_level": st.session_state.sim_patience,
                        "expected_resolution": expected_resolution
                    }

                    sim_req = {
                        "conversation_id": st.session_state.sim_conversation_id,
                        "agent_message": agent_input,
                        "config": cfg_payload
                    }

                    res = requests.post(f"{API_URL}/simulator/chat", json=sim_req, timeout=60)
                    res.raise_for_status()
                    data = res.json()

                    st.session_state.sim_conversation_id = data.get("conversation_id")
                    customer_msg = data.get("customer_message")
                    st.session_state.sim_current_emotion = data.get("current_emotion", st.session_state.sim_current_emotion)
                    st.session_state.sim_patience = data.get("patience_level", st.session_state.sim_patience)
                    analysis_data = data.get("analysis")
                    recs_data = data.get("knowledge_recommendations", [])

                    # Crucial: Immediately update latest coaching state
                    st.session_state.latest_analysis = analysis_data
                    st.session_state.latest_knowledge_recs = recs_data

                    st.session_state.sim_messages.append({
                        "role": "user",
                        "content": customer_msg,
                        "analysis": analysis_data,
                        "knowledge_recommendations": recs_data
                    })

                    st.rerun()

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to FastAPI backend. Ensure Uvicorn is running on port 8000.")
                except Exception as exc:
                    st.error(f"Error during simulation/coaching: {exc}")

    # -------------------------------------------------------------
    # PANEL 2: REAL-TIME LIVE COACHING FEED (UPDATES ON EACH MESSAGE)
    # -------------------------------------------------------------
    with col_coaching:
        st.subheader("📊 Panel 2: Live Coaching Feed")
        st.caption("Real-time sentiment, frustration scoring & intervention tips")

        # Find latest customer turn
        customer_turns = [m for m in st.session_state.sim_messages if m["role"] == "user"]
        latest_cust_msg = customer_turns[-1] if customer_turns else None

        analysis = None
        if latest_cust_msg and latest_cust_msg.get("analysis"):
            analysis = latest_cust_msg["analysis"]
        elif st.session_state.latest_analysis:
            analysis = st.session_state.latest_analysis

        if analysis:
            # Turn Badge
            turn_num = len(customer_turns)
            st.markdown(f"📍 **Turn #{turn_num} Live Analysis** *(Updated on Customer Message)*")
            if latest_cust_msg:
                st.info(f"🗣️ *Latest Customer Message:*\n\"{latest_cust_msg['content']}\"")

            # Escalation Risk Alert Banner
            risk = analysis.get("escalation_risk", "low").lower()
            if risk == "high":
                st.error("🚨 **ESCALATION RISK: HIGH**\nImmediate intervention required. Empathize and offer direct resolution or supervisor queue.")
            elif risk == "medium":
                st.warning("⚠️ **ESCALATION RISK: MEDIUM**\nCustomer patience is wearing thin. Provide clear timelines and concrete next actions.")
            else:
                st.success("🛡️ **ESCALATION RISK: LOW**\nRoutine customer exchange. Maintain polite, efficient communication.")

            # Metrics Grid Row 1
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Detected Intent", analysis.get("intent", "N/A").upper().replace("_", " "))
            with m2:
                sentiment = analysis.get("sentiment", "neutral").upper()
                sentiment_icon = "🟢" if sentiment == "POSITIVE" else ("🔴" if sentiment == "NEGATIVE" else "⚪")
                st.metric("Sentiment", f"{sentiment_icon} {sentiment}")

            # Metrics Grid Row 2
            m3, m4 = st.columns(2)
            with m3:
                emotion_icon = {"happy": "😊", "satisfied": "✨", "neutral": "😐", "confused": "😕", "worried": "😟", "impatient": "😤", "frustrated": "😠", "angry": "😡", "relieved": "😌"}.get(analysis.get("emotion", "neutral"), "👤")
                st.metric("Customer Emotion", f"{emotion_icon} {analysis.get('emotion', 'N/A').title()}")
            with m4:
                trend = analysis.get("satisfaction_trend", "stable").lower()
                trend_icon = "📈" if trend == "improving" else ("📉" if trend == "declining" else "➡️")
                st.metric("Satisfaction Trend", f"{trend_icon} {trend.title()}")

            # Frustration Level Meter (0 - 10)
            frustration = analysis.get("frustration_level", 0)
            st.markdown(f"**Frustration Level: {frustration} / 10**")
            st.progress(min(1.0, max(0.0, frustration / 10.0)))

            st.caption(f"Classification Confidence: {int(analysis.get('confidence', 0.9) * 100)}%")

            st.divider()

            # Dynamic Suggested Agent Action (Tailored per message)
            st.markdown("💡 **Suggested Agent Action** *(Turn Specific — Changes Each Message)*")
            action_text = analysis.get("suggested_action") or ""
            if action_text:
                st.info(action_text)
            else:
                st.info("Acknowledge customer concerns empathetically and provide prompt assistance.")

            # Recommended Agent Response Template
            suggested_resp = analysis.get("suggested_response") or ""
            if suggested_resp:
                st.markdown("💬 **Recommended Agent Response Template** *(Copy & adapt this response)*")
                st.code(suggested_resp, language="markdown")

            with st.expander("🔍 View Raw Analysis JSON"):
                st.json(analysis)
        else:
            st.info("📊 Coaching analytics will appear here once a customer turn is generated. Click **🔄 Start / Reset Simulation** in the sidebar to begin.")

    # -------------------------------------------------------------
    # PANEL 3: RAG KNOWLEDGE RECOMMENDATIONS PANEL (UPDATES ON EACH MESSAGE)
    # -------------------------------------------------------------
    with col_knowledge:
        st.subheader("📚 Panel 3: Knowledge Recommendations")
        st.caption("Contextually retrieved support articles, FAQs & policies (RAG)")

        recs = None
        if latest_cust_msg and latest_cust_msg.get("knowledge_recommendations"):
            recs = latest_cust_msg["knowledge_recommendations"]
        elif st.session_state.latest_knowledge_recs:
            recs = st.session_state.latest_knowledge_recs

        if recs:
            st.markdown(f"**Surfaced {len(recs)} Knowledge Assets:**")
            for i, item in enumerate(recs, 1):
                category = item.get("category", "Support Article")
                badge = {
                    "Policy": "⚖️ Policy",
                    "FAQ": "❓ FAQ",
                    "Troubleshooting": "🔧 Troubleshooting",
                    "Support Article": "📄 Article"
                }.get(category, "📄 Article")

                score = item.get("score", 0.0)
                relevance_pct = int(score * 100)

                with st.container():
                    st.markdown(f"**{i}. {item.get('title')}**")
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.caption(f"`{badge}` | `{item.get('document_name')}` (Page {item.get('page_number', 1)})")
                    with c2:
                        st.caption(f"Relevance: **{relevance_pct}%**")

                    summary = item.get("actionable_summary")
                    if summary:
                        st.markdown(f"📌 *Resolution Guidance:* {summary}")

                    with st.expander("📖 View Document Excerpt"):
                        st.write(item.get("snippet", ""))
                    st.markdown("---")
        else:
            st.info("📚 Knowledge recommendations automatically update here on each customer turn based on conversation context.")


# ============================================================
# MODE 2: SUPPORT KNOWLEDGE BASE INGESTION (TASK 2)
# ============================================================

else:
    st.title("🤖 Support Knowledge Base & Ingestion")
    st.caption("Upload and index company support documents, FAQs, and policies into ChromaDB.")

    col_up, col_list = st.columns([1, 1])

    with col_up:
        st.subheader("📄 Upload & Index Support Documents")
        uploaded_file = st.file_uploader(
            "Upload a support document",
            type=["pdf", "txt", "md"],
            help="Upload PDF, TXT, or Markdown support policies/manuals.",
        )

        if uploaded_file is not None:
            if st.button("⬆️ Upload & Index Document", use_container_width=True, type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/upload",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type,
                                )
                            },
                            timeout=120,
                        )
                        response.raise_for_status()
                        data = response.json()
                        st.success("Document indexed successfully into ChromaDB!")
                        st.write(f"**Filename:** {data.get('filename')}")
                        st.write(f"**Chunks Stored:** {data.get('chunks', 0)}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to FastAPI. Start Uvicorn first.")
                    except Exception as exc:
                        st.error(f"Upload failed: {exc}")

    with col_list:
        st.subheader("📚 Knowledge Base Status")
        st.markdown(
            "The knowledge base is continuously accessible by the **Knowledge Recommendation Agent** "
            "to surface context-aware policies and FAQs during live interactions."
        )
        st.info("Core Knowledge Corpus Indexed:\n"
                "- `delivery_shipping_faq.md`\n"
                "- `order_cancellation_policy.md`\n"
                "- `product_warranty_replacement.md`\n"
                "- `subscription_management_faq.md`\n"
                "- `escalation_dispute_policy.md`\n"
                "- `refund_policy.pdf`\n"
                "- `payment_faq.pdf`\n"
                "- `account_support.pdf`")