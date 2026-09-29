import streamlit as st
import requests
import json
import os
from app.models.schemas import SimulatorConfig

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Customer Support Coaching", page_icon="🎧", layout="wide")

st.sidebar.title("🎛️ Navigation & Modes")
app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    ["🏛️ Live Support Console", "📚 Support Knowledge Base & Ingestion"],
    index=0,
)
st.sidebar.divider()

def render_panel_2(analysis, escalation_threshold):
    st.subheader("📊 Panel 2: Live Coaching Feed")
    st.caption("Real-time sentiment, frustration scoring & intervention tips")
    if not analysis:
        st.info("📊 Coaching analytics will appear here once a customer turn is generated.")
        return
    
    risk = analysis.get("escalation_risk", "low").lower()
    reasoning = analysis.get("escalation_reasoning", "")
    threshold = escalation_threshold.lower()
    risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    if risk_levels.get(risk, 1) >= risk_levels.get(threshold, 3):
        st.toast(f"🚨 ALERT: Escalation Risk is {risk.upper()}!", icon="🚨")
        st.error(f"**🚨 GLOBAL ESCALATION ALERT ({risk.upper()}):** {reasoning}")
        
    if risk == "critical":
        st.error(f"💥 **ESCALATION RISK: CRITICAL**\n{reasoning}\n*Immediate supervisor intervention required!*")
    elif risk == "high":
        st.error(f"🚨 **ESCALATION RISK: HIGH**\n{reasoning}\n*Empathize and offer direct resolution or supervisor queue.*")
    elif risk == "medium":
        st.warning(f"⚠️ **ESCALATION RISK: MEDIUM**\n{reasoning}\n*Provide clear timelines and concrete next actions.*")
    else:
        st.success(f"🛡️ **ESCALATION RISK: LOW**\n{reasoning}\n*Maintain polite, efficient communication.*")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Detected Intent", analysis.get("intent", "N/A").upper().replace("_", " "))
    with m2:
        sentiment = analysis.get("sentiment", "neutral").upper()
        sentiment_icon = "🟢" if sentiment == "POSITIVE" else ("🔴" if sentiment == "NEGATIVE" else "⚪")
        st.metric("Sentiment", f"{sentiment_icon} {sentiment}")

    m3, m4 = st.columns(2)
    with m3:
        emotion_icon = {"happy": "😊", "satisfied": "✨", "neutral": "😐", "confused": "😕", "worried": "😟", "impatient": "😤", "frustrated": "😠", "angry": "😡", "relieved": "😌"}.get(analysis.get("emotion", "neutral"), "👤")
        st.metric("Customer Emotion", f"{emotion_icon} {analysis.get('emotion', 'N/A').title()}")
    with m4:
        trend = analysis.get("satisfaction_trend", "stable").lower()
        trend_icon = "📈" if trend == "improving" else ("📉" if trend == "declining" else "➡️")
        st.metric("Satisfaction Trend", f"{trend_icon} {trend.title()}")

    frustration = analysis.get("frustration_level", 0)
    st.markdown(f"**Frustration Level: {frustration} / 10**")
    st.progress(min(1.0, max(0.0, frustration / 10.0)))
    st.caption(f"Classification Confidence: {int(analysis.get('confidence', 0.9) * 100)}%")
    st.divider()
    
    st.markdown("💡 **Suggested Agent Action** *(Turn Specific — Changes Each Message)*")
    action_text = analysis.get("suggested_action") or ""
    if action_text: st.info(action_text)
    else: st.info("Acknowledge customer concerns empathetically and provide prompt assistance.")

    comm_tips = analysis.get("communication_tips", [])
    if comm_tips:
        st.markdown("🎯 **Actionable Communication Tips**")
        for tip in comm_tips:
            st.markdown(f"- {tip}")

    eval_data = analysis.get("response_evaluation", {})
    if eval_data:
        st.markdown("📝 **Response Evaluation**")
        e1, e2, e3, e4 = st.columns(4)
        with e1: st.metric("Tone", f"{eval_data.get('tone', 5)}/5")
        with e2: st.metric("Clarity", f"{eval_data.get('clarity', 5)}/5")
        with e3: st.metric("Empathy", f"{eval_data.get('empathy', 5)}/5")
        with e4: st.metric("Professional", f"{eval_data.get('professionalism', 5)}/5")

    suggested_resp = analysis.get("suggested_response") or ""
    if suggested_resp:
        st.markdown("💬 **Recommended Agent Response Template**")
        st.code(suggested_resp, language="markdown")

    with st.expander("🔍 View Raw Analysis JSON"):
        st.json(analysis)

def render_panel_3(recs):
    st.subheader("📚 Panel 3: Knowledge Recommendations")
    st.caption("Contextually retrieved support articles, FAQs & policies (RAG)")
    if not recs:
        st.info("📚 Knowledge recommendations will appear here.")
        return
        
    st.markdown(f"**Surfaced {len(recs)} Knowledge Assets:**")
    for i, item in enumerate(recs, 1):
        category = item.get("category", "Support Article")
        badge = {"Policy": "⚖️ Policy", "FAQ": "❓ FAQ", "Troubleshooting": "🔧 Troubleshooting", "Support Article": "📄 Article"}.get(category, "📄 Article")
        with st.expander(f"{i}. {badge} | {item.get('title', 'Document')}"):
            st.markdown(f"**Relevance Score:** `{round(item.get('score', 0.0), 3)}`")
            actionable = item.get("actionable_summary", "")
            if actionable:
                st.markdown("**Actionable Summary:**")
                st.success(actionable)
            st.markdown("**Excerpt:**")
            # API returns 'snippet' field (not 'text')
            st.info(item.get("snippet") or item.get("text", ""))
            st.caption(f"Source: `{item.get('document_name', 'Unknown')}`")

def render_chat_history(messages):
    customer_turn_idx = 0
    for msg in messages:
        role = msg["role"]
        is_agent = (role == "assistant")
        if not is_agent: customer_turn_idx += 1
        with st.chat_message(role, avatar="🧑‍💼" if is_agent else "👤"):
            if is_agent: st.markdown("**Support Agent:**")
            else: st.markdown(f"**Customer** *(Turn {customer_turn_idx})*:")
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

if app_mode == "🏛️ Live Support Console":
    st.title("🏛️ Live Support Console")
    st.caption("Proactive real-time coaching platform with Simulator, Manual, and Replay modes.")

    tab_sim, tab_manual, tab_replay = st.tabs(["🤖 Simulator Mode", "✍️ Manual Mode", "⏪ Replay Mode"])
    
    with st.sidebar:
        st.header("⚙️ Escalation Alert Config")
        escalation_threshold = st.selectbox("🚨 Escalation Alert Threshold:", options=["critical", "high", "medium"], index=1)
        st.divider()
    
    # === Simulator Mode ===
    with tab_sim:
        if "sim_conversation_id" not in st.session_state: st.session_state.sim_conversation_id = None
        if "sim_messages" not in st.session_state: st.session_state.sim_messages = []
        if "sim_current_emotion" not in st.session_state: st.session_state.sim_current_emotion = "frustrated"
        if "sim_patience" not in st.session_state: st.session_state.sim_patience = "low"
        if "sim_latest_analysis" not in st.session_state: st.session_state.sim_latest_analysis = None
        if "sim_latest_knowledge_recs" not in st.session_state: st.session_state.sim_latest_knowledge_recs = []
        if "last_scenario_preset" not in st.session_state: st.session_state.last_scenario_preset = None

        with st.sidebar:
            st.header("⚙️ Simulator Persona Setup")
            persona = st.selectbox("Customer Persona:", ["angry", "frustrated", "impatient", "confused", "calm", "polite"], index=0)
            scenario_presets = {
                "Refund Request": {"desc": "Customer received a defective item 2 days ago and demands an immediate full refund without restocking fees.", "emotion": "angry", "severity": "high", "patience": "low", "resolution": "immediate full refund with all fees waived"},
                "Delayed Order Delivery": {"desc": "Birthday gift order delayed 4 days past guaranteed date with no tracking updates.", "emotion": "impatient", "severity": "high", "patience": "low", "resolution": "priority dispatch trace and shipping fee refund"},
                "Payment Failure / Double Charge": {"desc": "Customer noticed duplicate charges during checkout on credit card statement, status pending.", "emotion": "confused", "severity": "medium", "patience": "medium", "resolution": "release of pre-authorization hold and single charge confirmation"},
                "Order Cancellation": {"desc": "Customer accidentally placed duplicate order 15 minutes ago and wants immediate cancellation.", "emotion": "calm", "severity": "low", "patience": "high", "resolution": "cancellation confirmation email and hold release"},
                "Product Replacement (Damaged)": {"desc": "Item arrived with glass screen completely shattered; customer asks for replacement.", "emotion": "frustrated", "severity": "high", "patience": "medium", "resolution": "complimentary replacement shipped overnight without returning broken glass"},
                "Subscription Renewal Issue": {"desc": "Customer billed for annual subscription renewal unexpectedly, inquires about cancellation.", "emotion": "worried", "severity": "medium", "patience": "medium", "resolution": "full refund within 14 days and cancellation of auto-renewal"},
                "Account Lockout & Security": {"desc": "Customer locked out of account after password reset attempts, reports security concern.", "emotion": "worried", "severity": "high", "patience": "low", "resolution": "emergency session lock and secure verification link"},
                "Supervisor Escalation Demand": {"desc": "Customer experienced multiple delays and explicitly demands to speak with a supervisor immediately.", "emotion": "angry", "severity": "high", "patience": "low", "resolution": "empathetic de-escalation and warm supervisor queue transfer with ticket ID"}
            }
            selected_preset = st.selectbox("Scenario Category:", list(scenario_presets.keys()), index=0)
            preset_data = scenario_presets[selected_preset]
            scenario_desc = st.text_area("Scenario Context:", value=preset_data["desc"], height=80)
            
            c_s1, c_s2 = st.columns(2)
            with c_s1: initial_emotion = st.selectbox("Starting Emotion:", ["angry", "frustrated", "impatient", "confused", "worried", "calm", "neutral"], index=["angry", "frustrated", "impatient", "confused", "worried", "calm", "neutral"].index(preset_data["emotion"]))
            with c_s2: issue_severity = st.selectbox("Severity:", ["critical", "high", "medium", "low"], index=["critical", "high", "medium", "low"].index(preset_data["severity"]))
            initial_patience = st.select_slider("Starting Patience Level:", options=["low", "medium", "high"], value=preset_data["patience"])
            expected_resolution = st.text_input("Expected Customer Resolution:", value=preset_data["resolution"])
            st.divider()
            reset_triggered = st.button("🔄 Start / Reset Simulation", use_container_width=True, type="primary")

        def initialize_session():
            st.session_state.sim_messages = []
            st.session_state.sim_latest_analysis = None
            st.session_state.sim_latest_knowledge_recs = []
            
            cfg_payload = {"persona": persona, "scenario": scenario_desc, "initial_emotion": initial_emotion, "current_emotion": initial_emotion, "issue_severity": issue_severity, "patience_level": initial_patience, "expected_resolution": expected_resolution}
            opening_text = f"I am {initial_emotion} about my issue: {scenario_desc}"
            try:
                res = requests.post(f"{API_URL}/simulator/chat", json={"conversation_id": None, "agent_message": "", "config": cfg_payload}, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.sim_conversation_id = data.get("conversation_id")
                    st.session_state.sim_latest_analysis = data.get("analysis")
                    st.session_state.sim_latest_knowledge_recs = data.get("knowledge_recommendations", [])
                    if data.get("customer_message"):
                        opening_text = data.get("customer_message")
            except Exception as e:
                pass

            st.session_state.sim_messages = [{"role": "user", "content": opening_text, "analysis": st.session_state.sim_latest_analysis, "knowledge_recommendations": st.session_state.sim_latest_knowledge_recs}]
            st.session_state.sim_current_emotion = initial_emotion
            st.session_state.sim_patience = initial_patience
            st.session_state.last_scenario_preset = selected_preset

        if reset_triggered or not st.session_state.sim_messages or st.session_state.last_scenario_preset != selected_preset:
            initialize_session()

        col_chat_sim, col_coaching_sim, col_knowledge_sim = st.columns([5, 4, 4])
        with col_chat_sim:
            st.subheader("💬 Panel 1: Conversation Window")
            render_chat_history(st.session_state.sim_messages)
            agent_input = st.chat_input("Type your support response to the customer...", key="sim_input")
            if agent_input:
                st.session_state.sim_messages.append({"role": "assistant", "content": agent_input})
                with st.spinner("Simulated customer is typing..."):
                    try:
                        cfg_payload = {"persona": persona, "scenario": scenario_desc, "initial_emotion": initial_emotion, "current_emotion": st.session_state.sim_current_emotion, "issue_severity": issue_severity, "patience_level": st.session_state.sim_patience, "expected_resolution": expected_resolution}
                        sim_req = {"conversation_id": st.session_state.sim_conversation_id, "agent_message": agent_input, "config": cfg_payload}
                        res = requests.post(f"{API_URL}/simulator/chat", json=sim_req, timeout=60)
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.sim_conversation_id = data.get("conversation_id", st.session_state.sim_conversation_id)
                            st.session_state.sim_messages.append({
                                "role": "user",
                                "content": data.get("customer_message", ""),
                                "analysis": data.get("analysis"),
                                "knowledge_recommendations": data.get("knowledge_recommendations")
                            })
                            st.session_state.sim_current_emotion = data.get("current_emotion", st.session_state.sim_current_emotion)
                            st.session_state.sim_patience = data.get("patience_level", st.session_state.sim_patience)
                            st.session_state.sim_latest_analysis = data.get("analysis")
                            st.session_state.sim_latest_knowledge_recs = data.get("knowledge_recommendations", [])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Simulator request failed: {e}")

        with col_coaching_sim: render_panel_2(st.session_state.sim_latest_analysis, escalation_threshold)
        with col_knowledge_sim: render_panel_3(st.session_state.sim_latest_knowledge_recs)

    # === Manual Mode ===
    with tab_manual:
        if "man_messages" not in st.session_state: st.session_state.man_messages = []
        if "man_latest_analysis" not in st.session_state: st.session_state.man_latest_analysis = None
        if "man_latest_recs" not in st.session_state: st.session_state.man_latest_recs = []

        c1, c2 = st.columns([1, 8])
        with c1:
            if st.button("🔄 Clear", key="man_clear"):
                st.session_state.man_messages = []
                st.session_state.man_latest_analysis = None
                st.session_state.man_latest_recs = []
                st.rerun()
        with c2:
            input_type = st.radio("Next speaker:", ["Customer", "Support Agent"], horizontal=True, key="man_speaker")

        if input_type == "Support Agent":
            col_chat_man, col_coaching_man, col_knowledge_man = st.columns([5, 4, 4])
            with col_chat_man:
                st.subheader("💬 Panel 1: Conversation Window (Manual)")
                render_chat_history(st.session_state.man_messages)
                man_input = st.chat_input("Enter the agent's response...", key="man_input")
                if man_input:
                    msg_obj = {"role": "assistant", "content": man_input}
                    st.session_state.man_messages.append(msg_obj)
                    st.rerun()
            with col_coaching_man: render_panel_2(st.session_state.man_latest_analysis, escalation_threshold)
            with col_knowledge_man: render_panel_3(st.session_state.man_latest_recs)
        else:
            # Customer mode: full-width chat panel only, no coaching panels
            st.subheader("💬 Panel 1: Conversation Window (Manual)")
            render_chat_history(st.session_state.man_messages)
            man_input = st.chat_input("Enter the customer's message...", key="man_input")
            if man_input:
                msg_obj = {"role": "user", "content": man_input}
                with st.spinner("Analyzing customer message..."):
                    hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.man_messages]
                    try:
                        a_res = requests.post(f"{API_URL}/analysis/intent-sentiment", json={"message": man_input, "history": hist}, timeout=10)
                        if a_res.status_code == 200:
                            msg_obj["analysis"] = a_res.json().get("analysis")
                            st.session_state.man_latest_analysis = msg_obj["analysis"]
                    except: pass
                    try:
                        k_res = requests.post(f"{API_URL}/recommendations/knowledge", json={"message": man_input, "history": hist, "top_k": 4}, timeout=10)
                        if k_res.status_code == 200:
                            msg_obj["knowledge_recommendations"] = k_res.json().get("recommendations", [])
                            st.session_state.man_latest_recs = msg_obj["knowledge_recommendations"]
                    except: pass
                st.session_state.man_messages.append(msg_obj)
                st.rerun()

    # === Replay Mode ===
    with tab_replay:
        if "rep_messages" not in st.session_state: st.session_state.rep_messages = []
        if "rep_current_idx" not in st.session_state: st.session_state.rep_current_idx = -1
        
        uploaded_transcript = st.file_uploader("Upload Conversation Transcript (JSON format)", type=["json"], key="rep_upload")
        if uploaded_transcript:
            if st.button("Load Transcript", key="load_transcript"):
                try:
                    data = json.load(uploaded_transcript)
                    st.session_state.rep_messages = data
                    st.session_state.rep_current_idx = 0
                except Exception as e:
                    st.error(f"Failed to load JSON: {e}")
        
        if st.session_state.rep_messages and st.session_state.rep_current_idx >= 0:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("⏪ Previous") and st.session_state.rep_current_idx > 0:
                    st.session_state.rep_current_idx -= 1
                    st.rerun()
            with c2:
                if st.button("🔄 Restart"):
                    st.session_state.rep_current_idx = 0
                    st.rerun()
            with c3:
                if st.button("Next ⏩") and st.session_state.rep_current_idx < len(st.session_state.rep_messages) - 1:
                    st.session_state.rep_current_idx += 1
                    st.rerun()

            current_msgs = st.session_state.rep_messages[:st.session_state.rep_current_idx + 1]
            latest_msg = current_msgs[-1]
            
            if latest_msg.get("role") == "user" and "analysis" not in latest_msg:
                with st.spinner("Analyzing historical customer message..."):
                    hist = [{"role": m["role"], "content": m["content"]} for m in current_msgs[:-1]]
                    try:
                        a_res = requests.post(f"{API_URL}/analysis/intent-sentiment", json={"message": latest_msg["content"], "history": hist}, timeout=10)
                        if a_res.status_code == 200: latest_msg["analysis"] = a_res.json().get("analysis")
                        
                        k_res = requests.post(f"{API_URL}/recommendations/knowledge", json={"message": latest_msg["content"], "history": hist, "top_k": 4}, timeout=10)
                        if k_res.status_code == 200: latest_msg["knowledge_recommendations"] = k_res.json().get("recommendations", [])
                    except: pass
            
            # Find the most recent customer message for display in Panel 2 & 3
            latest_analysis = None
            latest_recs = []
            for m in reversed(current_msgs):
                if m.get("role") == "user":
                    latest_analysis = m.get("analysis")
                    latest_recs = m.get("knowledge_recommendations", [])
                    break
            
            col_chat_rep, col_coaching_rep, col_knowledge_rep = st.columns([5, 4, 4])
            with col_chat_rep:
                st.subheader("💬 Panel 1: Conversation Replay")
                render_chat_history(current_msgs)
                
                # Summary if at the end
                if st.session_state.rep_current_idx == len(st.session_state.rep_messages) - 1:
                    st.success("🏁 Replay Complete!")
                    max_frustration = max([m.get("analysis", {}).get("frustration_level", 0) for m in st.session_state.rep_messages if m.get("role") == "user" and m.get("analysis")], default=0)
                    trends = [m.get("analysis", {}).get("satisfaction_trend", "stable") for m in st.session_state.rep_messages if m.get("role") == "user" and m.get("analysis")]
                    final_trend = trends[-1] if trends else "stable"
                    st.info(f"Summary: Max Frustration was {max_frustration}/10. Final Satisfaction Trend was '{final_trend}'.")
                    
            with col_coaching_rep: render_panel_2(latest_analysis, escalation_threshold)
            with col_knowledge_rep: render_panel_3(latest_recs)

elif app_mode == "📚 Support Knowledge Base & Ingestion":
    st.title("📚 Support Knowledge Base & Document Ingestion")
    st.info("Ingest documents to ChromaDB.")
    
    col_upload, col_list = st.columns(2)
    with col_upload:
        st.subheader("⬆️ Upload Document")
        uploaded_file = st.file_uploader("Upload a support document", type=["pdf", "txt", "md"])
        if uploaded_file is not None:
            if st.button("⬆️ Upload & Index Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/upload",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            timeout=120,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.success("Document indexed successfully into ChromaDB!")
                            st.write(f"**Filename:** {data.get('filename')}")
                            st.write(f"**Chunks Stored:** {data.get('chunks', 0)}")
                    except Exception as exc: st.error(f"Upload failed: {exc}")

    with col_list:
        st.subheader("📚 Knowledge Base Status")
        st.info("Core Knowledge Corpus Indexed:\n- `delivery_shipping_faq.md`\n- `order_cancellation_policy.md`\n- `product_warranty_replacement.md`\n- `subscription_management_faq.md`\n- `escalation_dispute_policy.md`\n- `refund_policy.pdf`\n- `payment_faq.pdf`\n- `account_support.pdf`")
