# from pydantic import BaseModel, Field
# from typing import List, Optional

# class SearchRequest(BaseModel):
#     query: str = Field(..., min_length=2)
#     top_k: int = Field(default=5, ge=1, le=20)

# class SearchResult(BaseModel):
#     text: str
#     score: float
#     document_id: str
#     document_name: str
#     page_number: Optional[int] = None
#     chunk_id: str
#     source: str

# class SearchResponse(BaseModel):
#     query: str
#     results: List[SearchResult]

# class IngestResponse(BaseModel):
#     document_id: str
#     document_name: str
#     pages: int
#     chunks: int
#     message: str


from typing import Optional

from pydantic import BaseModel, Field


# =========================================================
# CHAT REQUEST
# =========================================================

class ChatRequest(BaseModel):
    conversation_id: Optional[int] = None

    query: str = Field(
        ...,
        min_length=1,
        description="User's question",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )


# =========================================================
# CHAT RESPONSE
# =========================================================

class ChatResponse(BaseModel):
    success: bool
    conversation_id: int
    answer: str
    results: list = []


# =========================================================
# SIMULATOR SCHEMAS
# =========================================================

class SimulatorConfig(BaseModel):
    persona: str = Field(default="calm", description="Customer persona (calm, angry, etc.)")
    scenario: str = Field(default="general inquiry", description="The support scenario")
    initial_emotion: str = Field(default="neutral", description="Starting emotional state")
    current_emotion: str = Field(default="neutral", description="Current emotional state")
    issue_severity: str = Field(default="low", description="Severity of the issue")
    patience_level: str = Field(default="high", description="Customer's patience level")
    expected_resolution: str = Field(default="information", description="What the customer wants")


class SimulatorTurnRequest(BaseModel):
    conversation_id: Optional[int] = None
    agent_message: str = Field(..., description="The support agent's response")
    config: SimulatorConfig


class SimulatorTurnResponse(BaseModel):
    success: bool
    conversation_id: int
    customer_message: str = Field(..., description="The simulated customer's next message")
    current_emotion: str = Field(..., description="Updated emotional state")
    patience_level: str = Field(..., description="Updated patience level")
    analysis: Optional["IntentSentimentAnalysis"] = Field(
        default=None,
        description="Real-time intent and sentiment analysis of the customer message"
    )
    knowledge_recommendations: Optional[list] = Field(
        default=None,
        description="Ranked knowledge recommendations (articles, FAQs, troubleshooting steps) from RAG"
    )


# =========================================================
# INTENT & SENTIMENT ANALYSIS SCHEMAS (TASK 4)
# =========================================================

class ResponseEvaluation(BaseModel):
    tone: int = Field(default=5, ge=1, le=5, description="Tone score (1-5)")
    clarity: int = Field(default=5, ge=1, le=5, description="Clarity score (1-5)")
    empathy: int = Field(default=5, ge=1, le=5, description="Empathy score (1-5)")
    professionalism: int = Field(default=5, ge=1, le=5, description="Professionalism score (1-5)")

class IntentSentimentAnalysis(BaseModel):
    intent: str = Field(
        ...,
        description="Detected customer intent (e.g., refund, cancellation, delivery_issue, payment_issue, account_issue, complaint, return_exchange, general_inquiry)"
    )
    emotion: str = Field(
        ...,
        description="Customer emotion (e.g., happy, neutral, confused, worried, frustrated, angry, satisfied)"
    )
    sentiment: str = Field(
        ...,
        description="Sentiment classification: positive, neutral, negative"
    )
    frustration_level: int = Field(
        ...,
        ge=0,
        le=10,
        description="Frustration score between 0 and 10"
    )
    satisfaction_trend: str = Field(
        ...,
        description="Satisfaction trend: improving, declining, or stable"
    )
    escalation_risk: str = Field(
        ...,
        description="Escalation risk assessment: low, medium, high, or critical"
    )
    escalation_reasoning: str = Field(
        default="",
        description="Reasoning for the assigned escalation risk score"
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    suggested_action: str = Field(
        default="",
        description="Dynamic actionable coaching guidance tailored specifically to this message"
    )
    suggested_response: str = Field(
        default="",
        description="Recommended reply template tailored for the agent to respond with"
    )
    response_evaluation: Optional[ResponseEvaluation] = Field(
        default_factory=lambda: ResponseEvaluation(tone=5, clarity=5, empathy=5, professionalism=5),
        description="Evaluation of the agent's recent communication or suggested approach"
    )
    communication_tips: list = Field(
        default=[],
        description="List of actionable communication improvement tips"
    )


class AnalyzeMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Customer message to analyze")
    conversation_id: Optional[int] = Field(default=None, description="Optional conversation ID for history tracking")
    history: Optional[list] = Field(default=None, description="Optional explicit conversation history")


class AnalyzeMessageResponse(BaseModel):
    success: bool = True
    analysis: IntentSentimentAnalysis


# =========================================================
# KNOWLEDGE RECOMMENDATION SCHEMAS (MILESTONE 2)
# =========================================================

class KnowledgeRecommendationItem(BaseModel):
    title: str = Field(..., description="Title of the article, FAQ, or policy")
    snippet: str = Field(..., description="Relevant text excerpt")
    category: str = Field(default="Support Article", description="Category: Policy, FAQ, Troubleshooting, or Support Article")
    score: float = Field(..., description="Relevance similarity score (0.0 to 1.0)")
    document_name: str = Field(..., description="Source document filename")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable")
    actionable_summary: str = Field(default="", description="1-2 sentence key takeaway for the agent")


class KnowledgeRecommendationRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Current customer message to find knowledge for")
    conversation_id: Optional[int] = Field(default=None, description="Optional conversation ID for context formulation")
    history: Optional[list] = Field(default=None, description="Optional conversation history turns")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of recommendations to retrieve")


class KnowledgeRecommendationResponse(BaseModel):
    success: bool = True
    query: str = Field(..., description="Formulated context query used for search")
    recommendations: list = Field(default=[], description="List of ranked knowledge recommendation items")
    count: int = Field(default=0, description="Total count of retrieved recommendations")


# Resolve forward reference
SimulatorTurnResponse.model_rebuild()


# =========================================================
# POST-INTERACTION SUMMARY SCHEMAS (TASK 8)
# =========================================================

class SentimentJourneyPoint(BaseModel):
    turn: int = Field(..., description="Turn number in the conversation")
    role: str = Field(..., description="Speaker: user or assistant")
    content_snippet: str = Field(default="", description="First 80 chars of message")
    emotion: str = Field(default="neutral", description="Detected emotion at this turn")
    frustration: int = Field(default=0, ge=0, le=10, description="Frustration score 0-10")
    sentiment: str = Field(default="neutral", description="positive / neutral / negative")
    escalation_risk: str = Field(default="low", description="low / medium / high / critical")


class ResolutionQualityScore(BaseModel):
    total: float = Field(default=0.0, ge=0.0, le=100.0, description="Composite score 0-100")
    issue_resolution: float = Field(default=0.0, ge=0.0, le=100.0)
    communication_quality: float = Field(default=0.0, ge=0.0, le=100.0)
    empathy_shown: float = Field(default=0.0, ge=0.0, le=100.0)
    guideline_adherence: float = Field(default=0.0, ge=0.0, le=100.0)


class PostInteractionReport(BaseModel):
    session_id: str = Field(default="", description="Unique session identifier")
    timestamp: str = Field(default="", description="ISO timestamp of report generation")
    total_turns: int = Field(default=0)
    customer_turns: int = Field(default=0)
    agent_turns: int = Field(default=0)
    conversation_summary: str = Field(default="", description="Concise 3-5 sentence overview")
    primary_issue: str = Field(default="", description="What the customer needed")
    final_resolution: str = Field(default="", description="How or whether it was resolved")
    resolution_status: str = Field(default="unresolved", description="resolved / partially_resolved / unresolved / escalated")
    sentiment_journey: list = Field(default=[], description="List of SentimentJourneyPoint dicts")
    resolution_quality: Optional[ResolutionQualityScore] = None
    agent_strengths: list = Field(default=[], description="What the agent did well")
    agent_weaknesses: list = Field(default=[], description="Areas needing improvement")
    coaching_recommendations: list = Field(default=[], description="Personalized coaching tips")
    max_frustration: int = Field(default=0)
    final_sentiment: str = Field(default="neutral")
    final_escalation_risk: str = Field(default="low")
    dominant_intent: str = Field(default="general_inquiry")


class PostInteractionRequest(BaseModel):
    session_id: Optional[str] = None
    conversation: list = Field(..., description="List of message dicts with role, content, analysis")


class PostInteractionResponse(BaseModel):
    success: bool = True
    report: PostInteractionReport


# =========================================================
# PERFORMANCE ANALYTICS SCHEMAS (TASK 8)
# =========================================================

class SessionRecord(BaseModel):
    session_id: str
    timestamp: str
    scenario: str = Field(default="")
    total_turns: int = Field(default=0)
    resolution_status: str = Field(default="unresolved")
    resolution_quality_total: float = Field(default=0.0)
    max_frustration: int = Field(default=0)
    final_sentiment: str = Field(default="neutral")
    final_escalation_risk: str = Field(default="low")
    dominant_intent: str = Field(default="general_inquiry")
    coaching_recommendations: list = Field(default=[])
    agent_strengths: list = Field(default=[])
    agent_weaknesses: list = Field(default=[])


class AnalyticsDashboard(BaseModel):
    total_sessions: int = Field(default=0)
    resolved_count: int = Field(default=0)
    partially_resolved_count: int = Field(default=0)
    unresolved_count: int = Field(default=0)
    escalated_count: int = Field(default=0)
    avg_resolution_quality: float = Field(default=0.0)
    avg_frustration: float = Field(default=0.0)
    resolution_trend: list = Field(default=[], description="Per-session resolution quality scores")
    escalation_frequency: dict = Field(default={}, description="Count per risk level")
    common_intents: dict = Field(default={}, description="Count per intent")
    knowledge_gaps: list = Field(default=[], description="Intents with low resolution scores")
    improvement_indicators: dict = Field(default={}, description="Trend analysis")
    top_coaching_recommendations: list = Field(default=[])
    sessions: list = Field(default=[], description="List of SessionRecord dicts")
