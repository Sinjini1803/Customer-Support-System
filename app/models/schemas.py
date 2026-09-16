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
        description="Escalation risk assessment: low, medium, or high"
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

