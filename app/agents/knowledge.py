import re
from typing import Dict, Any, List, Optional
from app.retrieval.search import semantic_search
from app.models.schemas import KnowledgeRecommendationItem

def formulate_search_query(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Formulates an enriched, context-aware query combining the customer's current
    message with relevant topic keywords from recent conversation turns.
    """
    clean_msg = message.strip()
    if not clean_msg:
        return ""

    if not history:
        return clean_msg

    # Extract user and assistant context from recent turns
    recent = history[-4:]
    context_tokens = []
    
    # Check for domain keywords in history that resolve pronouns (it, that, them, the charge, the package)
    keywords = [
        "refund", "return", "exchange", "cancellation", "cancel", "delivery", "shipping",
        "tracking", "delayed", "charge", "payment", "card", "double", "declined", "password",
        "account", "login", "subscription", "replacement", "damaged", "broken", "manager",
        "supervisor", "warranty"
    ]
    
    for turn in recent:
        content = turn.get("content", "").lower()
        for kw in keywords:
            if kw in content and kw not in clean_msg.lower() and kw not in context_tokens:
                context_tokens.append(kw)

    if context_tokens and any(p in clean_msg.lower() for p in ["it", "this", "that", "them", "second one", "the order", "my package", "why", "how long", "can you"]):
        enriched = f"{clean_msg} ({' '.join(context_tokens[:3])})"
        return enriched

    return clean_msg


def classify_category(text: str, doc_name: str) -> str:
    """
    Determines whether a chunk represents a Policy, FAQ, Troubleshooting, or Support Article.
    """
    t_lower = text.lower()
    d_lower = doc_name.lower()

    if "policy" in d_lower or "guideline" in d_lower or "terms" in t_lower or "rules" in t_lower:
        return "Policy"
    if "faq" in d_lower or "how to" in t_lower or "question" in t_lower or "?" in text[:120]:
        return "FAQ"
    if "troubleshooting" in t_lower or "steps" in t_lower or "1." in text or "verify" in t_lower:
        return "Troubleshooting"
    return "Support Article"


def extract_title(text: str, doc_name: str) -> str:
    """
    Extracts a clean, descriptive title from markdown headings or document names.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
            clean = line.lstrip("#").strip()
            if len(clean) > 5:
                return clean
        if "?" in line and len(line) < 100:
            return line.strip()

    # Fallback to document name formatted
    name = doc_name.replace(".md", "").replace(".pdf", "").replace("_", " ").title()
    return f"{name} Guide"


def extract_actionable_summary(text: str, category: str) -> str:
    """
    Generates a concise 1-2 sentence takeaway for the agent.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    actionable_sentences = [
        s for s in sentences
        if any(w in s.lower() for w in ["support agent", "must", "can", "issued", "immediately", "offer", "refund", "replacement", "cancel", "verify", "check"])
    ]
    if actionable_sentences:
        summary = " ".join(actionable_sentences[:2]).strip()
        if len(summary) > 220:
            summary = summary[:217] + "..."
        return summary

    first_sentence = sentences[0] if sentences else text[:150]
    return first_sentence[:200].strip()


def recommend_knowledge(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 4,
    min_threshold: float = 0.30,
) -> List[KnowledgeRecommendationItem]:
    """
    Retrieves, ranks, and categorizes relevant support knowledge (articles, FAQs,
    troubleshooting steps, policies) from ChromaDB for the given customer turn.
    Returns top 3-5 recommendations; handles out-of-domain queries by returning empty list.
    """
    if not message or not message.strip():
        return []

    # 1. Formulate context-aware query
    query = formulate_search_query(message, history)
    if not query:
        return []

    # 2. Retrieve candidates via semantic search
    raw_results = semantic_search(query, top_k=top_k + 4)
    if not raw_results:
        return []

    # 3. Filter by similarity threshold
    valid_results = [r for r in raw_results if float(r.get("score", 0.0)) >= min_threshold]
    if not valid_results:
        return []

    # 4. Rank and format recommendations
    recommendations: List[KnowledgeRecommendationItem] = []
    seen_snippets = set()

    for item in valid_results:
        raw_text = item.get("text", "").strip()
        if not raw_text:
            continue

        # Deduplicate near-identical chunks
        snippet_sig = raw_text[:80].lower()
        if snippet_sig in seen_snippets:
            continue
        seen_snippets.add(snippet_sig)

        doc_name = item.get("document_name") or item.get("filename") or "Support Document"
        category = classify_category(raw_text, doc_name)
        title = extract_title(raw_text, doc_name)
        actionable_summary = extract_actionable_summary(raw_text, category)

        score = float(item.get("score", 0.0))
        # Format snippet to readable length
        snippet = raw_text if len(raw_text) <= 350 else raw_text[:347] + "..."

        recommendations.append(
            KnowledgeRecommendationItem(
                title=title,
                snippet=snippet,
                category=category,
                score=round(score, 3),
                document_name=doc_name,
                page_number=item.get("page_number"),
                actionable_summary=actionable_summary,
            )
        )

        if len(recommendations) >= top_k:
            break

    return recommendations
