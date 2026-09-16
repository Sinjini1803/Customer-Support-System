from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR, TOP_K
from app.database.database import (
    create_conversation,
    get_conversation,
    add_message,
    get_recent_messages,
    count_messages,
    update_conversation_summary,
)
from app.ingestion.loader import load_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedder import embed_texts
from app.vectorstore.chroma_store import add_chunks
from app.retrieval.search import semantic_search
from app.generation.answer import (
    generate_chat_answer,
    summarize_conversation,
)
from app.models.schemas import (
    ChatRequest,
    SimulatorTurnRequest,
    SimulatorTurnResponse,
    AnalyzeMessageRequest,
    AnalyzeMessageResponse,
    IntentSentimentAnalysis,
    KnowledgeRecommendationItem,
    KnowledgeRecommendationRequest,
    KnowledgeRecommendationResponse,
)
from app.agents.simulator import generate_customer_turn
from app.agents.intent_sentiment import analyze_customer_message
from app.agents.knowledge import recommend_knowledge


app = FastAPI(title="Support RAG API")


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Support RAG API is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ---------------------------------------------------------
# Upload document
# ---------------------------------------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected.",
            )

        original_name = file.filename
        suffix = Path(original_name).suffix.lower()

        allowed_extensions = {
            ".pdf",
            ".txt",
            ".md",
        }

        if suffix not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}",
            )

        # Make sure upload directory exists
        upload_dir = Path(UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Use a short physical filename.
        # This avoids Windows MAX_PATH / long filename problems.
        document_id = uuid4().hex
        saved_path = upload_dir / f"{document_id}{suffix}"

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        saved_path.write_bytes(file_bytes)

        print(f"UPLOAD: saved file -> {saved_path}")

        # -------------------------------------------------
        # Load document
        # -------------------------------------------------

        print("UPLOAD STEP 1: loading document")

        pages = load_document(saved_path)

        print(f"UPLOAD STEP 2: loaded {len(pages)} pages")

        # -------------------------------------------------
        # Clean document
        # -------------------------------------------------

        print("UPLOAD STEP 3: cleaning document")

        cleaned_pages = clean_pages(pages)

        print(
            f"UPLOAD STEP 4: cleaned {len(cleaned_pages)} pages"
        )

        # -------------------------------------------------
        # Chunk document
        # -------------------------------------------------

        print("UPLOAD STEP 5: creating chunks")

        chunks = chunk_pages(cleaned_pages)

        print(f"UPLOAD STEP 6: created {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No usable text was found in the document.",
            )

        # -------------------------------------------------
        # Prepare chunk IDs + metadata
        # -------------------------------------------------

        for chunk in chunks:
            chunk["chunk_id"] = (
                f"{document_id}_"
                f"page_{chunk['page_number']}_"
                f"chunk_{chunk['chunk_number']}"
            )

            chunk["metadata"] = {
                "document_id": document_id,
                "filename": original_name,
                "page_number": chunk["page_number"],
                "chunk_number": chunk["chunk_number"],
            }

        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        print("UPLOAD STEP 7: creating embeddings")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = embed_texts(texts)

        print(
            f"UPLOAD STEP 8: created {len(embeddings)} embeddings"
        )

        # -------------------------------------------------
        # Store in Chroma
        # -------------------------------------------------

        print("UPLOAD STEP 9: storing vectors")

        stored_count = add_chunks(
            chunks,
            embeddings,
        )

        print(
            f"UPLOAD STEP 10: stored {stored_count} chunks"
        )

        return {
            "success": True,
            "document_id": document_id,
            "filename": original_name,
            "chunks": stored_count,
            "message": "Document uploaded and indexed successfully.",
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("UPLOAD ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=(
                f"Upload failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ---------------------------------------------------------
# Chat
# ---------------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        print("\n================ CHAT REQUEST ================")

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        query = request.query.strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty.",
            )

        print(f"CHAT QUERY: {query}")

        # -------------------------------------------------
        # Create conversation if needed
        # -------------------------------------------------

        conversation_id = request.conversation_id

        if conversation_id is None:
            print("CHAT: creating new conversation")

            conversation_id = create_conversation()

            print(
                f"CHAT: created conversation "
                f"{conversation_id}"
            )

        # -------------------------------------------------
        # Get conversation
        # -------------------------------------------------

        print("CHAT STEP 1: loading conversation")

        conversation = get_conversation(
            conversation_id
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        # -------------------------------------------------
        # Get previous conversation history
        # IMPORTANT:
        # Get history BEFORE adding current user message.
        # -------------------------------------------------

        print("CHAT STEP 2: loading recent history")

        history = get_recent_messages(
            conversation_id,
            limit=10,
        )

        if history is None:
            history = []

        print(
            f"CHAT: loaded {len(history)} previous messages"
        )

        # -------------------------------------------------
        # Get existing summary
        # -------------------------------------------------

        summary = ""

        if isinstance(conversation, dict):
            summary = conversation.get(
                "summary",
                "",
            ) or ""
        else:
            try:
                summary = conversation["summary"] or ""
            except Exception:
                summary = ""

        print(
            f"CHAT: summary length = {len(summary)}"
        )

        # -------------------------------------------------
        # Save current user message
        # -------------------------------------------------

        print("CHAT STEP 3: saving user message")

        add_message(
            conversation_id=conversation_id,
            role="user",
            content=query,
        )

        # -------------------------------------------------
        # Semantic search
        # -------------------------------------------------

        print("CHAT STEP 4: semantic search")

        top_k = request.top_k or TOP_K

        results = semantic_search(
            query,
            top_k=top_k,
        )

        if results is None:
            results = []

        print(
            f"CHAT STEP 5: retrieved "
            f"{len(results)} results"
        )

        # -------------------------------------------------
        # Generate AI answer
        # -------------------------------------------------

        print("CHAT STEP 6: generating AI answer")

        answer = generate_chat_answer(
            query=query,
            history=history,
            summary=summary,
            results=results,
        )

        print("CHAT STEP 7: AI answer generated")

        if not answer:
            answer = (
                "I could not generate an answer "
                "for that question."
            )

        # -------------------------------------------------
        # Save assistant response
        # -------------------------------------------------

        print("CHAT STEP 8: saving assistant message")

        add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )

        # -------------------------------------------------
        # Count messages
        # -------------------------------------------------

        print("CHAT STEP 9: counting messages")

        message_count = count_messages(
            conversation_id
        )

        print(
            f"CHAT: message count = {message_count}"
        )

        # -------------------------------------------------
        # Summarize after enough messages
        # -------------------------------------------------

        if message_count >= 20:
            print(
                "CHAT STEP 10: summarizing conversation"
            )

            try:
                all_messages = get_recent_messages(
                    conversation_id,
                    limit=20,
                )

                new_summary = summarize_conversation(
                    all_messages
                )

                if new_summary:
                    update_conversation_summary(
                        conversation_id,
                        new_summary,
                    )

                    print(
                        "CHAT: conversation summary updated"
                    )

            except Exception as summary_error:
                # Do not fail the entire chat if summary fails.
                print(
                    "SUMMARY ERROR:",
                    repr(summary_error),
                )

        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        print("CHAT STEP 11: returning response")

        return {
            "success": True,
            "conversation_id": conversation_id,
            "answer": answer,
            "results": results,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("\n!!!!!!!!!!!!!!!! CHAT ERROR !!!!!!!!!!!!!!!!")
        print("ERROR TYPE:", type(exc).__name__)
        print("ERROR:", str(exc))
        print("ERROR REPR:", repr(exc))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        raise HTTPException(
            status_code=500,
            detail=(
                f"Chat failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ---------------------------------------------------------
# Simulator Chat
# ---------------------------------------------------------

@app.post("/simulator/chat", response_model=SimulatorTurnResponse)
def simulator_chat(request: SimulatorTurnRequest):
    try:
        print("\n================ SIMULATOR CHAT REQUEST ================")

        # -------------------------------------------------
        # Create conversation if needed
        # -------------------------------------------------

        conversation_id = request.conversation_id

        if conversation_id is None:
            print("SIMULATOR: creating new conversation")
            conversation_id = create_conversation("Simulator Session")
            print(f"SIMULATOR: created conversation {conversation_id}")
            
            # Initial state message if provided by the user (optional, could just rely on first turn)
            # Usually the agent starts, or the customer starts.
            # In our case, the agent responds to an implicitly empty history or ongoing.

        # -------------------------------------------------
        # Get existing conversation
        # -------------------------------------------------

        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        # -------------------------------------------------
        # Save Agent's message
        # -------------------------------------------------

        if request.agent_message.strip():
            print("SIMULATOR: saving agent message")
            add_message(
                conversation_id=conversation_id,
                role="assistant", # Using 'assistant' for agent to align with existing roles
                content=request.agent_message.strip(),
            )

        # -------------------------------------------------
        # Get previous history
        # -------------------------------------------------

        history = get_recent_messages(conversation_id, limit=20)
        if history is None:
            history = []

        # -------------------------------------------------
        # Generate Simulated Customer Response
        # -------------------------------------------------

        print("SIMULATOR: generating customer turn")
        result = generate_customer_turn(
            agent_message=request.agent_message.strip(),
            history=history,
            config=request.config,
        )

        customer_message = result.get("customer_message", "Error generating response.")
        current_emotion = result.get("current_emotion", request.config.current_emotion)
        patience_level = result.get("patience_level", request.config.patience_level)

        # -------------------------------------------------
        # Save Customer's message
        # -------------------------------------------------
        print("SIMULATOR: saving customer message")
        add_message(
            conversation_id=conversation_id,
            role="user", # Using 'user' for customer
            content=customer_message,
        )

        # -------------------------------------------------
        # Real-time Intent & Sentiment Analysis (Task 4)
        # -------------------------------------------------
        print("SIMULATOR: running intent & sentiment analysis")
        analysis = analyze_customer_message(
            message=customer_message,
            history=history,
        )

        # -------------------------------------------------
        # Knowledge Recommendations (Milestone 2)
        # -------------------------------------------------
        print("SIMULATOR: retrieving knowledge recommendations")
        knowledge_recs = recommend_knowledge(
            message=customer_message,
            history=history,
            top_k=4,
        )

        print("SIMULATOR: returning response with analysis and knowledge recommendations")
        return SimulatorTurnResponse(
            success=True,
            conversation_id=conversation_id,
            customer_message=customer_message,
            current_emotion=current_emotion,
            patience_level=patience_level,
            analysis=analysis,
            knowledge_recommendations=knowledge_recs,
        )

    except HTTPException:
        raise
    except Exception as exc:
        print("\n!!!!!!!!!!!!!!!! SIMULATOR ERROR !!!!!!!!!!!!!!!!")
        print("ERROR TYPE:", type(exc).__name__)
        print("ERROR:", str(exc))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------
# Intent & Sentiment Analysis Endpoint (Task 4)
# ---------------------------------------------------------

@app.post("/analysis/intent-sentiment", response_model=AnalyzeMessageResponse)
def analyze_intent_and_sentiment(request: AnalyzeMessageRequest):
    try:
        print("\n================ INTENT & SENTIMENT REQUEST ================")
        history = request.history or []
        if request.conversation_id is not None and not history:
            db_history = get_recent_messages(request.conversation_id, limit=10)
            if db_history:
                history = db_history

        analysis = analyze_customer_message(
            message=request.message,
            history=history,
        )

        return AnalyzeMessageResponse(
            success=True,
            analysis=analysis,
        )
    except HTTPException:
        raise
    except Exception as exc:
        print("ANALYSIS ERROR:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------
# Knowledge Recommendation Endpoint (Milestone 2)
# ---------------------------------------------------------

@app.post("/recommendations/knowledge", response_model=KnowledgeRecommendationResponse)
def get_knowledge_recommendations(request: KnowledgeRecommendationRequest):
    try:
        print("\n================ KNOWLEDGE RECOMMENDATION REQUEST ================")
        history = request.history or []
        if request.conversation_id is not None and not history:
            db_history = get_recent_messages(request.conversation_id, limit=10)
            if db_history:
                history = db_history

        recs = recommend_knowledge(
            message=request.message,
            history=history,
            top_k=request.top_k,
        )

        return KnowledgeRecommendationResponse(
            success=True,
            query=request.message,
            recommendations=recs,
            count=len(recs),
        )
    except HTTPException:
        raise
    except Exception as exc:
        print("KNOWLEDGE RECOMMENDATION ERROR:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation retrieval failed: {type(exc).__name__}: {exc}"
        )