# [file name]: api/main.py
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessageChunk
import json
from uuid import uuid4
import logging
import asyncio

# Import your existing system
from core.system_initializer import setup_system
from models.state_models import MultiCountryLegalState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
chat_manager = None
graph = None

async def initialize_system():
    """Initialize the legal assistant system"""
    global chat_manager, graph
    try:
        system = await setup_system()
        chat_manager = system["chat_manager"]
        graph = system["graph"]
        logger.info("✅ Legal assistant system initialized for FastAPI")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise

app = FastAPI(title="Legal Assistant API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["Content-Type"], 
)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    await initialize_system()

def serialize_ai_message_chunk(chunk): 
    """Serialize AI message chunks for streaming"""
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )

async def generate_legal_chat_responses(message: str, session_id: Optional[str] = None):
    """Generate streaming responses for legal chat"""
    
    is_new_conversation = session_id is None
    
    if is_new_conversation:
        # Generate new session ID for first message
        session_id = f"api_{uuid4()}"
        logger.info(f"🆕 New conversation session: {session_id}")
        
        # First send the session ID
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
    else:
        logger.info(f"🔄 Continuing session: {session_id}")

    try:
        # Prepare input state
        input_state = {
            "messages": [{"role": "user", "content": message, "meta": {}}],
            "legal_context": {
                "jurisdiction": "Unknown", 
                "user_type": "general", 
                "document_type": "legal",
                "detected_country": "unknown"
            },
            "session_id": session_id,
            "router_decision": None,
            "search_results": None,
            "route_explanation": None,
            "last_search_query": None,
            "detected_articles": [],
        }

        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        # Stream events from the graph
        events = graph.astream_events(
            MultiCountryLegalState(**input_state),
            version="v2",
            config=config
        )

        current_content = ""
        current_node = ""
        search_in_progress = False

        async for event in events:
            event_type = event["event"]
            node_name = event.get("name", "")
            
            # Track node transitions for debugging
            if node_name != current_node:
                current_node = node_name
                yield f"data: {json.dumps({'type': 'node_transition', 'node': node_name})}\n\n"

            if event_type == "on_chat_model_stream":
                # Stream LLM content
                chunk_content = serialize_ai_message_chunk(event["data"]["chunk"])
                current_content += chunk_content
                yield f"data: {json.dumps({'type': 'content', 'content': chunk_content})}\n\n"
                
            elif event_type == "on_chat_model_end":
                # LLM response complete
                yield f"data: {json.dumps({'type': 'content_end'})}\n\n"
                
            elif event_type == "on_chain_start" and "retrieval" in node_name:
                # Search starting
                search_in_progress = True
                country = node_name.replace("_retrieval", "")
                yield f"data: {json.dumps({'type': 'search_start', 'country': country})}\n\n"
                
            elif event_type == "on_chain_end" and "retrieval" in node_name:
                # Search completed
                search_in_progress = False
                country = node_name.replace("_retrieval", "")
                yield f"data: {json.dumps({'type': 'search_end', 'country': country})}\n\n"
                
            elif event_type == "on_tool_end":
                # Tool execution completed (like email sending)
                tool_name = event["name"]
                yield f"data: {json.dumps({'type': 'tool_complete', 'tool': tool_name})}\n\n"

            elif event_type == "on_graph_end":
                # Graph execution completed
                yield f"data: {json.dumps({'type': 'graph_end'})}\n\n"

    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    # Send final end event
    yield f"data: {json.dumps({'type': 'end'})}\n\n"

@app.get("/chat")
async def chat_stream(
    message: str = Query(..., description="User message"),
    session_id: Optional[str] = Query(None, description="Existing session ID")
):
    """Streaming chat endpoint for legal assistant"""
    return StreamingResponse(
        generate_legal_chat_responses(message, session_id), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_initialized": chat_manager is not None and graph is not None,
        "service": "Legal Assistant API"
    }

@app.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if not chat_manager:
        return {"error": "System not initialized"}
    
    try:
        history = await chat_manager.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "history": [
                {
                    "role": msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown'),
                    "content": msg.content if hasattr(msg, 'content') else msg.get('content', ''),
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in history
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (for cleanup)"""
    # Note: With PostgreSQL checkpoints, sessions persist in database
    # This would require custom implementation to clear checkpoints
    return {"message": "Session deletion would require custom checkpoint cleanup"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)