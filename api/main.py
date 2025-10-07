# api/main.py
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
import json
from uuid import uuid4
import logging
import os
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
system_initialized = False


async def initialize_system():
    global chat_manager, graph, system_initialized
    try:
        # Check for required environment variables based on YOUR settings
        required_vars = ['OPENAI_API_KEY', 'MONGO_URI', 'NEON_DB_URL', 'NEON_END_POINT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
            logger.warning("System will start but may not function properly")
        
        system = await setup_system()
        chat_manager = system["chat_manager"]
        graph = system["graph"]
        system_initialized = True
        logger.info("✅ Legal assistant system initialized for Hugging Face")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        system_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler"""
    # Startup logic
    logger.info("🚀 Starting Legal Assistant API...")
    
    # Initialize system in background
    initialization_task = asyncio.create_task(initialize_system())
    
    yield  # App runs here
    
    # Shutdown logic
    logger.info("🛑 Shutting down Legal Assistant API...")
    initialization_task.cancel()
    try:
        await initialization_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Legal Assistant API",
    version="1.0.0",
    description="Multi-country legal RAG system for Benin and Madagascar",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Simple homepage for better UX"""
    return """
    <html>
        <head>
            <title>Legal Assistant API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .card { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 8px; }
                .status-ready { color: green; }
                .status-starting { color: orange; }
                .status-error { color: red; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧑‍⚖️ Legal Assistant API</h1>
                <p>Multi-country legal RAG system for Benin and Madagascar</p>
                
                <div class="card">
                    <h3>📚 Available Endpoints</h3>
                    <ul>
                        <li><a href="/docs">API Documentation</a></li>
                        <li><a href="/health">Health Check</a></li>
                        <li><strong>GET /chat</strong> - Streaming chat</li>
                        <li><strong>GET /sessions/{id}/history</strong> - Conversation history</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>🔧 System Status</h3>
                    <div id="status">
                        <p>Loading system status...</p>
                    </div>
                </div>
                
                <script>
                    async function updateStatus() {
                        try {
                            const response = await fetch('/health');
                            const data = await response.json();
                            
                            const statusEl = document.getElementById('status');
                            let statusClass = 'status-starting';
                            let statusText = '🔄 Starting...';
                            
                            if (data.system_initialized) {
                                statusClass = 'status-ready';
                                statusText = '✅ System Ready';
                            } else if (data.status === 'error') {
                                statusClass = 'status-error';
                                statusText = '❌ System Error';
                            }
                            
                            statusEl.innerHTML = `
                                <p class="${statusClass}"><strong>${statusText}</strong></p>
                                <p><strong>MongoDB:</strong> ${data.mongodb_connected ? '✅ Connected' : '❌ Disconnected'}</p>
                                <p><strong>Countries:</strong> ${data.available_countries?.join(', ') || 'Loading...'}</p>
                                <p><strong>OpenAI:</strong> ${data.openai_configured ? '✅ Configured' : '❌ Not Configured'}</p>
                            `;
                        } catch (error) {
                            document.getElementById('status').innerHTML = 
                                '<p class="status-error">❌ Failed to load system status</p>';
                        }
                    }
                    
                    updateStatus();
                    setInterval(updateStatus, 5000);
                </script>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Enhanced health check with your specific environment variables"""
    return {
        "status": "healthy" if system_initialized else "starting",
        "system_initialized": system_initialized,
        "service": "Legal Assistant API",
        "available_countries": ["benin", "madagascar"] if system_initialized else [],
        "mongodb_connected": system_initialized and bool(os.getenv("MONGO_URI")),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "neon_postgres_configured": bool(os.getenv("NEON_END_POINT")),
        "missing_variables": [var for var in ['OPENAI_API_KEY', 'MONGO_URI', 'NEON_DB_URL', 'NEON_END_POINT'] if not os.getenv(var)],
    }

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
    if not system_initialized:
        yield f"data: {json.dumps({'type': 'error', 'message': 'System is still starting up. Please try again in a moment.'})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
        return
    
    is_new_conversation = session_id is None
    
    if is_new_conversation:
        session_id = f"api_{uuid4()}"
        logger.info(f"🆕 New conversation session: {session_id}")
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
    else:
        logger.info(f"🔄 Continuing session: {session_id}")

    try:
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

        events = graph.astream_events(
            MultiCountryLegalState(**input_state),
            version="v2",
            config=config
        )

        current_content = ""
        current_node = ""

        async for event in events:
            event_type = event["event"]
            node_name = event.get("name", "")
            
            if node_name != current_node:
                current_node = node_name
                yield f"data: {json.dumps({'type': 'node_transition', 'node': node_name})}\n\n"

            if event_type == "on_chat_model_stream":
                chunk_content = serialize_ai_message_chunk(event["data"]["chunk"])
                current_content += chunk_content
                yield f"data: {json.dumps({'type': 'content', 'content': chunk_content})}\n\n"
                
            elif event_type == "on_chat_model_end":
                yield f"data: {json.dumps({'type': 'content_end'})}\n\n"
                
            elif event_type == "on_chain_start" and "retrieval" in node_name:
                country = node_name.replace("_retrieval", "")
                yield f"data: {json.dumps({'type': 'search_start', 'country': country})}\n\n"
                
            elif event_type == "on_chain_end" and "retrieval" in node_name:
                country = node_name.replace("_retrieval", "")
                yield f"data: {json.dumps({'type': 'search_end', 'country': country})}\n\n"
                
            elif event_type == "on_tool_end":
                tool_name = event["name"]
                yield f"data: {json.dumps({'type': 'tool_complete', 'tool': tool_name})}\n\n"

            elif event_type == "on_graph_end":
                yield f"data: {json.dumps({'type': 'graph_end'})}\n\n"

    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    yield f"data: {json.dumps({'type': 'end'})}\n\n"

@app.get("/chat")
async def chat_stream(
    message: str = Query(..., description="User message"),
    session_id: Optional[str] = Query(None, description="Existing session ID")
):
    """Streaming chat endpoint with initialization check"""
    if not system_initialized:
        raise HTTPException(
            status_code=503, 
            detail="System is still starting up. Please try again in a moment."
        )
    
    return StreamingResponse(
        generate_legal_chat_responses(message, session_id), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

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