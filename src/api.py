"""
FastAPI wrapper for EduBot deployment.
Provides HTTP endpoints for frontend integration with automatic OpenAPI documentation.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.chat_system.bot import EduBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EduBot API",
    description="Nigerian Education Chatbot with conversation memory and RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    query_type: str
    confidence: float
    knowledge_base_used: bool
    sources_count: int
    session_id: str
    conversation_context: str
    message_count: int

class HealthResponse(BaseModel):
    status: str
    bot_available: bool
    timestamp: str
    knowledge_base_available: Optional[bool] = None
    active_sessions: Optional[int] = None

class SessionInfo(BaseModel):
    session_id: str
    created_at: Optional[str] = None
    message_count: int
    version: Optional[str] = None

class SessionListResponse(BaseModel):
    sessions: List[str]
    count: int

class ErrorResponse(BaseModel):
    error: str
    message: str

# Initialize EduBot
try:
    edubot = EduBot()
    logger.info("✅ EduBot initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize EduBot: {e}")
    edubot = None

@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Health check endpoint to verify bot status."""
    status_data = {
        'status': 'healthy' if edubot else 'unhealthy',
        'bot_available': edubot is not None,
        'timestamp': datetime.now().isoformat(),
    }
    
    if edubot:
        try:
            system_status = edubot.get_system_status()
            status_data.update(system_status)
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
    
    return HealthResponse(**status_data)

@app.post("/chat", response_model=ChatResponse, summary="Send Chat Message")
async def chat(request: ChatRequest):
    """
    Send a chat message to EduBot and get a response.
    
    - **message**: The user's message/question
    - **session_id**: Optional session identifier for conversation tracking
    """
    if not edubot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EduBot service is not available"
        )
    
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    try:
        # Process chat message
        response = edubot.chat(message, request.session_id)
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your request"
        )

@app.get("/sessions", response_model=SessionListResponse, summary="List Sessions")
async def list_sessions():
    """Get a list of all available conversation sessions."""
    if not edubot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EduBot service is not available"
        )
    
    try:
        sessions = edubot.list_sessions()
        return SessionListResponse(sessions=sessions, count=len(sessions))
    except Exception as e:
        logger.error(f"Sessions list error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/sessions/{session_id}", response_model=SessionInfo, summary="Get Session Info")
async def get_session_info(session_id: str):
    """Get detailed information about a specific conversation session."""
    if not edubot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EduBot service is not available"
        )
    
    try:
        session_info = edubot.get_session_info(session_id)
        if session_info:
            return SessionInfo(**session_info)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/sessions/{session_id}", summary="Clear Session")
async def clear_session(session_id: str):
    """Clear/delete a specific conversation session."""
    if not edubot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EduBot service is not available"
        )
    
    try:
        edubot.clear_conversation(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Session clear error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/status", summary="System Status")
async def system_status():
    """Get detailed system status including knowledge base and retrieval statistics."""
    if not edubot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EduBot service is not available"
        )
    
    try:
        status_info = edubot.get_system_status()
        return status_info
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/", summary="API Root")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "EduBot API",
        "version": "1.0.0",
        "description": "Nigerian Education Chatbot with conversation memory",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "endpoints": {
            "chat": "POST /chat",
            "sessions": "GET /sessions",
            "session_info": "GET /sessions/{session_id}",
            "clear_session": "DELETE /sessions/{session_id}",
            "status": "GET /status"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get('PORT', 8000))
    log_level = os.environ.get('LOG_LEVEL', 'info')
    
    logger.info(f"🚀 Starting EduBot FastAPI server on port {port}")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=os.environ.get('ENVIRONMENT', 'production') == 'development'
    )
