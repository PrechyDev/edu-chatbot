import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- Import yRAG pipeline dependencies ---
from src.rag_chain import build_rag_chain
from src.embedder import get_vector_store
from src.reranker import rerank_results
from langchain.chat_models import init_chat_model

# --- Add middleware for CORS ---
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Load environment variables from .env file
load_dotenv()

# --- Ensure required environment variables are set ---
if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY environment variable not set")
if not os.environ.get("LLM_MODEL"):
    os.environ["LLM_MODEL"] = "gemini-2.5-flash-lite-preview-06-17"
if not os.environ.get("LLM_PROVIDER"):
    os.environ["LLM_PROVIDER"] = "google_genai"
if not os.environ.get("QDRANT_URL"):
    raise RuntimeError("QDRANT_URL environment variable not set")
if not os.environ.get("QDRANT_API_KEY"):
    raise RuntimeError("QDRANT_API_KEY environment variable not set")

# --- Initialize FastAPI app ---
app = FastAPI(
    title="Sophia RAG API",
    description="An agentic RAG system for querying Nigerian education documents like JAMB brochures, handbooks, and more.",
    version="1.0.0"
)

# --- middleware setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses larger than 1000 bytes
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Adjust this in production
)

# --- SQLite DB Setup ---
DB_PATH = "chats.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
    chat_id TEXT,
    role TEXT,
    content TEXT,
    FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
)''')
conn.commit()

# --- Models ---
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatCreate(BaseModel):
    initial_question: str

class MessageInput(BaseModel):
    content: str

class ChatWithResponse(BaseModel):
    chat_id: str
    response: Message

class ChatSummary(BaseModel):
    chat_id: str
    title: str
    created_at: str

# --- Init RAG on startup ---
llm = init_chat_model(
    os.environ["LLM_MODEL"],
    model_provider=os.environ["LLM_PROVIDER"]
)
vector_store, _ = get_vector_store(
    client_url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    recreate=False
)
rag_chain = build_rag_chain(llm, vector_store, rerank_results)

# ROUTES

@app.post("/chats/", response_model=ChatWithResponse)
def create_chat(chat: ChatCreate):
    chat_id = str(uuid4())
    now = datetime.utcnow().isoformat()
    title = chat.initial_question[:40] + ("..." if len(chat.initial_question) > 40 else "")

    # Store chat + first user message
    cursor.execute("INSERT INTO chats (chat_id, title, created_at) VALUES (?, ?, ?)", (chat_id, title, now))
    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, "user", chat.initial_question))

    # Get assistant response via RAG
    result = rag_chain.invoke({"question": chat.initial_question})
    assistant_reply = result["answer"]

    # Store assistant response
    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, "assistant", assistant_reply))
    conn.commit()

    # Return both chat_id and the assistant's reply
    return ChatWithResponse(
        chat_id=chat_id,
        response=Message(role="assistant", content=assistant_reply)
    )


@app.get("/chats/{chat_id}/history", response_model=List[Message])
def get_chat_history(chat_id: str):
    cursor.execute("SELECT role, content FROM messages WHERE chat_id = ?", (chat_id,))
    rows = cursor.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Chat not found")
    return [Message(role=row[0], content=row[1]) for row in rows]

@app.get("/chats/list", response_model=List[ChatSummary])
def list_all_chats():
    cursor.execute("SELECT chat_id, title, created_at FROM chats ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return [ChatSummary(chat_id=row[0], title=row[1], created_at=row[2]) for row in rows]

@app.post("/chats/{chat_id}/message", response_model=Message)
def send_message(chat_id: str, message: MessageInput):
    cursor.execute("SELECT 1 FROM chats WHERE chat_id = ?", (chat_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Chat not found")

    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, "user", message.content))

    result = rag_chain.invoke({"question": message.content})
    assistant_reply = Message(role="assistant", content=result["answer"])
    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, assistant_reply.role, assistant_reply.content))
    conn.commit()
    return assistant_reply

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    return {"status": "cleared"}

@app.get("/")
def root():
    return {"message": "Welcome to the Sophia RAG API. Use /docs to test the endpoints."}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "The Sophia RAG API is healthy!"}

