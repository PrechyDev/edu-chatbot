# Description: Manages configuration for the application, such as API keys and file paths. No changes here.
# =================================================================

import os
from dotenv import load_dotenv

# Load environment variables from a .env file for security
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

# --- Model Names ---
METADATA_EXTRACTOR_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "embedding-001"
QUERY_MODEL = "gemini-2.5-flash"

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # edu-chatbot root
PARENT_DOCSTORE_DIR = os.path.join(PROJECT_ROOT, "storage", "parent_docstore")
QDRANT_DB_PATH = os.path.join(PROJECT_ROOT, "storage", "qdrant_db")

# --- Qdrant Configuration ---
QDRANT_COLLECTION_NAME = "edubot_documents"

# --- Chunking Configuration ---
PARENT_CHUNK_SIZE = 4096  # Increased to handle larger metadata
CHILD_CHUNK_SIZE = 1024   # Increased to handle larger metadata
