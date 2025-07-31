# 🤖 EduBot - Nigerian Educational AI Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green.svg)](https://fastapi.tiangolo.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12.52-orange.svg)](https://docs.llamaindex.ai/)

A Retrieval-Augmented Generation (RAG) chatbot system (currently in MVP) specifically designed for Nigerian educational institutions. EduBot provides intelligent, context-aware responses about university admissions, course requirements, and institutional information through a clean, modular architecture.

## ✨ Key Features

- 🧠 **Intelligent Query Routing** - Automatically determines when to use knowledge base vs general knowledge
- 💬 **Conversation Memory** - Maintains session-based context across conversations with file persistence
- 📚 **Multi-Format Ingestion** - Processes PDFs, URLs, and directories into searchable knowledge
- 🌐 **FastAPI Backend** - Modern REST API with automatic OpenAPI documentation at `/docs`
- 🔄 **Session Persistence** - Conversations survive bot restarts and are saved to JSON files
- 🎯 **Natural Language** - Human-like responses without technical jargon
- 🔍 **AutoMerging Retrieval** - Hierarchical document chunking for comprehensive answers
- 🏗️ **Modular Architecture** - Clean separation of concerns for maintainability

## 📁 Project Structure

```
edu-chatbot/
├── src/                          # 📦 Core Application Code
│   ├── ingestion/               # 📥 Document Processing Pipeline
│   │   ├── __init__.py
│   │   ├── ingest.py            # Main ingestion orchestrator
│   │   ├── loaders.py           # PDF, URL, and file loaders
│   │   ├── processors.py        # Text processing and chunking
│   │   ├── metadata_extractor.py # AI-powered metadata extraction
│   │   ├── metadata_schema.py   # Pydantic document schemas
│   │   ├── storage.py           # Vector database operations
│   │   └── document_updater.py  # Document update management
│   │
│   ├── chat_system/             # 💬 Conversational AI Engine
│   │   ├── __init__.py
│   │   ├── bot.py               # Main EduBot orchestrator
│   │   ├── router.py            # Smart query classification & routing
│   │   ├── retrieval.py         # Vector search and ranking
│   │   ├── query_documents.py   # RAG query processing
│   │   └── session.py           # Session management & persistence
│   │
│   ├── config.py                # 🔧 Global configuration
│   ├── api.py                   # 🌐 FastAPI web server
│   ├── chat.py                  # 💻 CLI chat interface
│   ├── ingest.py                # 📥 CLI ingestion tool
│   └── clear_kb.py              # 🗑️ Knowledge base cleanup
│
├── data/                        # 📊 Source Documents
│
├── storage/                     # 🗄️ Persistent Data
│   ├── qdrant_db/              # Vector database files
│   └── parent_docstore          # Document metadata store
│
├── conversations/               # 💾 Session Storage
│   └── conversation_*.json     # Saved conversation sessions
│
├── tests/                       # 🧪 Test Suite
│   ├── test_chat.py            # Chat system tests
│   ├── test_ingestion.py       # Document processing tests
│   ├── test_fastapi.py         # API endpoint tests
│   └── test_integration.py     # End-to-end tests
│
├── requirements.txt             # 📋 Python dependencies (9 top-level packages)
├── .env                         # 🔐 Environment variables
├── .gitignore                  # 📝 Git ignore rules
└── README.md                   # 📖 This documentation
```

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone and setup
git clone https://github.com/PrechyDev/edu-chatbot.git
cd edu-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (only 9 top-level packages!)
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

> **Get your Google Gemini API key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to generate your free API key.

### 3. Document Ingestion

```bash
# Ingest a single PDF document
python -m src.ingest data/handbook.pdf

# Ingest all documents in a directory
python -m src.ingest data/

# Ingest from URL with metadata
python -m src.ingest https://university.edu/admissions.pdf --metadata institution="University of Lagos"

# Check ingestion status
python -m src.ingest --help
```

### 4. Start Using EduBot

#### Option A: Interactive Chat (CLI)
```bash
# Start interactive chat session
python -m src.chat

# Ask a direct question
python -m src.chat "What are the admission requirements for Computer Science?"
```

#### Option B: REST API Server
```bash
# Start FastAPI server
python -m src.api

# Or with custom settings
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Access the API:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

#### Option C: Direct API Calls
```bash
# Health check
curl http://localhost:8000/health

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about OAU programs", "session_id": "user123"}'

# List conversation sessions
curl http://localhost:8000/sessions
```

## 🏗️ System Architecture

### Document Ingestion Pipeline
1. **Document Loading**: PDF parsing, URL crawling, directory scanning
2. **Content Merging**: Multi-page documents combined for context
3. **Metadata Extraction**: AI-powered document analysis with schema validation
4. **Hierarchical Chunking**: Multi-level text splitting (6144, 3072, 1536 tokens)
5. **Vector Embedding**: Google GenAI embeddings with batch processing
6. **Storage**: Qdrant vector database with persistent document store

### Conversational AI Engine
1. **Query Router**: Intelligent classification (greeting, educational, institutional, general)
2. **Session Manager**: File-based conversation persistence with automatic saving
3. **Retrieval System**: AutoMerging hierarchical search with result ranking
4. **Response Generator**: Context-aware LLM responses with fallback strategies
5. **Memory Management**: Conversation context tracking across sessions

### FastAPI Web Service
- **Auto-generated Documentation**: OpenAPI/Swagger UI at `/docs`
- **CORS Support**: Cross-origin requests for frontend integration
- **Error Handling**: Graceful HTTP error responses with detailed messages
- **Session Management**: RESTful session endpoints for conversation control
- **Health Monitoring**: System status and diagnostics endpoint

## 💡 Usage Examples

### Educational Queries
```bash
# Ask about specific programs
"What programs does University of Ibadan offer?"

# Admission requirements
"What are the UTME requirements for Computer Science at UNILAG?"

# General educational guidance
"How do I prepare for university admission in Nigeria?"

# Follow-up questions (with conversation memory)
"What about the direct entry requirements?"
"How many O-Level credits do I need?"
```

### System Commands (CLI)
```bash
# Check system status
> status

# Clear conversation history
> clear

# Exit chat
> quit
```

### API Integration Example
```python
import requests

# Initialize conversation
response = requests.post('http://localhost:8000/chat', json={
    "message": "Hello, I need help with university admission",
    "session_id": "student_123"
})

print(response.json()['response'])

# Continue conversation (maintains context)
response = requests.post('http://localhost:8000/chat', json={
    "message": "What about Computer Science requirements?",
    "session_id": "student_123"
})
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_API_KEY=your_api_key_here

# Optional (with defaults)
QDRANT_DB_PATH=storage/qdrant_db
QDRANT_COLLECTION_NAME=edu_documents
PARENT_DOCSTORE_DIR=storage/parent_docstore
METADATA_EXTRACTOR_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/embedding-001
```

### Advanced Configuration
Edit [`src/config.py`](src/config.py) for detailed settings:
- Chunk sizes and overlap
- Retrieval parameters
- Response generation settings
- Session management options

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suites
python -m pytest tests/test_chat.py          # Chat system tests
python -m pytest tests/test_ingestion.py    # Document processing tests
python -m pytest tests/test_fastapi.py      # API endpoint tests
python -m pytest tests/test_integration.py  # End-to-end tests

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📊 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 1GB storage
- Internet connection (for Google Gemini API)

### Recommended Production Setup
- Python 3.11+
- 8GB RAM
- 5GB storage
- Docker deployment
- Load balancer for API scaling

### Dependencies (9 Top-Level Packages)
```txt
beautifulsoup4==4.13.4    # HTML parsing
fastapi==0.116.1          # Web framework
llama_index==0.12.52      # RAG framework
pydantic==2.11.7          # Data validation
pytest==8.4.1             # Testing
python-dotenv==1.1.1      # Environment management
qdrant_client==1.15.0     # Vector database
requests==2.32.4          # HTTP client
uvicorn==0.35.0           # ASGI server
```

## 🚀 Deployment

### Local Development
```bash
# Development server with auto-reload
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Production server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Follow the existing architecture**:
   - Keep `ingestion`, `chat_system`, and `api` domains separate
   - Add tests for new functionality
   - Update documentation
4. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to new functions
- Include docstrings for public methods
- Write tests for new features
- Update documentation for API changes

## 🔮 Roadmap

### Phase 1: Core Features (MVP) ✅
- [x] Document ingestion pipeline
- [x] RAG-based chat system
- [x] Session management
- [x] FastAPI backend
- [x] Comprehensive testing

### Phase 2: Enhanced Features 🚧
- [ ] Web UI (React frontend)
- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] Plugin system for extensions
- [ ] Bulk document management

### Phase 3: Enterprise Features 📋
- [ ] User authentication
- [ ] Role-based access control
- [ ] API rate limiting
- [ ] Monitoring and logging
- [ ] Performance optimization

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# If you get module import errors, ensure you're running from project root:
python -m src.chat    # ✅ Correct
python src/chat.py    # ❌ May cause import issues
```

**API Key Issues**
```bash
# Verify your .env file has the correct API key
cat .env | grep GOOGLE_API_KEY

# Test API key validity
python -c "import os; from google.genai import generative_models; print('API key valid' if os.getenv('GOOGLE_API_KEY') else 'API key missing')"
```

**Vector Database Issues**
```bash
# Clear and rebuild knowledge base
python -m src.clear_kb
python -m src.ingest data/
```

**Memory Issues**
- Reduce chunk sizes in `src/config.py`
- Process documents individually instead of in batches
- Increase system RAM or use cloud deployment

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://docs.llamaindex.ai/) for the RAG framework
- [Google Gemini](https://ai.google.dev/) for language model capabilities
- [Qdrant](https://qdrant.tech/) for vector database technology
- [FastAPI](https://fastapi.tiangolo.com/) for the modern web framework

---

**Built with ❤️ for Nigerian Education**

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/PrechyDev/edu-chatbot) or open an issue.
