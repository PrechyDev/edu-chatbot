
# EduBot: Conversational AI Assistant for the Nigerian Education System

## Overview
EduBot is a Retrieval-Augmented Generation (RAG) system designed to answer any question about the Nigerian education system. It ingests data from diverse sources (documents, tables, websites, PDFs, PPTs, etc.), extracts metadata, embeds content, retrieves relevant information using hybrid methods, reranks results, and supports conversational interactions.

## Features
- **Multi-format Data Ingestion:** Supports PDFs, PPTs, tables, websites, and more.
- **Hybrid Metadata Extraction:** Combines document parsing and LLM-based extraction.
- **Semantic Chunking:** Preserves context for better embeddings.
- **Gemini LLM Integration:** For both LLM and embedding tasks.
- **Hybrid Retrieval:** Combines sparse (keyword) and dense (embedding) retrieval using Qdrant.
- **Reranking:** Reranks results based on relevance, accuracy, similarity, metadata, keywords, recency, and time relevance.
- **Conversational Memory:** Maintains chat history for context-aware answers.
- **API:** RESTful API for chat, chat history retrieval, and deletion.
- **Local Chat Storage (MVP):** Stores chat history on the user's device, with future support for MongoDB.

## Architecture
- **Metadata Extractor:** Hybrid approach using both document structure and LLMs.
- **Embedder:** Uses semantic chunking and Gemini embeddings.
- **Retriever:** Hybrid retriever (sparse + dense) with Qdrant as the vector DB.
- **Reranker:** Reranks based on multiple criteria.
- **Conversational RAG:** Answers are context-aware, considering chat history.
- **API Layer:** Exposes endpoints for chat and history management.

## File Structure
```
edubot/
    app.py                # Main application entry point
    main.py               # Alternative entry point or script
    streamlit_app.py      # Streamlit dashboard (if any)
    src/
        chunking.py       # Semantic chunking logic
        embedder.py       # Embedding logic (Gemini)
        loader.py         # Data/document loaders
        metadata.py       # Metadata extraction logic
        rag_chain.py      # RAG orchestration
        reranker.py       # Reranking logic
requirements.txt          # Python dependencies
rag_chain.ipynb       # RAG chain experiments
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**
   - Add your Gemini API keys and Qdrant configuration as needed.

## Usage
- **Run the main app:**
  ```bash
  python edubot/app.py
  ```
- **Streamlit dashboard (if available):**
  ```bash
  streamlit run edubot/streamlit_app.py
  ```
- **API endpoints:**
  - `/chat` - Send a message and receive a response.
  - `/history` - Retrieve chat history.
  - `/delete` - Delete chat history.

## Roadmap
- [ ] Multi-format data ingestion
- [ ] Hybrid metadata extraction
- [ ] Semantic chunking
- [ ] Gemini LLM integration
- [ ] Hybrid retrieval (sparse + dense)
- [ ] Reranking
- [ ] Conversational memory
- [ ] RESTful API
- [ ] MongoDB integration for chat history
- [ ] User authentication
- [ ] Admin dashboard for uploads
- [ ] Advanced analytics and monitoring

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)

## Contact
For questions or support, please contact the maintainer.
