from src.loader import load_document
from src.chunking import split_documents
from src.embedder import get_vector_store, embed_documents
from src.metadata import process_docs_with_metadata
from src.reranker import rerank_results
from src.rag_chain import build_rag_chain
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Set required environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite-preview-06-17")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google_genai")
QDRANT_URL = os.getenv("QDRANT_URL")  # or your Qdrant URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def main():
    # 1. Load document(s)
    docs = []
    data_folder = "data"
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            docs.extend(load_document(file_path))

    # 3. Chunk and enrich with metadata
    chunks = split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    # 2. Process documents with metadata
    meta_chunks = process_docs_with_metadata(
        chunks,
        source_type="brochure",
        institution="JAMB/UTME",
        doc_type="general",
        origin="user_upload"
    )


    # 4. Set up vector store and embed
    vector_store, client = get_vector_store(client_url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embed_documents(meta_chunks, vector_store, client)

    # 5. Set up LLM and RAG chain
    llm = init_chat_model("gemini-2.5-flash-lite-preview-06-17", model_provider="google_genai")
    rag_chain = build_rag_chain(llm, vector_store, rerank_results)

    # 5. Run RAG chain for a question
    question = "What are the courses offered by OAU in faculty of Technology?"
    result = rag_chain.invoke({"question": question})
    print(result["answer"])

if __name__ == "__main__":
    main()

