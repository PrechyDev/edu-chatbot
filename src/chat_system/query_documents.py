#!/usr/bin/env python3
"""
Enhanced query script with AutoMerging Retriever support.
"""

import argparse
import logging
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
import qdrant_client

from .router import QueryRouter

from ..config import (
    GOOGLE_API_KEY,
    METADATA_EXTRACTOR_MODEL,
    EMBEDDING_MODEL,
    QUERY_MODEL,
    PARENT_DOCSTORE_DIR,
    QDRANT_DB_PATH,
    QDRANT_COLLECTION_NAME,
)

class QueryRouter:
    """Smart router to determine whether to use RAG or general knowledge."""
    
    def __init__(self):
        self.greetings = {
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'greetings', 'howdy', 'what\'s up', 'how are you', 'how do you do'
        }
        
        self.general_topics = {
            'python', 'java', 'javascript', 'c++', 'c programming', 'programming language',
            'programming', 'software development', 'artificial intelligence',
            'machine learning', 'data science', 'web development', 'mobile development',
            'career advice', 'study tips', 'time management', 'general education',
            'algorithms', 'data structures', 'coding', 'software engineering'
        }
        
        self.institutional_keywords = {
            'university', 'college', 'admission', 'requirement', 'program', 'course',
            'degree', 'bachelor', 'master', 'phd', 'transcript', 'gpa', 'scholarship',
            'tuition', 'fee', 'campus', 'dormitory', 'registration', 'semester',
            'academic', 'faculty', 'department', 'curriculum', 'syllabus'
        }
    
    def should_use_rag(self, question: str) -> tuple[bool, str]:
        """Determine if question should use RAG system or general knowledge."""
        question_lower = question.lower().strip()
        
        # Check for greetings first (more comprehensive detection)
        greeting_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'greetings', 'howdy', 'what\'s up', 'how are you', 'how do you do',
            'hi there', 'hello there', 'hey there'
        ]
        
        for greeting in greeting_patterns:
            if (question_lower == greeting or 
                question_lower.startswith(f"{greeting}!") or 
                question_lower.startswith(f"{greeting}.") or
                question_lower.startswith(f"{greeting} ") or
                question_lower.startswith(f"{greeting},") or
                greeting in question_lower and len(question.split()) <= 6):  # Short greeting-like questions
                return False, "greeting"
        
        # Check for general programming/tech topics BEFORE institutional keywords
        if any(topic in question_lower for topic in self.general_topics):
            return False, "general_knowledge"
        
        # Check for institutional keywords (high priority for RAG)
        if any(keyword in question_lower for keyword in self.institutional_keywords):
            return True, "institutional"
        
        # Default to RAG for educational context
        if len(question.split()) > 2:  # More complex questions likely need RAG
            return True, "educational"
        
        return False, "simple"
    
    def generate_unified_response(self, question: str, rag_response: str) -> str:
        """Generate a unified response combining limited RAG info with general knowledge."""
        try:
            llm = GoogleGenAI(
                model=QUERY_MODEL,  # Use the dedicated query model
                api_key=GOOGLE_API_KEY,
                temperature=0.3
            )
            
            prompt = f"""The user asked: "{question}"

I have some specific information from an educational institution: "{rag_response}"

Please create a single, unified response that:
1. Starts with the specific institutional information (if useful)
2. Seamlessly transitions to helpful general advice about the topic
3. Maintains a natural, conversational flow without mentioning "knowledge base", "documents", or "provided information"
4. Gives practical next steps or recommendations

Make it feel like one cohesive answer from a knowledgeable educational advisor, not two separate responses.

Question: {question}

Unified Response:"""
            
            response = llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            # Fallback to just the RAG response if unified generation fails
            if rag_response:
                return rag_response
            else:
                return f"I'd be happy to help with your question about {question.lower()}, but I'm having some technical difficulties right now. I'd recommend checking the official website or contacting the institution directly for the most current information."

    def generate_fallback_response(self, question: str, rag_attempted: bool = False) -> str:
        """Generate a helpful fallback response using general knowledge."""
        try:
            llm = GoogleGenAI(
                model=QUERY_MODEL,  # Use the dedicated query model
                api_key=GOOGLE_API_KEY,
                temperature=0.3
            )
            
            if rag_attempted:
                prompt = f"""The user asked about: "{question}"

I don't have specific information about this in my educational knowledge base. Please provide a helpful general response about this topic, and suggest where they might find more detailed, current information. 

Be conversational and helpful, as if you're a knowledgeable educational advisor. Never mention "knowledge base", "documents", or "provided information" - just speak naturally.

If it's about a specific institution, suggest they check that institution's official website or contact them directly.

Question: {question}

Response:"""
            else:
                prompt = f"""Please provide a helpful, conversational response to this question: "{question}"

Be natural and friendly, as if you're a knowledgeable educational advisor speaking directly to the person.

Question: {question}

Response:"""
            
            response = llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"I'd be happy to help with your question about {question.lower()}, but I'm having some technical difficulties right now. I'd recommend checking the official website or contacting the institution directly for the most current information."

def load_automerging_query_engine(use_automerging=True, top_k=12):
    """Load the existing vector store and create an AutoMerging query engine."""
    
    try:
        # Initialize models with balanced token management for better retrieval
        llm = GoogleGenAI(
            model=QUERY_MODEL,  # Use the dedicated query model
            api_key=GOOGLE_API_KEY,
            temperature=0.1,  # Lower temperature for more factual responses
            max_tokens=1024   # Increased to handle better retrieval results
        )
        embed_model = GoogleGenAIEmbedding(model=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
        
        # Load existing vector store with proper connection management
        # Note: Don't close client here - it's needed by the query engine
        client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)
        vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME)
        
        # Load document store (contains ALL nodes including parents)
        docstore = SimpleDocumentStore.from_persist_path(PARENT_DOCSTORE_DIR)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore
        )
        
        # Load the index (contains only leaf nodes)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        if use_automerging:
            # Create base retriever with higher retrieval count for better coverage
            base_retriever = index.as_retriever(
                similarity_top_k=min(top_k, 15),  # Increased to 15 for better coverage
                verbose=False
            )
            
            # Create AutoMerging retriever
            auto_merging_retriever = AutoMergingRetriever(
                base_retriever,
                storage_context,
                verbose=False  # Disable verbose to reduce token overhead
            )
            
            # Create more effective prompt template for information extraction
            chat_prompt_template = PromptTemplate(
                """You are EduBot for Nigerian educational institutions.

Context information from the university documents:
{context_str}

Instructions: 
- Search the context carefully for specific information about the question
- If you find relevant details, provide them fully
- If the context mentions related information, include that too
- Be specific about course codes, requirements, and numbers when available

Question: {query_str}

Answer based on the context:"""
            )
            
            # Create query engine with custom prompt
            query_engine = RetrieverQueryEngine.from_args(
                retriever=auto_merging_retriever,
                llm=llm,
                text_qa_template=chat_prompt_template,
                verbose=False  # Reduce technical output for chat
            )
        else:
            # Create base retriever for comparison
            base_retriever = index.as_retriever(
                similarity_top_k=top_k,
                verbose=False
            )
            
            query_engine = RetrieverQueryEngine.from_args(
                retriever=base_retriever,
                llm=llm,
                verbose=False
            )
        
        return query_engine, index
        
    except Exception as e:
        print(f"❌ Failed to load query engine: {e}")
        print("Make sure you have run ingestion first: python run_ingestion.py data/")
        return None, None

def smart_query(question: str, top_k: int = 15) -> str:
    """
    Smart query function that routes to RAG or general knowledge based on query type.
    Returns the response text without printing it.
    """
    router = QueryRouter()
    use_rag, query_type = router.should_use_rag(question)
    
    print(f"🔍 Query: {question}")
    print(f"🧠 Query Type: {query_type}")
    print(f"📚 Using Knowledge Base: {'Yes' if use_rag else 'No'}")
    print("-" * 50)
    
    if not use_rag:
        # Handle greetings and general questions without RAG
        print("💭 Using general knowledge...")
        response_text = router.generate_fallback_response(question, rag_attempted=False)
        return response_text
    
    # Use RAG system for institutional queries with reduced retrieval count
    print("📖 Searching knowledge base...")
    query_engine, index = load_automerging_query_engine(use_automerging=True, top_k=top_k)
    
    if not query_engine:
        # Fallback if RAG system fails
        print("⚠️ Knowledge base unavailable, using general knowledge...")
        response_text = router.generate_fallback_response(question, rag_attempted=True)
        return response_text
    
    try:
        # Query the RAG system
        response = query_engine.query(question)
        response_text = response.response
        
        # Check if response seems insufficient (very short or generic)
        if len(response_text.strip()) < 50 or "I don't have" in response_text or "not found" in response_text.lower():
            print("ℹ️ Limited information found, creating enhanced response...")
            # Create a unified response using both specific info and general knowledge
            unified_response = router.generate_unified_response(question, response_text.strip())
            response_text = unified_response
        
        return response_text
        
    except Exception as e:
        error_msg = str(e)
        if "MAX_TOKENS" in error_msg or "token" in error_msg.lower():
            print(f"⚠️ Found relevant information but response too long. Providing focused answer...")
            # When we hit token limits, it means we found relevant content
            # Let the router generate an appropriate response acknowledging we found info
            response_text = router.generate_fallback_response(question, rag_attempted=True)
            return response_text
        
        print(f"❌ Query failed: {e}")
        print("⚠️ Using fallback response...")
        response_text = router.generate_fallback_response(question, rag_attempted=True)
        return response_text

def query_documents(question: str, use_automerging: bool = True, top_k: int = 15):
    """Query the document collection with optional AutoMerging."""
    
    print(f"🔍 Querying: {question}")
    print(f"🤖 Using AutoMerging: {'Yes' if use_automerging else 'No'}")
    print(f"📊 Retrieving top {top_k} chunks from knowledge base")
    print("-" * 50)
    
    # Load query engine
    query_engine, index = load_automerging_query_engine(use_automerging, top_k)
    if not query_engine:
        return
    
    try:
        # Execute query
        response = query_engine.query(question)
        
        # Print clean response for chat assistant
        print(f"💬 Response:")
        print(response.response)
        print()
        
        return response
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return None

def compare_retrieval_methods(question: str, top_k: int = 25):
    """Compare AutoMerging vs Base retrieval for the same question."""
    
    print(f"🔬 Comparing Retrieval Methods for: {question}")
    print("=" * 60)
    print()
    
    # Test WITH AutoMerging
    print("🤖 WITH AutoMerging:")
    print("-" * 30)
    auto_response = query_documents(question, use_automerging=True, top_k=top_k)
    print()
    
    # Test WITHOUT AutoMerging  
    print("🔍 WITHOUT AutoMerging (Base Retrieval):")
    print("-" * 40)
    base_response = query_documents(question, use_automerging=False, top_k=top_k)
    print()
    
    return auto_response, base_response

def interactive_mode():
    """Start interactive query mode with smart routing."""
    print("🤖 EduBot Interactive Mode (Smart Routing)")
    print("Commands:")
    print("  - Type your question normally (smart routing)")
    print("  - Type 'rag: <question>' to force RAG system")
    print("  - Type 'compare: <question>' to compare methods")
    print("  - Type 'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
            
            if user_input.startswith('rag:'):
                question = user_input[4:].strip()
                query_documents(question, use_automerging=True)
            elif user_input.startswith('compare:'):
                question = user_input[8:].strip()
                compare_retrieval_methods(question)
            else:
                # Use smart routing by default
                smart_query(user_input)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Query EduBot with smart routing')
    parser.add_argument('question', nargs='?', help='Question to ask (if not provided, starts interactive mode)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive mode')
    parser.add_argument('--base', '-b', action='store_true', help='Use base retrieval instead of AutoMerging')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare both methods')
    parser.add_argument('--rag-only', '-r', action='store_true', help='Force RAG system (bypass smart routing)')
    parser.add_argument('--top-k', '-k', type=int, default=25, help='Number of chunks to retrieve (default: 25)')
    
    args = parser.parse_args()
    
    # Setup logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    if args.interactive or not args.question:
        interactive_mode()
    elif args.compare:
        compare_retrieval_methods(args.question, args.top_k)
    elif args.rag_only:
        if args.base:
            query_documents(args.question, use_automerging=False, top_k=args.top_k)
        else:
            query_documents(args.question, use_automerging=True, top_k=args.top_k)
    else:
        # Use smart routing by default
        smart_query(args.question, args.top_k)

if __name__ == "__main__":
    main()