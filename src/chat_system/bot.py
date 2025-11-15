"""
Main EduBot chat system for Nigerian education queries.
Integrates routing, retrieval, and response generation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from .router import QueryRouter, ConversationContext, QueryType
from .retrieval import RetrievalManager, ResultRanker
from .session import SessionManager
from src.config import (
    GOOGLE_API_KEY,
    METADATA_EXTRACTOR_MODEL,
    EMBEDDING_MODEL,
    PARENT_DOCSTORE_DIR,
    QDRANT_DB_PATH,
    QDRANT_COLLECTION_NAME,
)
import nest_asyncio

nest_asyncio.apply()

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates contextual responses using retrieved information."""
    
    def __init__(self, llm: GoogleGenAI):
        self.llm = llm
        logger.info("ResponseGenerator initialized")
    
    def generate_knowledge_based_response(self, query: str, retrieved_nodes: List,
                                        context: str = "") -> str:
        """Generate response using retrieved knowledge, with fallback to general knowledge."""
        if not retrieved_nodes:
            # No relevant knowledge found - use general knowledge with source recommendations
            return self._generate_general_knowledge_with_sources(query, context)
        
        # Check if retrieved nodes are actually relevant
        if not self._nodes_are_relevant(retrieved_nodes, query):
            # Nodes exist but aren't relevant - combine limited KB info with general knowledge
            return self._generate_hybrid_response(query, retrieved_nodes, context)
        
        # Prepare context from retrieved nodes
        knowledge_context = self._prepare_knowledge_context(retrieved_nodes)
        
        # Create prompt for knowledge-based response
        prompt = self._create_knowledge_prompt(query, knowledge_context, context)
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate knowledge-based response: {e}")
            # Fallback: Present retrieved information directly without LLM
            return self._generate_direct_knowledge_response(query, retrieved_nodes, context)
    
    def generate_general_response(self, query: str, query_type: QueryType,
                                context: str = "") -> str:
        """Generate response for non-knowledge-base queries."""
        
        if query_type == QueryType.GREETING:
            return self._generate_greeting_response()
        elif query_type == QueryType.SIMPLE:
            return self._generate_chitchat_response(query)
        else:
            return self._generate_general_knowledge_response(query, context)
    
    def _prepare_knowledge_context(self, nodes: List) -> str:
        """Prepare context text from retrieved nodes."""
        contexts = []
        for i, node in enumerate(nodes[:3]):  # Use top 3 most relevant
            context_text = node.node.text[:1000]  # Limit context length
            contexts.append(f"Context {i+1}: {context_text}")
        
        return "\n\n".join(contexts)
    
    def _create_knowledge_prompt(self, query: str, knowledge_context: str, 
                               conversation_context: str) -> str:
        """Create prompt for knowledge-based response."""
        prompt = f"""You are EduBot, a helpful assistant specializing in Nigerian education. 
You have access to authoritative educational documents and information from universities.

{conversation_context}

User Query: {query}

Relevant Information:
{knowledge_context}

Instructions:
1. Answer the query directly using the information available
2. Present the information naturally as if you know it, without mentioning "provided text" or "documents"
3. Be specific and cite details like course codes, requirements, or procedures when available
4. If the information covers the question completely, provide a comprehensive answer
5. If the information is partial, supplement with general knowledge while being clear about what's specific vs general
6. Keep responses helpful, clear, and focused on Nigerian education
7. When appropriate, suggest checking with specific institutions for the most current information

Response:"""
        return prompt
    
    def _generate_greeting_response(self) -> str:
        """Generate a greeting response."""
        return ("Hello! I'm Sophia, your helpful assistant for Nigerian education. "
                "I can help you with questions about universities, admissions, courses, "
                "educational policies, and more. What would you like to know?")
    
    def _generate_chitchat_response(self, query: str) -> str:
        """Generate response for chitchat queries."""
        try:
            prompt = f"""You are Sophia, a friendly educational assistant. 
Respond briefly and kindly to this message, then gently redirect to educational topics.

User message: {query}

Keep your response warm but brief, and suggest how you can help with education."""
            
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate chitchat response: {e}")
            return "Thank you! Is there anything about Nigerian education I can help you with?"
    
    def _generate_general_knowledge_response(self, query: str, context: str) -> str:
        """Generate response using general knowledge."""
        try:
            prompt = f"""You are Sophia, an educational assistant specializing in Nigerian education.
The user asked about something that might not be in your specific knowledge base.

{context}

User Query: {query}

Provide a helpful response using your general knowledge, but:
1. Focus on educational aspects when possible
2. If it's about Nigerian education specifically, acknowledge any limitations
3. Suggest where they might find more specific/official information
4. Keep it concise and helpful

Response:"""
            
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate general response: {e}")
            return self._generate_fallback_response(query)
    
    def _generate_no_knowledge_response(self, query: str) -> str:
        """Generate response when no relevant knowledge is found."""
        return (f"I don't have specific information about that in my current knowledge base. "
                f"For the most accurate and up-to-date information, I recommend checking "
                f"with the relevant educational institution directly or visiting official "
                f"websites like NUC, JAMB, or the specific university's official site.")
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when other methods fail."""
        return ("I'm sorry, I'm having trouble processing your request right now. "
                "Please try rephrasing your question or ask about a specific educational topic.")

    def _generate_general_knowledge_with_sources(self, query: str, context: str) -> str:
        """Generate response using general knowledge and suggest better sources."""
        try:
            prompt = f"""You are Sophia, an educational assistant specializing in Nigerian education.
The user asked a question that's not covered in your specific knowledge base, but you can provide helpful general information.

{context}

User Query: {query}

Instructions:
1. Provide a helpful response using your general knowledge about education
2. Be honest that this information is general and not from your specific knowledge base
3. Focus on educational aspects relevant to Nigeria when possible
4. Always suggest specific, authoritative sources where they can get more accurate information
5. Keep it helpful and encouraging

Format your response like this:
- First, provide helpful general information
- Then suggest specific sources like: "For more specific and up-to-date information, I recommend checking with [specific sources]"

Response:"""
            
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate general knowledge response: {e}")
            return (f"I don't have specific information about that in my knowledge base, "
                   f"but I'd recommend checking with the relevant educational institutions, "
                   f"NUC (National Universities Commission), JAMB, or official university websites "
                   f"for the most accurate information.")

    def _generate_hybrid_response(self, query: str, retrieved_nodes: List, context: str) -> str:
        """Generate response combining limited KB info with general knowledge."""
        try:
            # Get limited context from KB
            kb_context = self._prepare_knowledge_context(retrieved_nodes)
            
            prompt = f"""You are Sophia, an educational assistant specializing in Nigerian education.
You have some limited information from your knowledge base, but it doesn't fully answer the user's question.

{context}

User Query: {query}

Limited Information from Knowledge Base:
{kb_context}

Instructions:
1. Acknowledge what information you do have from your knowledge base
2. Provide additional helpful information using your general knowledge
3. Be clear about what comes from your KB vs general knowledge
4. Suggest specific sources for more comprehensive information
5. Keep it helpful and informative

Response:"""
            
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate hybrid response: {e}")
            return self._generate_general_knowledge_with_sources(query, context)

    def _nodes_are_relevant(self, nodes: List, query: str, min_score: float = 0.3) -> bool:
        """Check if retrieved nodes are actually relevant to the query."""
        if not nodes:
            return False
        
        # Check if any node has a decent relevance score
        for node in nodes[:3]:  # Check top 3 nodes
            if hasattr(node, 'score') and node.score >= min_score:
                return True
        
        # If no score info, do a simple text relevance check
        query_words = set(query.lower().split())
        for node in nodes[:2]:  # Check top 2 nodes
            node_text = node.node.text.lower() if hasattr(node, 'node') else str(node).lower()
            node_words = set(node_text.split())
            
            # If there's reasonable word overlap, consider it relevant
            overlap = len(query_words.intersection(node_words))
            if overlap >= min(2, len(query_words) // 2):
                return True
        
        return False

    def _generate_direct_knowledge_response(self, query: str, retrieved_nodes: List, context: str) -> str:
        """Generate response directly from retrieved nodes without LLM when API is unavailable."""
        if not retrieved_nodes:
            return "I found relevant information but cannot process it right now. Please try again later."
        
        # Extract relevant text snippets from nodes
        relevant_snippets = []
        for i, node in enumerate(retrieved_nodes[:3]):  # Use top 3 nodes
            text = node.node.text if hasattr(node, 'node') else str(node)
            
            # Extract key phrases related to the query
            query_words = set(query.lower().split())
            sentences = text.split('.')
            
            relevant_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Skip very short sentences
                    sentence_words = set(sentence.lower().split())
                    # Check if sentence contains query-related words
                    if query_words.intersection(sentence_words) or \
                       any(word in sentence.lower() for word in ['requirement', 'course', 'subject', 'admission', 'utme', 'jamb']):
                        relevant_sentences.append(sentence)
            
            if relevant_sentences:
                snippet = '. '.join(relevant_sentences[:2])  # Take first 2 relevant sentences
                if snippet:
                    relevant_snippets.append(f"• {snippet}")
        
        if not relevant_snippets:
            # Fallback: use raw content from first node
            first_node_text = retrieved_nodes[0].node.text if hasattr(retrieved_nodes[0], 'node') else str(retrieved_nodes[0])
            snippet = first_node_text[:300].strip()
            if snippet:
                relevant_snippets.append(f"• {snippet}")
        
        # Construct response
        response_parts = [
            "Here's what I found regarding your question:",
            "",
        ]
        
        response_parts.extend(relevant_snippets)
        
        response_parts.extend([
            "",
            "Note: I'm currently experiencing technical difficulties with detailed responses.",
            "For the most current information, you may also want to check:",
            "- The university's official website",
            "- JAMB's official resources",
            "- Contact the admissions office directly"
        ])
        
        return '\n'.join(response_parts)


class EduBot:
    """
    Main EduBot class that orchestrates the complete chat system.
    """
    
    def __init__(self):
        """Initialize EduBot with all required components."""
        logger.info("Initializing EduBot...")
        
        # Initialize models
        self.llm = GoogleGenAI(model=METADATA_EXTRACTOR_MODEL, api_key=GOOGLE_API_KEY)
        self.embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
        
        # Initialize components
        self.query_router = QueryRouter()
        self.response_generator = ResponseGenerator(self.llm)
        self.result_ranker = ResultRanker(relevance_threshold=0.2)  # Lower threshold for more lenient filtering
        
        # Session management - store conversations per session
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.session_manager = SessionManager()
        
        # Initialize retrieval system
        try:
            self.retrieval_manager = RetrievalManager(
                qdrant_path=QDRANT_DB_PATH,
                collection_name=QDRANT_COLLECTION_NAME,
                docstore_path=PARENT_DOCSTORE_DIR,
                embed_model=self.embed_model
            )
            self.knowledge_base_available = True
            logger.info("✅ Knowledge base loaded successfully")
        except Exception as e:
            logger.warning(f"Knowledge base not available: {e}")
            self.retrieval_manager = None
            self.knowledge_base_available = False
        
        logger.info("✅ EduBot initialized successfully")
    
    def chat(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a chat message and return response.
        
        Args:
            message: User's message
            session_id: Session identifier for conversation tracking
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing message: {message[:100]}... (Session: {session_id})")
        
        try:
            # Get or create conversation context for this session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = ConversationContext()
                # Try to load existing conversation
                conversation_data = self.session_manager.load_conversation(session_id)
                if conversation_data:
                    self._restore_conversation_context(session_id, conversation_data)
                    logger.info(f"📂 Restored conversation context for session: {session_id}")
            
            conversation_context = self.active_sessions[session_id]
            
            # Classify the query
            classification = self.query_router.classify_query(message)
            
            # Determine response strategy
            response_data = self._generate_response(message, classification, conversation_context)
            
            # Add to conversation context
            conversation_context.add_exchange(
                user_query=message,
                bot_response=response_data['response'],
                query_classification=classification
            )
            
            # Save conversation periodically (every 3 messages)
            if len(conversation_context.history) % 3 == 0:
                self._save_session_context(session_id, conversation_context)
            
            # Prepare final response
            result = {
                'response': response_data['response'],
                'query_type': classification['primary_type'].value,
                'confidence': classification['confidence'],
                'knowledge_base_used': response_data.get('knowledge_base_used', False),
                'sources_count': response_data.get('sources_count', 0),
                'session_id': session_id,
                'conversation_context': conversation_context.get_context_summary(),
                'message_count': len(conversation_context.history)
            }
            
            logger.info(f"Response generated - Type: {result['query_type']}, KB Used: {result['knowledge_base_used']}, Session: {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                'response': "I'm sorry, I encountered an error processing your request. Please try again.",
                'query_type': 'error',
                'confidence': 0.0,
                'knowledge_base_used': False,
                'sources_count': 0,
                'session_id': session_id,
                'error': str(e)
            }
    
    def _generate_response(self, message: str, classification: Dict[str, Any], 
                          conversation_context: ConversationContext) -> Dict[str, Any]:
        """Generate response based on query classification."""
        query_type = classification['primary_type']
        requires_kb = classification['requires_knowledge_base']
        
        if requires_kb and self.knowledge_base_available:
            return self._generate_knowledge_based_response(message, classification, conversation_context)
        else:
            return self._generate_general_response(message, classification, conversation_context)
    
    def _generate_knowledge_based_response(self, message: str, 
                                         classification: Dict[str, Any],
                                         conversation_context: ConversationContext) -> Dict[str, Any]:
        """Generate response using knowledge base."""
        try:
            # Retrieve relevant information
            retrieved_nodes = self.retrieval_manager.retrieve(
                query=message,
                strategy='auto_merging',
                top_k=5
            )
            
            # Rank and filter results
            ranked_nodes = self.result_ranker.rank_and_filter(retrieved_nodes, message)
            
            # Generate response
            context = conversation_context.get_context_summary()
            response = self.response_generator.generate_knowledge_based_response(
                query=message,
                retrieved_nodes=ranked_nodes,
                context=context
            )
            
            return {
                'response': response,
                'knowledge_base_used': True,
                'sources_count': len(ranked_nodes),
                'retrieval_scores': [node.score for node in ranked_nodes]
            }
            
        except Exception as e:
            logger.error(f"Knowledge-based response generation failed: {e}")
            # Fallback to general response
            return self._generate_general_response(message, classification, conversation_context)
    
    def _generate_general_response(self, message: str, 
                                 classification: Dict[str, Any],
                                 conversation_context: ConversationContext) -> Dict[str, Any]:
        """Generate response without knowledge base."""
        query_type = classification['primary_type']
        context = conversation_context.get_context_summary()
        
        response = self.response_generator.generate_general_response(
            query=message,
            query_type=query_type,
            context=context
        )
        
        return {
            'response': response,
            'knowledge_base_used': False,
            'sources_count': 0
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status information about the bot system."""
        status = {
            'knowledge_base_available': self.knowledge_base_available,
            'active_sessions': len(self.active_sessions),
            'available_sessions': self.session_manager.get_available_sessions()
        }
        
        if self.knowledge_base_available:
            status['retrieval_stats'] = self.retrieval_manager.get_retrieval_stats()
        
        return status
    
    def clear_conversation(self, session_id: str = "default"):
        """Clear conversation history for a specific session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].clear_history()
            # Delete the saved session file
            self.session_manager.delete_session(session_id)
            logger.info(f"Conversation history cleared for session: {session_id}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        return self.session_manager.get_session_info(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all available conversation sessions."""
        return self.session_manager.get_available_sessions()
    
    def export_session(self, session_id: str, output_path: str) -> bool:
        """Export a specific session conversation."""
        conversation_data = self.session_manager.load_conversation(session_id)
        if conversation_data:
            try:
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                logger.error(f"Failed to export session {session_id}: {e}")
        return False
    
    def _restore_conversation_context(self, session_id: str, conversation_data: Dict[str, Any]):
        """Restore conversation context from saved data."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationContext()
        
        context = self.active_sessions[session_id]
        history = conversation_data.get('history', [])
        
        # Restore conversation history
        for exchange in history:
            context.history.append(exchange)
        
        # Update current topic from summary
        summary = conversation_data.get('summary', {})
        context.current_topic = summary.get('current_topic')
    
    def _save_session_context(self, session_id: str, conversation_context: ConversationContext):
        """Save conversation context to disk."""
        try:
            summary = {
                'current_topic': conversation_context.current_topic,
                'message_count': len(conversation_context.history),
                'last_updated': datetime.now().isoformat()
            }
            
            self.session_manager.save_conversation(
                conversation_history=conversation_context.history,
                summary=summary
            )
        except Exception as e:
            logger.error(f"Failed to save session context for {session_id}: {e}")
