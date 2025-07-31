"""
Smart query routing logic for EduBot.
Determines when to use RAG vs general knowledge based on query analysis.
"""

import re
from typing import Tuple, List, Dict, Set, Optional, Any
from enum import Enum
from datetime import datetime

class QueryType(Enum):
    """Enumeration of query types."""
    GREETING = "greeting"
    INSTITUTIONAL = "institutional"
    GENERAL_KNOWLEDGE = "general_knowledge"
    EDUCATIONAL = "educational"
    EDUCATIONAL_FOLLOWUP = "educational_followup"
    SIMPLE = "simple"
    ERROR = "error"

class QueryRouter:
    """
    Smart router that determines whether to use RAG or general knowledge.
    Uses pattern matching and keyword analysis for classification.
    """
    
    def __init__(self):
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize pattern sets for query classification."""
        
        # Greeting patterns
        self.greetings: Set[str] = {
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'greetings', 'howdy', 'what\'s up', 'how are you', 'how do you do',
            'good day', 'morning', 'afternoon', 'evening'
        }
        
        # General knowledge topics (don't need RAG)
        self.general_topics: Set[str] = {
            'python', 'programming', 'software development', 'artificial intelligence',
            'machine learning', 'data science', 'web development', 'mobile development',
            'career advice', 'study tips', 'time management', 'general education',
            'coding', 'algorithms', 'data structures', 'computer science basics'
        }
        
        # Institutional keywords (high priority for RAG)
        self.institutional_keywords: Set[str] = {
            'university', 'college', 'admission', 'requirement', 'program', 'course',
            'degree', 'bachelor', 'master', 'phd', 'doctorate', 'transcript', 'gpa',
            'scholarship', 'tuition', 'fee', 'campus', 'dormitory', 'hostel',
            'registration', 'semester', 'academic', 'faculty', 'department',
            'curriculum', 'syllabus', 'examination', 'grade', 'credit', 'unit'
        }
        
        # Follow-up indicators
        self.followup_indicators: Set[str] = {
            'more', 'tell me', 'what about', 'how about', 'explain', 'clarify',
            'details', 'elaborate', 'expand', 'continue', 'further', 'additional'
        }
        
        # Gratitude patterns
        self.gratitude_patterns: Set[str] = {
            'thank', 'thanks', 'appreciate', 'grateful', 'cheers'
        }
    
    def should_use_rag(self, question: str) -> Tuple[bool, str]:
        """
        Determine if question should use RAG system or general knowledge.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (use_rag: bool, query_type: str)
        """
        question_lower = question.lower().strip()
        
        # Check for greetings first (highest priority)
        if self._is_greeting(question_lower):
            return False, QueryType.GREETING.value
        
        # Check for gratitude
        if self._is_gratitude(question_lower):
            return False, QueryType.SIMPLE.value
        
        # Check for institutional keywords (high priority for RAG)
        if self._has_institutional_keywords(question_lower):
            return True, QueryType.INSTITUTIONAL.value
        
        # Check for general programming/tech topics
        if self._is_general_topic(question_lower):
            return False, QueryType.GENERAL_KNOWLEDGE.value
        
        # Complex questions likely need RAG
        if self._is_complex_question(question):
            return True, QueryType.EDUCATIONAL.value
        
        # Default to simple for short, unclear queries
        return False, QueryType.SIMPLE.value

    def classify_query(self, question: str) -> Dict[str, Any]:
        """
        Classify a query and return detailed classification information.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with classification details
        """
        use_rag, query_type = self.should_use_rag(question)
        
        # Convert string query_type back to enum
        query_type_enum = None
        for qt in QueryType:
            if qt.value == query_type:
                query_type_enum = qt
                break
        
        if query_type_enum is None:
            query_type_enum = QueryType.SIMPLE
        
        # Calculate confidence based on clarity of classification
        confidence = self._calculate_confidence(question, query_type_enum)
        
        return {
            'primary_type': query_type_enum,
            'requires_knowledge_base': use_rag,
            'confidence': confidence,
            'raw_query': question
        }
    
    def _calculate_confidence(self, question: str, query_type: QueryType) -> float:
        """Calculate confidence score for the classification."""
        question_lower = question.lower()
        
        if query_type == QueryType.GREETING:
            # High confidence for clear greetings
            return 0.9 if any(greeting in question_lower for greeting in self.greetings) else 0.7
        
        elif query_type == QueryType.INSTITUTIONAL:
            # Count institutional keywords
            keyword_count = sum(1 for keyword in self.institutional_keywords if keyword in question_lower)
            return min(0.95, 0.5 + (keyword_count * 0.15))
        
        elif query_type == QueryType.GENERAL_KNOWLEDGE:
            # Count general topics
            topic_count = sum(1 for topic in self.general_topics if topic in question_lower)
            return min(0.9, 0.6 + (topic_count * 0.1))
        
        elif query_type == QueryType.EDUCATIONAL:
            # Medium confidence for educational queries
            return 0.7
        
        else:
            # Lower confidence for unclear queries
            return 0.5
    
    def _is_greeting(self, question: str) -> bool:
        """Check if the question is a greeting."""
        # Exact matches for greetings
        for greeting in self.greetings:
            if (question == greeting or 
                question.startswith(f"{greeting}!") or 
                question.startswith(f"{greeting}.")):
                return True
        return False
    
    def _is_gratitude(self, question: str) -> bool:
        """Check if the question is expressing gratitude."""
        return any(pattern in question for pattern in self.gratitude_patterns)
    
    def _has_institutional_keywords(self, question: str) -> bool:
        """Check if question contains institutional keywords."""
        return any(keyword in question for keyword in self.institutional_keywords)
    
    def _is_general_topic(self, question: str) -> bool:
        """Check if question is about general topics."""
        return any(topic in question for topic in self.general_topics)
    
    def _is_complex_question(self, question: str) -> bool:
        """Check if question is complex enough to warrant RAG."""
        return len(question.split()) > 3
    
    def is_followup_question(self, question: str) -> bool:
        """Check if question is a follow-up question."""
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in self.followup_indicators)
    
    def get_query_stats(self, questions: List[str]) -> Dict[str, int]:
        """Get statistics for a list of questions."""
        stats = {}
        for question in questions:
            _, query_type = self.should_use_rag(question)
            stats[query_type] = stats.get(query_type, 0) + 1
        return stats

class ConversationalQueryRouter(QueryRouter):
    """
    Enhanced router that considers conversation history for better routing decisions.
    """
    
    def should_use_rag_with_context(
        self, 
        message: str, 
        conversation_history: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Enhanced routing logic that considers conversation history.
        
        Args:
            message: Current message
            conversation_history: Previous conversation messages
            
        Returns:
            Tuple of (use_rag: bool, query_type: str)
        """
        # Get recent context (last 3 exchanges)
        recent_messages = conversation_history[-6:] if conversation_history else []
        
        # Check if we're in the middle of an educational conversation
        educational_context = any(
            self.should_use_rag(msg.get('content', ''))[0] 
            for msg in recent_messages 
            if msg.get('role') == 'user'
        )
        
        # Use standard router for current message
        use_rag, query_type = self.should_use_rag(message)
        
        # If we have educational context and current message is a follow-up
        if educational_context and query_type in [QueryType.SIMPLE.value, QueryType.GENERAL_KNOWLEDGE.value]:
            if self.is_followup_question(message):
                return True, QueryType.EDUCATIONAL_FOLLOWUP.value
        
        return use_rag, query_type


class ConversationContext:
    """
    Manages conversation history and context for better routing and responses.
    """
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, any]] = []
        self.max_history = max_history
        self.current_topic: Optional[str] = None
        self.context_summary: str = ""
    
    def add_exchange(self, user_query: str, bot_response: str, 
                    query_classification: Dict[str, any]):
        """Add a user-bot exchange to conversation history."""
        # Convert QueryType enum to string for JSON serialization
        query_type = query_classification.get('primary_type')
        if hasattr(query_type, 'value'):
            query_type = query_type.value
        
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'bot_response': bot_response,
            'query_type': query_type,
            'confidence': query_classification.get('confidence'),
            'used_kb': query_classification.get('requires_knowledge_base', False)
        }
        
        self.history.append(exchange)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Update current topic
        self._update_current_topic(query_classification)
    
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation for context."""
        if not self.history:
            return ""
        
        if len(self.history) == 1:
            return f"Previous query: {self.history[-1]['user_query']}"
        
        recent_topics = []
        for exchange in self.history[-3:]:  # Last 3 exchanges
            query_type = exchange.get('query_type', 'unknown')
            if hasattr(query_type, 'value'):
                query_type = query_type.value
            recent_topics.append(f"User asked about {query_type}")
        
        return f"Recent conversation context: {'; '.join(recent_topics)}"
    
    def _update_current_topic(self, classification: Dict[str, any]):
        """Update the current conversation topic based on classification."""
        query_type = classification.get('primary_type')
        if hasattr(query_type, 'value'):
            query_type = query_type.value
        
        if query_type in ['institutional', 'educational']:
            self.current_topic = query_type
        elif query_type in ['greeting', 'simple']:
            # Don't change topic for these
            pass
        else:
            self.current_topic = query_type
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.current_topic = None
        self.context_summary = ""
