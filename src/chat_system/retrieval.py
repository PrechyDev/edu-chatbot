"""
Advanced retrieval system for the EduBot knowledge base.
Handles different retrieval strategies and result ranking.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import qdrant_client

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Retrieve relevant nodes for the query."""
        pass


class VectorRetrieval(RetrievalStrategy):
    """Standard vector similarity retrieval."""
    
    def __init__(self, vector_index: VectorStoreIndex):
        self.vector_index = vector_index
        self.retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=10  # Retrieve more for reranking
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Retrieve using vector similarity."""
        nodes = self.retriever.retrieve(query)
        return nodes[:top_k]


class HybridRetrieval(RetrievalStrategy):
    """Hybrid retrieval combining vector and keyword search."""
    
    def __init__(self, vector_index: VectorStoreIndex, docstore: SimpleDocumentStore):
        self.vector_index = vector_index
        self.docstore = docstore
        self.vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=8
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Retrieve using hybrid approach."""
        # Get vector results
        vector_nodes = self.vector_retriever.retrieve(query)
        
        # For now, return vector results (can be enhanced with keyword search)
        return vector_nodes[:top_k]


class AutoMergingRetrieval(RetrievalStrategy):
    """Auto-merging retrieval for hierarchical chunks."""
    
    def __init__(self, vector_index: VectorStoreIndex, storage_context: StorageContext):
        self.storage_context = storage_context
        self.retriever = AutoMergingRetriever(
            vector_retriever=VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=12
            ),
            storage_context=storage_context,
            verbose=True
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Retrieve using auto-merging."""
        nodes = self.retriever.retrieve(query)
        return nodes[:top_k]


class RetrievalManager:
    """
    Manages different retrieval strategies and provides unified interface.
    """
    
    def __init__(self, qdrant_path: str, collection_name: str, 
                 docstore_path: str, embed_model: GoogleGenAIEmbedding):
        """
        Initialize the retrieval manager.
        
        Args:
            qdrant_path: Path to Qdrant database
            collection_name: Name of the Qdrant collection
            docstore_path: Path to document store
            embed_model: Embedding model for queries
        """
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.docstore_path = docstore_path
        self.embed_model = embed_model
        
        # Initialize components
        self.vector_index: Optional[VectorStoreIndex] = None
        self.storage_context: Optional[StorageContext] = None
        self.strategies: Dict[str, RetrievalStrategy] = {}
        
        # Initialize retrieval system
        self._initialize_retrieval_system()
        
        logger.info("RetrievalManager initialized successfully")
    
    def _initialize_retrieval_system(self):
        """Initialize the retrieval system components."""
        try:
            # Load docstore
            docstore = SimpleDocumentStore.from_persist_path(self.docstore_path)
            logger.info(f"Loaded docstore with {len(docstore.docs)} nodes")
            
            # Initialize Qdrant vector store
            qdrant_client_instance = qdrant_client.QdrantClient(path=self.qdrant_path)
            vector_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name=self.collection_name
            )
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                docstore=docstore,
                vector_store=vector_store
            )
            
            # Load vector index
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
            
            # Initialize retrieval strategies
            self.strategies = {
                'vector': VectorRetrieval(self.vector_index),
                'hybrid': HybridRetrieval(self.vector_index, docstore),
                'auto_merging': AutoMergingRetrieval(self.vector_index, self.storage_context)
            }
            
            logger.info("Retrieval system initialized with all strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval system: {e}")
            raise ValueError(f"Retrieval system initialization failed: {e}")
    
    def retrieve(self, query: str, strategy: str = 'auto_merging', 
                 top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            strategy: Retrieval strategy to use ('vector', 'hybrid', 'auto_merging')
            top_k: Number of top results to return
            
        Returns:
            List of relevant nodes with scores
        """
        if not self.vector_index:
            raise ValueError("Retrieval system not properly initialized")
        
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy '{strategy}', falling back to 'vector'")
            strategy = 'vector'
        
        try:
            logger.info(f"Retrieving with strategy '{strategy}' for query: {query[:100]}...")
            
            retrieval_strategy = self.strategies[strategy]
            nodes = retrieval_strategy.retrieve(query, top_k)
            
            logger.info(f"Retrieved {len(nodes)} nodes using {strategy} strategy")
            return nodes
            
        except Exception as e:
            logger.error(f"Retrieval failed with strategy '{strategy}': {e}")
            # Fallback to basic vector retrieval
            if strategy != 'vector':
                logger.info("Falling back to basic vector retrieval")
                return self.strategies['vector'].retrieve(query, top_k)
            raise
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        stats = {
            'available_strategies': list(self.strategies.keys()),
            'vector_index_initialized': self.vector_index is not None,
            'storage_context_initialized': self.storage_context is not None,
            'qdrant_path': self.qdrant_path,
            'collection_name': self.collection_name,
            'docstore_path': self.docstore_path
        }
        
        if self.storage_context and self.storage_context.docstore:
            stats['total_nodes'] = len(self.storage_context.docstore.docs)
        
        return stats


class ResultRanker:
    """
    Ranks and filters retrieval results based on relevance and quality.
    """
    
    def __init__(self, relevance_threshold: float = 0.5):
        """
        Initialize the result ranker.
        
        Args:
            relevance_threshold: Minimum relevance score to include results
        """
        self.relevance_threshold = relevance_threshold
        logger.info(f"ResultRanker initialized with threshold {relevance_threshold}")
    
    def rank_and_filter(self, nodes: List[NodeWithScore], 
                       query: str) -> List[NodeWithScore]:
        """
        Rank and filter retrieval results.
        
        Args:
            nodes: Retrieved nodes with scores
            query: Original query for relevance checking
            
        Returns:
            Filtered and ranked nodes
        """
        if not nodes:
            return nodes
        
        # Filter by relevance threshold
        filtered_nodes = [
            node for node in nodes 
            if node.score >= self.relevance_threshold
        ]
        
        if not filtered_nodes:
            logger.warning(f"No nodes above threshold {self.relevance_threshold}, returning top result")
            filtered_nodes = nodes[:1]
        
        # Additional ranking based on metadata quality
        ranked_nodes = self._rank_by_metadata_quality(filtered_nodes)
        
        logger.info(f"Ranked and filtered {len(nodes)} -> {len(ranked_nodes)} nodes")
        return ranked_nodes
    
    def _rank_by_metadata_quality(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Rank nodes by metadata quality indicators."""
        
        def get_quality_score(node: NodeWithScore) -> float:
            """Calculate quality score for a node."""
            base_score = node.score
            metadata = node.node.metadata
            
            # Boost score based on metadata presence and quality
            quality_boost = 0.0
            
            # Check for important metadata fields
            if metadata.get('document_id'):
                quality_boost += 0.05
            
            if metadata.get('edubot_metadata'):
                quality_boost += 0.1
            
            # Check for source type preference
            source_type = metadata.get('source_type', '')
            if source_type == 'file':
                quality_boost += 0.05  # Prefer file sources
            
            # Check content length (prefer substantial content)
            content_length = len(node.node.text)
            if content_length > 500:
                quality_boost += 0.05
            elif content_length < 100:
                quality_boost -= 0.05
            
            return base_score + quality_boost
        
        # Sort by quality score
        ranked_nodes = sorted(nodes, key=get_quality_score, reverse=True)
        
        return ranked_nodes
