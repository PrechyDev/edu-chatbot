"""
Document processing pipeline for chunking and node creation.
Handles the transformation of documents into searchable chunks with metadata preservation.
"""

import logging
from typing import List, Dict, Any

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents into hierarchical chunks suitable for vector storage."""
    
    def __init__(self, chunk_sizes: List[int] = None):
        """
        Initialize the document processor.
        
        Args:
            chunk_sizes: List of chunk sizes for hierarchical parsing.
                        Default: [6144, 3072, 1536] for comprehensive context.
        """
        self.chunk_sizes = chunk_sizes or [6144, 3072, 1536]
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes
        )
        logger.info(f"Initialized DocumentProcessor with chunk sizes: {self.chunk_sizes}")
    
    def process_documents(self, documents: List[Document]) -> Dict[str, List[BaseNode]]:
        """
        Process documents into hierarchical chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Dictionary containing 'all_nodes' and 'leaf_nodes'
        """
        logger.info(f"Processing {len(documents)} documents into hierarchical chunks")
        
        all_nodes = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc.metadata.get('file_path', 'unknown')}")
            
            # Parse document into hierarchical nodes
            doc_nodes = self.node_parser.get_nodes_from_documents([doc])
            
            # Preserve metadata in all nodes
            for node in doc_nodes:
                self._preserve_metadata(node, doc)
            
            all_nodes.extend(doc_nodes)
        
        # Get leaf nodes for vector indexing
        leaf_nodes = get_leaf_nodes(all_nodes)
        
        logger.info(f"Created {len(all_nodes)} total nodes, {len(leaf_nodes)} leaf nodes")
        
        return {
            'all_nodes': all_nodes,
            'leaf_nodes': leaf_nodes
        }
    
    def _preserve_metadata(self, node: BaseNode, source_doc: Document):
        """Preserve important metadata from source document in the node."""
        # Copy essential metadata
        preserved_fields = [
            'document_id', 'edubot_metadata', 'file_path', 
            'source_url', 'source_type', 'page_count'
        ]
        
        for field in preserved_fields:
            if field in source_doc.metadata:
                node.metadata[field] = source_doc.metadata[field]
        
        # Add processing information
        node.metadata['chunk_level'] = getattr(node, 'chunk_level', 'leaf')
        node.metadata['processed_at'] = str(source_doc.metadata.get('date_uploaded', 'unknown'))


class ChunkOptimizer:
    """Optimizes chunk sizes based on document characteristics."""
    
    @staticmethod
    def get_optimal_chunk_sizes(documents: List[Document]) -> List[int]:
        """
        Determine optimal chunk sizes based on document characteristics.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of optimal chunk sizes
        """
        avg_doc_length = sum(len(doc.text) for doc in documents) / len(documents)
        
        if avg_doc_length < 5000:
            # Small documents - use smaller chunks
            return [3072, 1536, 768]
        elif avg_doc_length > 20000:
            # Large documents - use larger chunks
            return [8192, 4096, 2048]
        else:
            # Medium documents - default sizes
            return [6144, 3072, 1536]
    
    @staticmethod
    def analyze_document_characteristics(documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze document characteristics for optimization.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary with document characteristics
        """
        if not documents:
            return {}
        
        doc_lengths = [len(doc.text) for doc in documents]
        
        characteristics = {
            'total_documents': len(documents),
            'avg_length': sum(doc_lengths) / len(doc_lengths),
            'min_length': min(doc_lengths),
            'max_length': max(doc_lengths),
            'total_characters': sum(doc_lengths),
            'document_types': list(set(doc.metadata.get('source_type', 'unknown') for doc in documents))
        }
        
        logger.info(f"Document characteristics: {characteristics}")
        return characteristics
