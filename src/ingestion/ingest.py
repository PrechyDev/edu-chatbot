"""
Core ingestion orchestrator for the EduBot knowledge base.
Coordinates document loading, metadata extraction, processing, and storage.
"""

import logging
from typing import Dict, Any, Optional, List

from llama_index.core import Document
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from .loaders import DocumentLoaderFactory
from .processors import DocumentProcessor, ChunkOptimizer
from .storage import StorageManager
from .metadata_extractor import create_metadata_extractor
from src.config import (
    GOOGLE_API_KEY,
    METADATA_EXTRACTOR_MODEL,
    EMBEDDING_MODEL,
    PARENT_DOCSTORE_DIR,
    QDRANT_DB_PATH,
    QDRANT_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

class IngestionOrchestrator:
    """
    Orchestrates the complete document ingestion pipeline.
    Coordinates loading, metadata extraction, processing, and storage.
    """
    
    def __init__(self):
        """Initialize the ingestion orchestrator with all required components."""
        # Initialize components
        self.document_loader = DocumentLoaderFactory()
        self.processor = DocumentProcessor()
        self.storage_manager = StorageManager(
            qdrant_path=QDRANT_DB_PATH,
            collection_name=QDRANT_COLLECTION_NAME,
            docstore_path=PARENT_DOCSTORE_DIR
        )
        self.metadata_extractor = create_metadata_extractor()
        
        # Initialize models
        self.llm = GoogleGenAI(model=METADATA_EXTRACTOR_MODEL, api_key=GOOGLE_API_KEY)
        self.embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
        
        logger.info("✅ IngestionOrchestrator initialized successfully")
    
    def ingest(self, input_source: str, additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline for a given source.
        
        Args:
            input_source: Path to file/directory or URL to ingest
            additional_metadata: Optional additional metadata to apply
            
        Returns:
            Dictionary with ingestion results and statistics
        """
        logger.info(f"🚀 Starting ingestion for source: {input_source}")
        
        try:
            # Step 1: Load documents
            documents = self._load_documents(input_source)
            
            # Step 2: Extract metadata
            documents_with_metadata = self._extract_metadata(documents, additional_metadata)
            
            # Step 3: Process documents into chunks
            processed_nodes = self._process_documents(documents_with_metadata)
            
            # Step 4: Store in knowledge base
            storage_results = self._store_documents(processed_nodes)
            
            # Step 5: Finalize and persist
            persistence_results = self._finalize_ingestion()
            
            # Compile results
            results = {
                'status': 'success',
                'source': input_source,
                'documents_loaded': len(documents),
                'documents_with_metadata': len(documents_with_metadata),
                'total_nodes_created': len(processed_nodes['all_nodes']),
                'leaf_nodes_created': len(processed_nodes['leaf_nodes']),
                'storage': storage_results,
                'persistence': persistence_results
            }
            
            logger.info("✅ Ingestion completed successfully")
            self._log_completion_summary(results)
            return results
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {str(e)}")
            raise
    
    def _load_documents(self, input_source: str) -> List[Document]:
        """Load documents from the input source."""
        logger.info(f"📂 Loading documents from: {input_source}")
        
        documents = self.document_loader.load_documents(input_source)
        
        if not documents:
            raise ValueError(f"No documents could be loaded from source: {input_source}")
        
        # Optimize chunk sizes based on document characteristics
        characteristics = ChunkOptimizer.analyze_document_characteristics(documents)
        optimal_chunks = ChunkOptimizer.get_optimal_chunk_sizes(documents)
        
        # Update processor with optimal chunk sizes
        if optimal_chunks != self.processor.chunk_sizes:
            logger.info(f"📊 Adjusting chunk sizes based on document analysis: {optimal_chunks}")
            self.processor = DocumentProcessor(chunk_sizes=optimal_chunks)
        
        logger.info(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def _extract_metadata(self, documents: List[Document], 
                         additional_metadata: Dict[str, Any] = None) -> List[Document]:
        """Extract metadata from documents."""
        logger.info("🔍 Extracting metadata from documents...")
        
        successful_extractions = 0
        fallback_extractions = 0
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc.metadata.get('file_path', 'unknown')}")
            
            # Extract metadata
            extraction_response = self.metadata_extractor.extract_metadata(doc, additional_metadata)
            
            if extraction_response.success:
                # Store metadata in document
                metadata = extraction_response.metadata
                serialized_metadata = self.metadata_extractor.serialize_metadata(metadata)
                doc.metadata['edubot_metadata'] = serialized_metadata
                doc.metadata['document_id'] = metadata.document_id
                
                if extraction_response.fallback_used:
                    fallback_extractions += 1
                    logger.warning(f"Used fallback metadata extraction for {doc.metadata.get('file_path', 'unknown')}")
                else:
                    successful_extractions += 1
                    logger.info(f"Successfully extracted metadata using LLM for {doc.metadata.get('file_path', 'unknown')}")
            else:
                logger.error(f"Failed to extract metadata for {doc.metadata.get('file_path', 'unknown')}: {extraction_response.error_message}")
                raise ValueError(f"Metadata extraction failed for document: {extraction_response.error_message}")
        
        logger.info(f"✅ Metadata extraction complete - LLM: {successful_extractions}, Fallback: {fallback_extractions}")
        return documents
    
    def _process_documents(self, documents: List[Document]) -> Dict[str, List]:
        """Process documents into hierarchical chunks."""
        logger.info("⚙️ Processing documents into chunks...")
        
        processed_nodes = self.processor.process_documents(documents)
        
        logger.info(f"✅ Created {len(processed_nodes['all_nodes'])} total nodes, {len(processed_nodes['leaf_nodes'])} leaf nodes")
        return processed_nodes
    
    def _store_documents(self, processed_nodes: Dict[str, List]) -> Dict[str, Any]:
        """Store processed documents in the knowledge base."""
        logger.info("💾 Storing documents in knowledge base...")
        
        # Initialize storage
        storage_status = self.storage_manager.initialize_storage(self.embed_model)
        
        # Add nodes to storage
        storage_results = self.storage_manager.add_nodes(
            all_nodes=processed_nodes['all_nodes'],
            leaf_nodes=processed_nodes['leaf_nodes'],
            embed_model=self.embed_model
        )
        
        storage_results['initialization'] = storage_status
        logger.info("✅ Documents stored successfully")
        return storage_results
    
    def _finalize_ingestion(self) -> Dict[str, Any]:
        """Finalize ingestion and persist storage."""
        logger.info("🏁 Finalizing ingestion...")
        
        persistence_results = self.storage_manager.persist()
        
        logger.info("✅ Ingestion finalized and persisted")
        return persistence_results
    
    def _log_completion_summary(self, results: Dict[str, Any]):
        """Log a summary of the ingestion results."""
        logger.info("=" * 60)
        logger.info("📊 INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📄 Source: {results['source']}")
        logger.info(f"📚 Documents loaded: {results['documents_loaded']}")
        logger.info(f"🔍 Documents with metadata: {results['documents_with_metadata']}")
        logger.info(f"⚙️ Total nodes created: {results['total_nodes_created']}")
        logger.info(f"🍃 Leaf nodes for search: {results['leaf_nodes_created']}")
        logger.info(f"💾 Final knowledge base size: {results['storage']['docstore_total_nodes']} nodes")
        logger.info("🎯 Knowledge base ready for queries!")
        logger.info("=" * 60)


# Legacy function for backward compatibility
def run_ingestion(input_source: str, additional_metadata: dict = None):
    """
    Legacy wrapper function for backward compatibility.
    
    Args:
        input_source: Path to file/directory or URL to ingest
        additional_metadata: Optional additional metadata to apply
        
    Returns:
        Tuple of (vector_index, storage_context) for compatibility
    """
    orchestrator = IngestionOrchestrator()
    results = orchestrator.ingest(input_source, additional_metadata)
    
    # Return legacy format for compatibility
    return orchestrator.storage_manager.vector_index, None
