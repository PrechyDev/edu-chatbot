# Description: Functions to handle document updates and metadata changes
# =================================================================

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

from .metadata_schema import EduBotMetadata
from ..config import (
    PARENT_DOCSTORE_DIR,
    QDRANT_DB_PATH,
    QDRANT_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


class DocumentUpdateManager:
    """Handles updates to document metadata and re-indexing when needed."""
    
    def __init__(self):
        self.docstore = SimpleDocumentStore()
        try:
            self.docstore.persist(persist_path=PARENT_DOCSTORE_DIR)
        except Exception as e:
            logger.warning(f"Could not load existing docstore: {e}")
        
        self.qdrant_client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=QDRANT_COLLECTION_NAME
        )
    
    def update_document_metadata(self, document_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update document metadata after human review.
        
        Args:
            document_id: UUID of the document to update
            metadata_updates: Dictionary with metadata fields to update
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Get the document from docstore
            doc = self.docstore.get_document(document_id)
            if not doc:
                logger.error(f"Document {document_id} not found in docstore")
                return False
            
            # Get current metadata
            current_metadata = doc.metadata.get('edubot_metadata', {})
            
            # Update the metadata
            updated_metadata = self._merge_metadata_updates(current_metadata, metadata_updates)
            
            # Update version and timestamps
            updated_metadata['processing']['version'] += 1
            updated_metadata['processing']['last_modified'] = datetime.now().date()
            updated_metadata['processing']['human_verified'] = True
            updated_metadata['processing']['last_reviewed'] = datetime.now().date()
            
            # Store updated metadata back in document
            doc.metadata['edubot_metadata'] = updated_metadata
            
            # Persist the updated docstore
            self.docstore.persist(persist_path=PARENT_DOCSTORE_DIR)
            
            # Check if we need to update vector embeddings
            if self._needs_reindexing(metadata_updates):
                self._update_vector_embeddings(document_id, doc)
            
            logger.info(f"Successfully updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def _merge_metadata_updates(self, current_metadata: Dict, updates: Dict) -> Dict:
        """Merge new updates into existing metadata."""
        if isinstance(current_metadata, dict):
            merged = current_metadata.copy()
        else:
            # If current_metadata is a Pydantic model, convert to dict
            merged = current_metadata.dict() if hasattr(current_metadata, 'dict') else {}
        
        # Deep merge the updates
        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        
        return merged
    
    def _needs_reindexing(self, updates: Dict) -> bool:
        """Determine if changes require re-embedding the document."""
        # Fields that affect search should trigger re-indexing
        reindex_fields = {
            'title', 'summary', 'keywords', 'common_queries', 
            'classification', 'search_optimization'
        }
        
        for field in reindex_fields:
            if field in updates:
                return True
        return False
    
    def _update_vector_embeddings(self, document_id: str, doc) -> bool:
        """Update vector embeddings for a document."""
        try:
            # Delete old embeddings from vector store
            self.vector_store.delete(document_id)
            
            # Re-chunk and re-embed the document
            # This would typically involve re-running the embedding pipeline
            # For now, we'll just log that this needs to be done
            logger.info(f"Document {document_id} marked for re-embedding")
            
            # In a full implementation, you'd:
            # 1. Re-chunk the document
            # 2. Generate new embeddings
            # 3. Store in vector database
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update embeddings for {document_id}: {e}")
            return False
    
    def mark_for_reprocessing(self, document_id: str) -> bool:
        """Mark a document for complete reprocessing."""
        try:
            doc = self.docstore.get_document(document_id)
            if not doc:
                return False
            
            metadata = doc.metadata.get('edubot_metadata', {})
            if 'processing' in metadata:
                metadata['processing']['needs_reprocessing'] = True
                metadata['processing']['last_modified'] = datetime.utcnow()
            
            doc.metadata['edubot_metadata'] = metadata
            self.docstore.persist(persist_path=PARENT_DOCSTORE_DIR)
            
            logger.info(f"Document {document_id} marked for reprocessing")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark document {document_id} for reprocessing: {e}")
            return False
    
    def get_documents_needing_reprocessing(self) -> list:
        """Get list of documents that need reprocessing."""
        try:
            docs_to_reprocess = []
            
            # Get all documents from docstore
            for doc_id in self.docstore.get_all_document_ids():
                doc = self.docstore.get_document(doc_id)
                metadata = doc.metadata.get('edubot_metadata', {})
                
                if (metadata.get('processing', {}).get('needs_reprocessing', False) or
                    not metadata.get('processing', {}).get('human_verified', False)):
                    docs_to_reprocess.append(doc_id)
            
            return docs_to_reprocess
            
        except Exception as e:
            logger.error(f"Failed to get documents for reprocessing: {e}")
            return []


# Convenience functions for admin dashboard
def update_document_metadata(document_id: str, metadata_updates: Dict[str, Any]) -> bool:
    """Update document metadata - convenience function for admin dashboard."""
    manager = DocumentUpdateManager()
    return manager.update_document_metadata(document_id, metadata_updates)

def mark_document_for_review(document_id: str) -> bool:
    """Mark document as needing human review."""
    manager = DocumentUpdateManager()
    return manager.mark_for_reprocessing(document_id)
