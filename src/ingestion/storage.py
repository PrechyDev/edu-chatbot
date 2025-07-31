"""
Storage management for the knowledge base.
Handles vector storage, document storage, and persistence operations.
"""

import logging
import os
from typing import List, Optional, Dict, Any

import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages all storage operations for the knowledge base."""
    
    def __init__(self, qdrant_path: str, collection_name: str, docstore_path: str):
        """
        Initialize storage manager.
        
        Args:
            qdrant_path: Path to Qdrant database
            collection_name: Name of the Qdrant collection
            docstore_path: Path to document store
        """
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.docstore_path = docstore_path
        
        # Storage components
        self.docstore: Optional[SimpleDocumentStore] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.vector_index: Optional[VectorStoreIndex] = None
        
        logger.info(f"Initialized StorageManager - Qdrant: {qdrant_path}, Docstore: {docstore_path}")
    
    def initialize_storage(self, embed_model: GoogleGenAIEmbedding) -> Dict[str, Any]:
        """
        Initialize or load existing storage components.
        
        Args:
            embed_model: Embedding model for vector operations
            
        Returns:
            Dictionary with storage status information
        """
        logger.info("🔍 Initializing storage components...")
        
        # Initialize document store
        docstore_status = self._initialize_docstore()
        
        # Initialize vector store
        vector_status = self._initialize_vector_store()
        
        # Initialize vector index
        index_status = self._initialize_vector_index(embed_model)
        
        status = {
            'docstore': docstore_status,
            'vector_store': vector_status,
            'vector_index': index_status
        }
        
        logger.info(f"Storage initialization complete: {status}")
        return status
    
    def _initialize_docstore(self) -> Dict[str, Any]:
        """Initialize or load existing document store."""
        if os.path.exists(self.docstore_path):
            try:
                self.docstore = SimpleDocumentStore.from_persist_path(self.docstore_path)
                node_count = len(self.docstore.docs)
                logger.info(f"📚 Loaded existing docstore with {node_count} nodes")
                return {'status': 'loaded', 'node_count': node_count}
            except Exception as e:
                logger.warning(f"Failed to load existing docstore: {e}")
                self.docstore = SimpleDocumentStore()
                return {'status': 'created_new', 'error': str(e)}
        else:
            self.docstore = SimpleDocumentStore()
            logger.info("📝 Created new document store")
            return {'status': 'created_new'}
    
    def _initialize_vector_store(self) -> Dict[str, Any]:
        """Initialize or connect to existing vector store."""
        try:
            # Create Qdrant client
            qdrant_client_instance = qdrant_client.QdrantClient(path=self.qdrant_path)
            
            # Check if collection exists
            collection_exists = False
            if os.path.exists(self.qdrant_path):
                try:
                    collections = qdrant_client_instance.get_collections()
                    collection_exists = any(col.name == self.collection_name for col in collections.collections)
                except Exception:
                    pass
            
            # Create vector store
            self.vector_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name=self.collection_name
            )
            
            if collection_exists:
                logger.info(f"🗄️ Connected to existing Qdrant collection: {self.collection_name}")
                return {'status': 'connected', 'collection_exists': True}
            else:
                logger.info(f"🆕 Created new Qdrant collection: {self.collection_name}")
                return {'status': 'created_new', 'collection_exists': False}
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise ValueError(f"Vector store initialization failed: {e}")
    
    def _initialize_vector_index(self, embed_model: GoogleGenAIEmbedding) -> Dict[str, Any]:
        """Initialize vector index."""
        try:
            storage_context = StorageContext.from_defaults(
                docstore=self.docstore,
                vector_store=self.vector_store
            )
            
            # Try to load existing index
            try:
                self.vector_index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                    embed_model=embed_model
                )
                logger.info("📈 Loaded existing vector index")
                return {'status': 'loaded'}
            except Exception:
                # Create new index (will be populated later)
                self.vector_index = None
                logger.info("🆕 Will create new vector index")
                return {'status': 'will_create_new'}
                
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            raise ValueError(f"Vector index initialization failed: {e}")
    
    def add_nodes(self, all_nodes: List[BaseNode], leaf_nodes: List[BaseNode], 
                  embed_model: GoogleGenAIEmbedding) -> Dict[str, Any]:
        """
        Add new nodes to storage.
        
        Args:
            all_nodes: All hierarchical nodes to add to docstore
            leaf_nodes: Leaf nodes to add to vector index
            embed_model: Embedding model for vector operations
            
        Returns:
            Dictionary with operation results
        """
        logger.info(f"Adding {len(all_nodes)} nodes to docstore, {len(leaf_nodes)} to vector index")
        
        # Add all nodes to docstore
        self.docstore.add_documents(all_nodes)
        total_nodes = len(self.docstore.docs)
        logger.info(f"📚 Docstore now contains {total_nodes} total nodes")
        
        # Add leaf nodes to vector index
        storage_context = StorageContext.from_defaults(
            docstore=self.docstore,
            vector_store=self.vector_store
        )
        
        if self.vector_index is None:
            # Create new vector index
            self.vector_index = VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True
            )
            logger.info(f"🆕 Created new vector index with {len(leaf_nodes)} nodes")
            index_status = 'created_new'
        else:
            # Add to existing index
            self.vector_index.insert_nodes(leaf_nodes)
            logger.info(f"📈 Added {len(leaf_nodes)} nodes to existing vector index")
            index_status = 'updated_existing'
        
        return {
            'docstore_total_nodes': total_nodes,
            'added_nodes': len(all_nodes),
            'added_leaf_nodes': len(leaf_nodes),
            'index_status': index_status
        }
    
    def persist(self) -> Dict[str, Any]:
        """
        Persist all storage components to disk.
        
        Returns:
            Dictionary with persistence results
        """
        logger.info("💾 Persisting storage to disk...")
        
        try:
            # Persist docstore
            self.docstore.persist(persist_path=self.docstore_path)
            final_node_count = len(self.docstore.docs)
            
            logger.info(f"💾 Persisted knowledge base with {final_node_count} nodes to {self.docstore_path}")
            
            return {
                'status': 'success',
                'final_node_count': final_node_count,
                'docstore_path': self.docstore_path,
                'qdrant_path': self.qdrant_path
            }
            
        except Exception as e:
            logger.error(f"Failed to persist storage: {e}")
            raise ValueError(f"Storage persistence failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'docstore_nodes': len(self.docstore.docs) if self.docstore else 0,
            'qdrant_path': self.qdrant_path,
            'collection_name': self.collection_name,
            'docstore_path': self.docstore_path,
            'vector_index_initialized': self.vector_index is not None
        }
        
        return stats
