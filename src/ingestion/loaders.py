"""
Document loaders for different source types.
Each loader handles a specific type of input source and returns standardized Document objects.
"""

import logging
import os
import requests
from typing import List, Optional
from abc import ABC, abstractmethod

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.web import BeautifulSoupWebReader
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """Check if this loader can handle the given source."""
        pass
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """Load documents from the source."""
        pass


class URLLoader(DocumentLoader):
    """Loads documents from web URLs."""
    
    def can_handle(self, source: str) -> bool:
        return source.startswith(('http://', 'https://'))
    
    def load(self, source: str) -> List[Document]:
        logger.info(f"Loading URL: {source}")
        
        if not self.can_handle(source):
            raise ValueError(f"Invalid URL format: {source}")
        
        # First test if URL is accessible
        try:
            response = requests.get(source, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code != 200:
                raise ValueError(f"URL returned status {response.status_code}: {source}")
            logger.info(f"URL is accessible, content length: {len(response.text)}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Cannot access URL {source}: {str(e)}")
        
        # Try with BeautifulSoupWebReader
        reader = BeautifulSoupWebReader()
        try:
            documents = reader.load_data(urls=[source])
            if not documents:
                # Fallback: create document from requests content
                logger.warning("BeautifulSoupWebReader returned no documents, using fallback")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                text_content = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if not text or len(text) < 100:
                    raise ValueError(f"No meaningful content extracted from URL: {source}")
                
                # Create document manually
                documents = [Document(
                    text=text,
                    metadata={
                        'source_url': source,
                        'source_type': 'url',
                        'extraction_method': 'fallback_requests'
                    }
                )]
            
            # Add source URL to metadata for BeautifulSoupWebReader results
            else:
                for doc in documents:
                    doc.metadata['source_url'] = source
                    doc.metadata['source_type'] = 'url'
            
            # Log first 100 characters of content
            for i, doc in enumerate(documents):
                content_preview = doc.text[:100].replace('\n', ' ').strip()
                logger.info(f"Document {i+1} content preview: {content_preview}...")
            
            logger.info(f"Successfully loaded {len(documents)} document(s) from URL")
            return documents
            
        except Exception as e:
            # If BeautifulSoupWebReader fails, try fallback with requests
            logger.warning(f"BeautifulSoupWebReader failed: {e}, trying fallback")
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                text_content = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if not text or len(text) < 100:
                    raise ValueError(f"No meaningful content extracted from URL: {source}")
                
                # Create document manually
                documents = [Document(
                    text=text,
                    metadata={
                        'source_url': source,
                        'source_type': 'url',
                        'extraction_method': 'fallback_requests'
                    }
                )]
                
                # Log first 100 characters of content
                content_preview = text[:100].replace('\n', ' ').strip()
                logger.info(f"Fallback document content preview: {content_preview}...")
                
                logger.info(f"Successfully loaded {len(documents)} document(s) from URL using fallback")
                return documents
                
            except Exception as fallback_error:
                error_msg = self._categorize_url_error(str(fallback_error), source)
                logger.error(error_msg)
                raise ValueError(error_msg)
    
    def _categorize_url_error(self, error_str: str, source: str) -> str:
        """Categorize URL loading errors for better user feedback."""
        error_lower = error_str.lower()
        
        if "not a valid url" in error_lower:
            return f"Invalid or inaccessible URL: {source}. Please check the URL and try again."
        elif "connection" in error_lower or "network" in error_lower:
            return f"Network error accessing URL: {source}. Please check your internet connection."
        elif "timeout" in error_lower:
            return f"Timeout accessing URL: {source}. The server may be slow or unavailable."
        elif "404" in error_lower or "not found" in error_lower:
            return f"URL not found: {source}. Please verify the URL is correct."
        else:
            return f"Failed to load content from URL {source}: {error_str}"


class FileLoader(DocumentLoader):
    """Loads documents from local files."""
    
    def can_handle(self, source: str) -> bool:
        return os.path.isfile(source)
    
    def load(self, source: str) -> List[Document]:
        logger.info(f"Loading file: {source}")
        
        if not self.can_handle(source):
            raise ValueError(f"File not found: {source}")
        
        try:
            reader = SimpleDirectoryReader(input_files=[source])
            documents = reader.load_data()
            
            if not documents:
                raise ValueError(f"No content could be extracted from file: {source}")
            
            # Merge multi-page documents
            merged_doc = self._merge_document_pages(documents, source)
            
            # Log first 100 characters of content
            content_preview = merged_doc.text[:100].replace('\n', ' ').strip()
            logger.info(f"File document content preview: {content_preview}...")
            
            logger.info(f"Successfully loaded and merged document from file")
            return [merged_doc]
            
        except Exception as e:
            error_msg = f"Failed to load file {source}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _merge_document_pages(self, documents: List[Document], file_path: str) -> Document:
        """Merge multiple pages from a single file into one document."""
        if len(documents) == 1:
            documents[0].metadata['source_type'] = 'file'
            return documents[0]
        
        # Merge content from all pages
        full_text = "\n\n".join(doc.text for doc in documents)
        merged_metadata = documents[0].metadata.copy()
        merged_metadata.update({
            'file_path': file_path,
            'source_type': 'file',
            'page_count': len(documents)
        })
        
        logger.info(f"Merged {len(documents)} pages into single document")
        return Document(text=full_text, metadata=merged_metadata)


class DirectoryLoader(DocumentLoader):
    """Loads documents from directories."""
    
    def can_handle(self, source: str) -> bool:
        return os.path.isdir(source)
    
    def load(self, source: str) -> List[Document]:
        logger.info(f"Loading directory: {source}")
        
        if not self.can_handle(source):
            raise ValueError(f"Directory not found: {source}")
        
        try:
            reader = SimpleDirectoryReader(input_dir=source, recursive=True)
            all_docs = reader.load_data()
            
            if not all_docs:
                raise ValueError(f"No documents found in directory: {source}")
            
            # Group documents by file path and merge pages
            merged_documents = self._group_and_merge_documents(all_docs)
            
            # Log first 100 characters of each document
            for i, doc in enumerate(merged_documents):
                content_preview = doc.text[:100].replace('\n', ' ').strip()
                file_path = doc.metadata.get('file_path', 'unknown')
                logger.info(f"Directory document {i+1} ({file_path}) content preview: {content_preview}...")
            
            logger.info(f"Successfully loaded {len(merged_documents)} documents from directory")
            return merged_documents
            
        except Exception as e:
            error_msg = f"Failed to load directory {source}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _group_and_merge_documents(self, documents: List[Document]) -> List[Document]:
        """Group documents by file path and merge pages from the same file."""
        file_to_docs = {}
        
        for doc in documents:
            file_path = doc.metadata.get('file_path', 'unknown_file')
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)
        
        merged_documents = []
        for file_path, docs in file_to_docs.items():
            if len(docs) == 1:
                docs[0].metadata['source_type'] = 'file'
                merged_documents.append(docs[0])
            else:
                # Merge multiple pages
                full_text = "\n\n".join(doc.text for doc in docs)
                merged_metadata = docs[0].metadata.copy()
                merged_metadata.update({
                    'file_path': file_path,
                    'source_type': 'file',
                    'page_count': len(docs)
                })
                merged_doc = Document(text=full_text, metadata=merged_metadata)
                merged_documents.append(merged_doc)
                logger.info(f"Merged {len(docs)} pages for file: {file_path}")
        
        return merged_documents


class DocumentLoaderFactory:
    """Factory class to get the appropriate loader for a source."""
    
    def __init__(self):
        self.loaders = [
            URLLoader(),
            FileLoader(),
            DirectoryLoader()
        ]
    
    def get_loader(self, source: str) -> DocumentLoader:
        """Get the appropriate loader for the given source."""
        for loader in self.loaders:
            if loader.can_handle(source):
                return loader
        
        raise ValueError(f"No loader found for source: {source}")
    
    def load_documents(self, source: str) -> List[Document]:
        """Load documents from any supported source type."""
        loader = self.get_loader(source)
        return loader.load(source)
