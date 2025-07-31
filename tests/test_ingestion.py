"""
Test script for ingestion pipeline functionality.
Tests document loading, metadata extraction, processing, and storage.
"""

import sys
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ingestion.ingest import IngestionOrchestrator
from src.ingestion.loaders import URLLoader, FileLoader, DirectoryLoader
from src.ingestion.processors import DocumentProcessor, ChunkOptimizer
from src.ingestion.storage import StorageManager
from src.ingestion.metadata_extractor import create_metadata_extractor


class TestDocumentLoaders:
    """Test document loading functionality."""
    
    def test_url_loader_can_handle(self):
        """Test URL loader detection."""
        loader = URLLoader()
        assert loader.can_handle("https://example.com")
        assert loader.can_handle("http://example.com")
        assert not loader.can_handle("file.pdf")
        assert not loader.can_handle("/path/to/file")
    
    def test_file_loader_can_handle(self):
        """Test file loader detection."""
        loader = FileLoader()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"test content")
            tmp.flush()
            
            assert loader.can_handle(tmp.name)
            assert not loader.can_handle("https://example.com")
            
            os.unlink(tmp.name)
    
    def test_directory_loader_can_handle(self):
        """Test directory loader detection."""
        loader = DirectoryLoader()
        with tempfile.TemporaryDirectory() as tmpdir:
            assert loader.can_handle(tmpdir)
            assert not loader.can_handle("https://example.com")
            assert not loader.can_handle("nonexistent_directory")


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def test_chunk_optimizer_analysis(self):
        """Test document characteristics analysis."""
        # Mock document with different characteristics
        from llama_index.core import Document
        
        # Short document
        short_doc = Document(text="This is a short document.")
        characteristics = ChunkOptimizer.analyze_document_characteristics([short_doc])
        
        assert "total_length" in characteristics
        assert "avg_length" in characteristics
        assert "document_types" in characteristics
    
    def test_optimal_chunk_sizes(self):
        """Test optimal chunk size calculation."""
        from llama_index.core import Document
        
        # Create test documents
        short_doc = Document(text="Short" * 10)
        long_doc = Document(text="Long text " * 1000)
        
        optimal_sizes = ChunkOptimizer.get_optimal_chunk_sizes([short_doc, long_doc])
        
        assert isinstance(optimal_sizes, list)
        assert len(optimal_sizes) > 0
        assert all(isinstance(size, int) for size in optimal_sizes)


class TestMetadataExtraction:
    """Test metadata extraction functionality."""
    
    def test_metadata_extractor_creation(self):
        """Test metadata extractor initialization."""
        extractor = create_metadata_extractor()
        assert extractor is not None
        assert hasattr(extractor, 'extract_metadata')
    
    @patch('src.ingestion.metadata_extractor.GoogleGenAI')
    def test_metadata_extraction_fallback(self, mock_llm):
        """Test metadata extraction with fallback."""
        from llama_index.core import Document
        
        # Mock LLM to fail
        mock_llm.return_value.complete.side_effect = Exception("API Error")
        
        extractor = create_metadata_extractor()
        doc = Document(text="Test document about university admissions", 
                      metadata={"file_path": "test.pdf"})
        
        result = extractor.extract_metadata(doc)
        
        # Should succeed with fallback even if LLM fails
        assert result.success or result.fallback_used


class TestIngestionOrchestrator:
    """Test complete ingestion pipeline."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        orchestrator = IngestionOrchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, 'ingest')
    
    def test_document_loading_validation(self):
        """Test input validation for document loading."""
        orchestrator = IngestionOrchestrator()
        
        # Test invalid source
        with pytest.raises(ValueError):
            orchestrator._load_documents("nonexistent_source_12345")
    
    @patch('src.ingestion.loaders.BeautifulSoupWebReader')
    def test_url_ingestion_flow(self, mock_reader):
        """Test URL ingestion pipeline."""
        from llama_index.core import Document
        
        # Mock successful URL loading
        mock_reader.return_value.load_data.return_value = [
            Document(text="Test content from URL", metadata={"source_url": "https://test.com"})
        ]
        
        orchestrator = IngestionOrchestrator()
        
        # This should not raise an exception for valid URL format
        try:
            documents = orchestrator._load_documents("https://test.com")
            assert len(documents) > 0
        except Exception as e:
            # Should be a controlled error, not an import/init error
            assert "network" in str(e).lower() or "connection" in str(e).lower()


def run_ingestion_tests():
    """Run all ingestion tests."""
    print("🧪 Running Ingestion Pipeline Tests")
    print("=" * 50)
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    test_classes = [TestDocumentLoaders, TestDocumentProcessor, 
                   TestMetadataExtraction, TestIngestionOrchestrator]
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                print(f"  🔹 {method_name}", end=" ... ")
                try:
                    method = getattr(instance, method_name)
                    method()
                    print("✅ PASSED")
                    test_results['passed'] += 1
                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    test_results['failed'] += 1
                    test_results['errors'].append(f"{test_class.__name__}.{method_name}: {e}")
    
    print(f"\n🏁 Test Results:")
    print(f"   ✅ Passed: {test_results['passed']}")
    print(f"   ❌ Failed: {test_results['failed']}")
    
    if test_results['errors']:
        print(f"\n❌ Errors:")
        for error in test_results['errors']:
            print(f"   - {error}")
    
    return test_results


if __name__ == "__main__":
    run_ingestion_tests()
